from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.core.models import (
    AnswerWithCitations,
    CacheKey,
    ComponentIdentification,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    QueryLogSource,
    RetrievedChunk,
)
from backend.services.history import QueryHistoryService


class FakeSnapshot:
    def __init__(self, doc_id: str, payload: dict[str, Any]) -> None:
        self.id = doc_id
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


class FakeQuery:
    def __init__(self, documents: list[FakeSnapshot]) -> None:
        self.documents = documents
        self.tenant_id: str | None = None
        self.component_serial: str | None = None
        self.limit_value: int | None = None

    def where(self, field: str, op: str, value: str) -> "FakeQuery":
        del op
        if field == "tenant_id":
            self.tenant_id = value
        if field == "component_serial":
            self.component_serial = value
        return self

    def order_by(self, field: str, direction: str) -> "FakeQuery":
        del field
        del direction
        return self

    def limit(self, value: int) -> "FakeQuery":
        self.limit_value = value
        return self

    def stream(self) -> list[FakeSnapshot]:
        items = [
            snapshot
            for snapshot in self.documents
            if (self.tenant_id is None or snapshot.payload["tenant_id"] == self.tenant_id)
            and (
                self.component_serial is None
                or snapshot.payload["component_serial"] == self.component_serial
            )
        ]
        items.sort(key=lambda snapshot: snapshot.payload["timestamp"], reverse=True)
        if self.limit_value is not None:
            items = items[: self.limit_value]
        return items


class FakeCollection:
    def __init__(self) -> None:
        self.added_payloads: list[dict[str, Any]] = []
        self.documents: list[FakeSnapshot] = []

    def add(self, payload: dict[str, Any]) -> None:
        self.added_payloads.append(payload)
        self.documents.append(FakeSnapshot(f"entry-{len(self.documents) + 1}", payload))

    def where(self, field: str, op: str, value: str) -> FakeQuery:
        return FakeQuery(self.documents).where(field, op, value)


class FakeFirestoreClient:
    def __init__(self) -> None:
        self.collection_instance = FakeCollection()

    def collection(self, name: str) -> FakeCollection:
        assert name == "query_log"
        return self.collection_instance


def test_record_answer_persists_expected_query_log_shape(test_settings) -> None:
    firestore_client = FakeFirestoreClient()
    history = QueryHistoryService(settings=test_settings, firestore_client=firestore_client)
    identification = ComponentIdentification(
        manufacturer="ABB",
        model_number="S202",
        part_number="2CDS252001R0404",
        component_type="Miniature circuit breaker",
        confidence_score=0.91,
        raw_ocr_text="ABB S202 2CDS252001R0404",
        should_attempt_document_lookup=True,
    )

    entry = history.record_answer(
        identification=identification,
        question="What is the rated current?",
        answer=AnswerWithCitations(
            answer_text="The rated current is 40 A.",
            citations=[],
            confidence=0.83,
        ),
    )

    assert entry is not None
    payload = firestore_client.collection_instance.added_payloads[0]
    assert payload["tenant_id"] == "public_demo"
    assert payload["component_serial"] == "ABB_S202_2CDS252001R0404"
    assert payload["component_model"] == "S202"
    assert payload["question"] == "What is the rated current?"
    assert payload["answer"] == "The rated current is 40 A."
    assert payload["source"] == QueryLogSource.CACHE.value
    assert payload["confidence"] == 83
    assert "timestamp" in payload


def test_get_component_history_returns_latest_entries(test_settings) -> None:
    firestore_client = FakeFirestoreClient()
    history = QueryHistoryService(settings=test_settings, firestore_client=firestore_client)
    firestore_client.collection_instance.documents = [
        FakeSnapshot(
            "entry-1",
            {
                "tenant_id": "public_demo",
                "component_serial": "ABB_S202_2CDS252001R0404",
                "component_model": "ABB S202",
                "question": "What is the rated current?",
                "answer": "40 A",
                "source": "web",
                "confidence": 96,
                "timestamp": datetime(2026, 3, 20, 9, 0, tzinfo=timezone.utc),
                "doc_source": "ABB datasheet",
            },
        ),
        FakeSnapshot(
            "entry-2",
            {
                "tenant_id": "public_demo",
                "component_serial": "ABB_S202_2CDS252001R0404",
                "component_model": "ABB S202",
                "question": "What standards apply?",
                "answer": "IEC 60898",
                "source": "web",
                "confidence": 94,
                "timestamp": datetime(2026, 3, 22, 9, 0, tzinfo=timezone.utc),
                "doc_source": "ABB datasheet",
            },
        ),
        FakeSnapshot(
            "entry-3",
            {
                "tenant_id": "another_tenant",
                "component_serial": "ABB_S202_2CDS252001R0404",
                "component_model": "ABB S202",
                "question": "Ignore me",
                "answer": "Ignore me",
                "source": "cache",
                "confidence": 10,
                "timestamp": datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
                "doc_source": None,
            },
        ),
    ]

    entries = history.get_component_history("ABB_S202_2CDS252001R0404", limit=5)

    assert [entry.id for entry in entries] == ["entry-2", "entry-1"]
    assert entries[0].question == "What standards apply?"
    assert entries[1].question == "What is the rated current?"


def test_record_answer_marks_cited_file_answers_as_private_docs(test_settings) -> None:
    firestore_client = FakeFirestoreClient()
    history = QueryHistoryService(settings=test_settings, firestore_client=firestore_client)
    identification = ComponentIdentification(
        manufacturer="ABB",
        model_number="S202",
        part_number="2CDS252001R0404",
        component_type="Miniature circuit breaker",
        confidence_score=0.91,
        raw_ocr_text="ABB S202 2CDS252001R0404",
        should_attempt_document_lookup=True,
    )
    metadata = DocumentMetadata(
        source_url="file:///private/ABB_manual.pdf",
        source_title="ABB_manual.pdf",
        manufacturer="ABB",
        model_number="S202",
        part_number="2CDS252001R0404",
        document_type=DocumentType.MANUAL,
        revision=None,
        page_map={0: 4},
        retrieved_at=datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc),
        content_hash="abc123",
        cache_key=CacheKey.from_parts("ABB", "S202", "2CDS252001R0404"),
    )
    answer = AnswerWithCitations(
        answer_text="The breaker is rated to IEC 60898.",
        citations=[
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_text="IEC 60898 applies.",
                    chunk_index=0,
                    metadata=metadata,
                    page_number=4,
                    section_title="Standards",
                ),
                similarity_score=0.91,
            )
        ],
        confidence=0.93,
    )

    entry = history.record_answer(
        identification=identification,
        question="What standards apply?",
        answer=answer,
    )

    assert entry is not None
    assert entry.source is QueryLogSource.PRIVATE_DOC
    assert entry.doc_source == "ABB_manual.pdf"
