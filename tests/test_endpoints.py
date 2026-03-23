from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from backend.agent.nodes import build_safe_answer
from backend.api.routes.history import get_history_service as get_history_route_service
from backend.api.routes.identify import (
    get_agent_runner as get_identify_agent_runner,
    get_vision_service,
)
from backend.api.routes.query import (
    get_agent_runner as get_query_agent_runner,
    get_history_service as get_query_history_service,
)
from backend.core.models import AnswerWithCitations, ComponentIdentification, QueryLogEntry, QueryLogSource
from backend.main import app, get_health_vector_store
from backend.services.vision import VisionIdentificationService

client = TestClient(app)


class FakeVisionService(VisionIdentificationService):
    def identify_component(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
    ) -> ComponentIdentification:
        del image_bytes
        del mime_type
        return ComponentIdentification(
            manufacturer="Siemens",
            model_number="SIMATIC S7-1200",
            part_number="6ES7214-1AG40-0XB0",
            component_type="PLC",
            confidence_score=0.92,
            raw_ocr_text="SIEMENS 6ES7214-1AG40-0XB0",
            should_attempt_document_lookup=True,
        )


class RunAgent:
    def __init__(self, answer: AnswerWithCitations) -> None:
        self.answer = answer

    def run(
        self,
        image_bytes: bytes,
        question: str,
        mime_type: str = "image/jpeg",
        identification: ComponentIdentification | None = None,
    ) -> AnswerWithCitations:
        del image_bytes
        del question
        del mime_type
        del identification
        return self.answer


class DetailedAgent:
    def __init__(
        self,
        answer: AnswerWithCitations,
        identification: ComponentIdentification | None = None,
        answer_from_identification: bool = False,
    ) -> None:
        self.answer = answer
        self.identification = identification
        self.answer_from_identification = answer_from_identification

    def run_detailed(
        self,
        image_bytes: bytes,
        question: str,
        mime_type: str = "image/jpeg",
        identification: ComponentIdentification | None = None,
    ) -> dict[str, object]:
        del image_bytes
        del question
        del mime_type
        return {
            "answer": self.answer,
            "identification": self.identification or identification,
            "answer_from_identification": self.answer_from_identification,
        }


class RaisingAgent:
    def run(
        self,
        image_bytes: bytes,
        question: str,
        mime_type: str = "image/jpeg",
        identification: ComponentIdentification | None = None,
    ) -> AnswerWithCitations:
        del image_bytes
        del question
        del mime_type
        del identification
        raise RuntimeError("provider failure")


class CapturingRunAgent:
    def __init__(self) -> None:
        self.question: str | None = None
        self.identification: ComponentIdentification | None = None

    def run(
        self,
        image_bytes: bytes,
        question: str,
        mime_type: str = "image/jpeg",
        identification: ComponentIdentification | None = None,
    ) -> AnswerWithCitations:
        del image_bytes
        del mime_type
        self.question = question
        self.identification = identification
        return build_safe_answer()


class FakeVectorStore:
    def __init__(self, healthy: bool) -> None:
        self.healthy = healthy

    def health_check(self) -> bool:
        return self.healthy


class HistoryServiceStub:
    def __init__(self) -> None:
        self.record_calls: list[tuple[ComponentIdentification | None, str, AnswerWithCitations]] = []
        self.history_entries: list[QueryLogEntry] = []

    def record_answer(
        self,
        identification: ComponentIdentification | None,
        question: str,
        answer: AnswerWithCitations,
        tenant_id: str | None = None,
    ) -> QueryLogEntry | None:
        del tenant_id
        self.record_calls.append((identification, question, answer))
        if identification is None:
            return None
        entry = QueryLogEntry(
            id="entry-1",
            tenant_id="public_demo",
            component_serial=identification.component_serial or "unknown",
            component_model=identification.model_number or "Unknown",
            question=question,
            answer=answer.answer_text,
            source=QueryLogSource.CACHE,
            confidence=int(round(answer.confidence * 100)),
            timestamp=datetime(2026, 3, 23, 9, 4, tzinfo=timezone.utc),
            doc_source=None,
        )
        self.history_entries = [entry, *self.history_entries]
        return entry

    def get_component_history(
        self,
        component_serial: str,
        tenant_id: str | None = None,
        limit: int = 10,
    ) -> list[QueryLogEntry]:
        del component_serial
        del tenant_id
        return self.history_entries[:limit]


def test_identify_endpoint_returns_typed_identification() -> None:
    app.dependency_overrides[get_vision_service] = lambda: FakeVisionService()
    app.dependency_overrides[get_identify_agent_runner] = lambda: None

    response = client.post(
        "/identify",
        files={"image": ("component.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["manufacturer"] == "Siemens"
    assert body["part_number"] == "6ES7214-1AG40-0XB0"
    assert body["should_attempt_document_lookup"] is True
    assert body["component_serial"] == "SIEMENS_SIMATIC_S7_1200_6ES7214_1AG40_0XB0"


def test_identify_endpoint_primes_cache_in_background(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class PrimingAgent:
        def prime_cache(self, identification: ComponentIdentification) -> bool:
            captured["identification"] = identification
            return True

    def fake_launch_cache_prime(agent, identification):
        agent.prime_cache(identification)

    app.dependency_overrides[get_vision_service] = lambda: FakeVisionService()
    app.dependency_overrides[get_identify_agent_runner] = lambda: PrimingAgent()
    monkeypatch.setattr(
        "backend.api.routes.identify._launch_cache_prime",
        fake_launch_cache_prime,
    )

    response = client.post(
        "/identify",
        files={"image": ("component.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    identification = captured["identification"]
    assert isinstance(identification, ComponentIdentification)
    assert identification.model_number == "SIMATIC S7-1200"


def test_query_endpoint_replaces_uncited_answer_with_safe_message() -> None:
    app.dependency_overrides[get_query_agent_runner] = lambda: RunAgent(
        AnswerWithCitations(
            answer_text="Uncited raw answer",
            citations=[],
            confidence=0.88,
        )
    )

    response = client.post(
        "/query",
        data={"question": "What is the input voltage?"},
        files={"image": ("component.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer_text"] == build_safe_answer().answer_text
    assert body["has_citations"] is False
    assert body["citations"] == []


def test_query_endpoint_returns_safe_message_on_runtime_failure() -> None:
    app.dependency_overrides[get_query_agent_runner] = lambda: RaisingAgent()

    response = client.post(
        "/query",
        data={"question": "What is the battery capacity?"},
        files={"image": ("component.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["answer_text"] == build_safe_answer().answer_text


def test_query_endpoint_accepts_prior_identification_payload() -> None:
    agent = CapturingRunAgent()
    app.dependency_overrides[get_query_agent_runner] = lambda: agent

    response = client.post(
        "/query",
        data={
            "question": "What is the input voltage?",
            "identification": ComponentIdentification(
                manufacturer="Huawei",
                model_number="WS7200",
                component_type="router",
                confidence_score=0.81,
                raw_ocr_text="HUAWEI WS7200",
                should_attempt_document_lookup=True,
            ).model_dump_json(),
        },
        files={"image": ("component.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert agent.question == "What is the input voltage?"
    assert isinstance(agent.identification, ComponentIdentification)
    assert agent.identification.model_number == "WS7200"


def test_query_endpoint_ignores_invalid_identification_payload() -> None:
    agent = CapturingRunAgent()
    app.dependency_overrides[get_query_agent_runner] = lambda: agent

    response = client.post(
        "/query",
        data={
            "question": "What is the input voltage?",
            "identification": '{"manufacturer": "ABB"',
        },
        files={"image": ("component.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert agent.identification is None


def test_query_endpoint_preserves_identification_based_answer() -> None:
    identification = ComponentIdentification(
        manufacturer="ABB",
        model_number="S202",
        part_number="2CDS252001R0404",
        component_type="Miniature circuit breaker",
        confidence_score=0.87,
        raw_ocr_text="ABB S202 2CDS252001R0404",
        should_attempt_document_lookup=False,
    )
    app.dependency_overrides[get_query_agent_runner] = lambda: DetailedAgent(
        answer=AnswerWithCitations(
            answer_text=(
                "From the image identification, the manufacturer appears to be ABB. "
                "Official documentation was not available for a cited answer, so this response is based on the label/image only."
            ),
            citations=[],
            confidence=0.87,
        ),
        identification=identification,
        answer_from_identification=True,
    )

    response = client.post(
        "/query",
        data={"question": "Who is the manufacturer?"},
        files={"image": ("component.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer_text"].startswith("From the image identification")
    assert body["has_citations"] is False


def test_query_endpoint_records_history_after_answer() -> None:
    history = HistoryServiceStub()
    identification = ComponentIdentification(
        manufacturer="ABB",
        model_number="S202",
        part_number="2CDS252001R0404",
        component_type="Miniature circuit breaker",
        confidence_score=0.87,
        raw_ocr_text="ABB S202 2CDS252001R0404",
        should_attempt_document_lookup=True,
    )
    app.dependency_overrides[get_query_agent_runner] = lambda: DetailedAgent(
        answer=AnswerWithCitations(
            answer_text="The rated current is 40 A.",
            citations=[],
            confidence=0.54,
        ),
        identification=identification,
        answer_from_identification=True,
    )
    app.dependency_overrides[get_query_history_service] = lambda: history

    response = client.post(
        "/query",
        data={"question": "What is the model number?"},
        files={"image": ("component.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert len(history.record_calls) == 1
    logged_identification, logged_question, logged_answer = history.record_calls[0]
    assert logged_identification is not None
    assert logged_identification.model_number == "S202"
    assert logged_question == "What is the model number?"
    assert logged_answer.answer_text == "The rated current is 40 A."


def test_history_endpoint_returns_component_history() -> None:
    history = HistoryServiceStub()
    history.history_entries = [
        QueryLogEntry(
            id="entry-1",
            tenant_id="public_demo",
            component_serial="ABB_S202_2CDS252001R0404",
            component_model="ABB S202",
            question="What is the rated current?",
            answer="40 A",
            source=QueryLogSource.CACHE,
            confidence=93,
            timestamp=datetime(2026, 3, 23, 9, 4, tzinfo=timezone.utc),
            doc_source="ABB datasheet",
        )
    ]
    app.dependency_overrides[get_history_route_service] = lambda: history

    response = client.get(
        "/history",
        params={"component_serial": "ABB_S202_2CDS252001R0404"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["question"] == "What is the rated current?"
    assert body[0]["component_serial"] == "ABB_S202_2CDS252001R0404"


def test_health_endpoint_reports_vector_store_status() -> None:
    app.dependency_overrides[get_health_vector_store] = lambda: FakeVectorStore(True)

    root_response = client.get("/")
    response = client.get("/health")

    assert root_response.status_code == 200
    assert root_response.json()["name"] == "Visual Technical Assistant API"
    assert root_response.json()["health_url"] == "/health"
    assert root_response.json()["query_endpoint"] == "/query"
    assert root_response.json()["history_endpoint"] == "/history"
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "vector_store_healthy": True}

