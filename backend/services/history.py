from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from backend.core.config import Settings, get_settings
from backend.core.models import (
    AnswerWithCitations,
    ComponentIdentification,
    QueryLogEntry,
    QueryLogSource,
)


class QueryHistoryService:
    def __init__(
        self,
        settings: Settings | None = None,
        firestore_client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.firestore_client = firestore_client
        self.collection = None
        self.order_descending: str = "DESCENDING"

        if firestore_client is not None:
            self.collection = firestore_client.collection(
                self.settings.query_log_collection
            )
            return

        if not self.settings.firestore_project_id:
            return

        try:
            from google.cloud import firestore
        except ImportError:
            return

        self.order_descending = getattr(firestore.Query, "DESCENDING", "DESCENDING")
        self.firestore_client = firestore.Client(
            project=self.settings.firestore_project_id
        )
        self.collection = self.firestore_client.collection(
            self.settings.query_log_collection
        )

    def record_answer(
        self,
        identification: ComponentIdentification | None,
        question: str,
        answer: AnswerWithCitations,
        tenant_id: str | None = None,
    ) -> QueryLogEntry | None:
        if self.collection is None or identification is None:
            return None

        component_serial = (
            identification.component_serial or identification.build_component_serial()
        )
        if not component_serial:
            return None

        resolved_tenant = tenant_id or self.settings.default_tenant_id
        entry = QueryLogEntry(
            tenant_id=resolved_tenant,
            component_serial=component_serial,
            component_model=(
                identification.model_number
                or identification.part_number
                or identification.component_type
                or "Unknown component"
            ),
            question=question.strip(),
            answer=answer.answer_text.strip(),
            source=self._resolve_source(answer),
            confidence=int(round(answer.confidence * 100)),
            timestamp=datetime.now(timezone.utc),
            doc_source=self._resolve_doc_source(answer),
        )
        self.collection.add(entry.model_dump(mode="json", exclude={"id"}))
        return entry

    def get_component_history(
        self,
        component_serial: str,
        tenant_id: str | None = None,
        limit: int | None = None,
    ) -> list[QueryLogEntry]:
        if self.collection is None or not component_serial:
            return []

        resolved_tenant = tenant_id or self.settings.default_tenant_id
        resolved_limit = limit or self.settings.component_history_limit
        query = (
            self.collection.where("tenant_id", "==", resolved_tenant)
            .where("component_serial", "==", component_serial)
            .order_by("timestamp", direction=self.order_descending)
            .limit(resolved_limit)
        )
        snapshots = query.stream() if hasattr(query, "stream") else query.get()
        entries: list[QueryLogEntry] = []
        for snapshot in snapshots:
            payload = snapshot.to_dict() or {}
            payload["id"] = getattr(snapshot, "id", None)
            entries.append(QueryLogEntry.model_validate(payload))
        return entries

    @staticmethod
    def _resolve_source(answer: AnswerWithCitations) -> QueryLogSource:
        if not answer.has_citations:
            return QueryLogSource.CACHE

        urls = [citation.chunk.metadata.source_url for citation in answer.citations]
        if any(urlparse(url).scheme in {"file", "gs"} for url in urls):
            return QueryLogSource.PRIVATE_DOC
        return QueryLogSource.WEB

    @staticmethod
    def _resolve_doc_source(answer: AnswerWithCitations) -> str | None:
        if not answer.citations:
            return None

        metadata = answer.citations[0].chunk.metadata
        if metadata.source_title:
            return metadata.source_title

        parsed = urlparse(metadata.source_url)
        name = PurePosixPath(parsed.path).name
        return name or metadata.source_url
