from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from backend.core.config import Settings, get_settings
from backend.core.models import (
    CacheKey,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    RetrievedChunk,
    generate_chunk_id,
)
from backend.services.embedder import TextEmbedder, get_embedder
from backend.vector_store.base import VectorStore

logger = logging.getLogger(__name__)
FIRESTORE_BATCH_LIMIT = 500


class VertexVectorStoreError(RuntimeError):
    """Raised when Vertex Vector Search operations fail."""


class VertexVectorStore(VectorStore):
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: TextEmbedder | None = None,
        index_endpoint: Any | None = None,
        index: Any | None = None,
        firestore_client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.deployed_index_id = self.settings.vertex_deployed_index_id or ""
        self.vertex_project_id = self.settings.vertex_project_id or ""
        self.vertex_location = self.settings.vertex_ai_location or ""

        if index_endpoint is None or index is None or firestore_client is None:
            try:
                from google.cloud import aiplatform, firestore
            except ImportError as exc:
                raise VertexVectorStoreError(
                    "google-cloud-aiplatform and google-cloud-firestore are required "
                    "for Vertex Vector Search."
                ) from exc

            aiplatform.init(
                project=self.vertex_project_id,
                location=self.vertex_location,
            )
            self.index_endpoint = index_endpoint or aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=self.settings.vertex_index_endpoint_id or "",
                project=self.vertex_project_id,
                location=self.vertex_location,
            )
            resolved_index_name = self._resolve_index_resource_name(
                self.index_endpoint,
                self.deployed_index_id,
            )
            self.index = index or aiplatform.MatchingEngineIndex(
                index_name=resolved_index_name,
                project=self.vertex_project_id,
                location=self.vertex_location,
            )
            self.firestore_client = firestore_client or firestore.Client(
                project=self.settings.firestore_project_id,
            )
        else:
            self.index_endpoint = index_endpoint
            self.index = index
            self.firestore_client = firestore_client

        self.collection = self.firestore_client.collection(
            self.settings.firestore_collection,
        )

        if not self.health_check():
            raise VertexVectorStoreError(
                "Vertex Vector Search health check failed during initialization."
            )

    def add_documents(
        self,
        chunks: Sequence[DocumentChunk],
        metadata: DocumentMetadata,
    ) -> None:
        if not chunks:
            return

        chunk_ids = [
            generate_chunk_id(metadata.cache_key, chunk.chunk_index, metadata.content_hash)
            for chunk in chunks
        ]
        embeddings = self.embedder.embed_documents([chunk.chunk_text for chunk in chunks])

        try:
            from google.cloud import aiplatform_v1
        except ImportError as exc:
            raise VertexVectorStoreError(
                "google-cloud-aiplatform is required for Vertex Vector Search writes."
            ) from exc

        datapoints = [
            aiplatform_v1.IndexDatapoint(
                datapoint_id=chunk_id,
                feature_vector=embedding,
                restricts=[
                    aiplatform_v1.IndexDatapoint.Restriction(
                        namespace="cache_key",
                        allow_list=[metadata.cache_key.normalized],
                    )
                ],
            )
            for chunk_id, embedding in zip(chunk_ids, embeddings)
        ]
        self.index.upsert_datapoints(datapoints=datapoints)

        chunk_pairs = list(zip(chunks, chunk_ids))
        for batch_start in range(0, len(chunk_pairs), FIRESTORE_BATCH_LIMIT):
            batch = self.firestore_client.batch()
            batch_pairs = chunk_pairs[batch_start : batch_start + FIRESTORE_BATCH_LIMIT]
            for chunk, chunk_id in batch_pairs:
                reference = self.collection.document(chunk_id)
                batch.set(reference, self._build_firestore_doc(chunk, metadata, chunk_id))
            batch.commit()

    def similarity_search(
        self,
        query: str,
        k: int,
        filter_key: CacheKey | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query)
        filters = self._build_filters(filter_key) if filter_key else None
        neighbor_groups = self.index_endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_embedding],
            num_neighbors=k,
            filter=filters,
            return_full_datapoint=False,
        )
        neighbors = list(neighbor_groups[0]) if neighbor_groups else []
        if not neighbors:
            return []

        neighbor_ids = [
            chunk_id
            for chunk_id in (self._extract_neighbor_id(neighbor) for neighbor in neighbors)
            if chunk_id
        ]
        references = [self.collection.document(chunk_id) for chunk_id in neighbor_ids]
        snapshots = {
            snapshot.id: snapshot
            for snapshot in self.firestore_client.get_all(references)
            if getattr(snapshot, "exists", True)
        }

        retrieved: list[tuple[float, RetrievedChunk]] = []
        for neighbor in neighbors:
            chunk_id = self._extract_neighbor_id(neighbor)
            if not chunk_id:
                continue
            snapshot = snapshots.get(chunk_id)
            if snapshot is None:
                logger.warning(
                    "Skipping Vertex result with missing Firestore metadata: %s",
                    chunk_id,
                )
                continue
            distance = self._extract_neighbor_distance(neighbor)
            chunk = self._build_chunk_from_firestore_doc(snapshot.to_dict() or {})
            retrieved.append(
                (
                    distance,
                    RetrievedChunk(
                        chunk=chunk,
                        similarity_score=distance,
                    ),
                )
            )

        retrieved.sort(key=lambda item: item[0])
        return [chunk for _, chunk in retrieved]

    def key_exists(self, cache_key: CacheKey) -> bool:
        documents = (
            self.collection.where("cache_key", "==", cache_key.normalized)
            .limit(1)
            .get()
        )
        return len(list(documents)) > 0

    def delete_by_key(self, cache_key: CacheKey) -> None:
        documents = list(
            self.collection.where("cache_key", "==", cache_key.normalized).get()
        )
        datapoint_ids = [
            document.id for document in documents if getattr(document, "exists", True)
        ]
        if datapoint_ids:
            self.index.remove_datapoints(datapoint_ids=datapoint_ids)

        for batch_start in range(0, len(documents), FIRESTORE_BATCH_LIMIT):
            batch = self.firestore_client.batch()
            for document in documents[batch_start : batch_start + FIRESTORE_BATCH_LIMIT]:
                batch.delete(document.reference)
            batch.commit()

    def health_check(self) -> bool:
        try:
            self.collection.limit(1).get()
            self._resolve_index_resource_name(self.index_endpoint, self.deployed_index_id)
        except Exception:
            return False
        return True

    @staticmethod
    def _resolve_index_resource_name(index_endpoint: Any, deployed_index_id: str) -> str:
        deployed_indexes = getattr(index_endpoint, "deployed_indexes", []) or []
        for deployed_index in deployed_indexes:
            candidate_id = VertexVectorStore._read_value(
                deployed_index,
                "id",
                "deployed_index_id",
            )
            if candidate_id != deployed_index_id:
                continue
            index_name = VertexVectorStore._read_value(
                deployed_index,
                "index",
                "index_name",
            )
            if index_name:
                return str(index_name)
        raise VertexVectorStoreError(
            f"Could not find deployed index '{deployed_index_id}' on the configured endpoint."
        )

    @staticmethod
    def _read_value(candidate: Any, *names: str) -> Any:
        if isinstance(candidate, dict):
            for name in names:
                if name in candidate:
                    return candidate[name]
            return None
        for name in names:
            value = getattr(candidate, name, None)
            if value is not None:
                return value
        return None

    @staticmethod
    def _build_filters(cache_key: CacheKey) -> list[Any]:
        try:
            from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
        except ImportError as exc:
            raise VertexVectorStoreError(
                "google-cloud-aiplatform is required for Vertex Vector Search queries."
            ) from exc
        return [Namespace(name="cache_key", allow_tokens=[cache_key.normalized])]

    @staticmethod
    def _extract_neighbor_id(neighbor: Any) -> str | None:
        if isinstance(neighbor, dict):
            direct_id = neighbor.get("id") or neighbor.get("datapoint_id")
            if isinstance(direct_id, str) and direct_id:
                return direct_id
            datapoint = neighbor.get("datapoint")
            if isinstance(datapoint, dict):
                nested_id = datapoint.get("datapoint_id") or datapoint.get("id")
                if isinstance(nested_id, str) and nested_id:
                    return nested_id
            return None

        direct_id = getattr(neighbor, "id", None) or getattr(neighbor, "datapoint_id", None)
        if isinstance(direct_id, str) and direct_id:
            return direct_id

        datapoint = getattr(neighbor, "datapoint", None)
        nested_id = getattr(datapoint, "datapoint_id", None) or getattr(datapoint, "id", None)
        if isinstance(nested_id, str) and nested_id:
            return nested_id
        return None

    @staticmethod
    def _extract_neighbor_distance(neighbor: Any) -> float:
        raw_distance = (
            neighbor.get("distance")
            if isinstance(neighbor, dict)
            else getattr(neighbor, "distance", 0.0)
        )
        if raw_distance is None:
            return 0.0
        try:
            return float(raw_distance)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _build_firestore_doc(
        chunk: DocumentChunk,
        metadata: DocumentMetadata,
        chunk_id: str,
    ) -> dict[str, Any]:
        retrieved_at = metadata.retrieved_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "chunk_id": chunk_id,
            "chunk_text": chunk.chunk_text,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "cache_key": metadata.cache_key.normalized,
            "source_url": metadata.source_url,
            "manufacturer": metadata.manufacturer,
            "model_number": metadata.model_number,
            "document_type": metadata.document_type.value,
            "revision": metadata.revision,
            "retrieved_at": retrieved_at,
            "content_hash": metadata.content_hash,
        }

    @staticmethod
    def _build_chunk_from_firestore_doc(document: dict[str, Any]) -> DocumentChunk:
        retrieved_at = VertexVectorStore._parse_datetime(str(document["retrieved_at"]))
        metadata = DocumentMetadata(
            source_url=str(document["source_url"]),
            source_title=None,
            manufacturer=str(document.get("manufacturer") or "") or None,
            model_number=str(document.get("model_number") or "") or None,
            part_number=None,
            document_type=DocumentType(str(document["document_type"])),
            revision=str(document.get("revision") or "") or None,
            page_map={},
            retrieved_at=retrieved_at,
            content_hash=str(document["content_hash"]),
            cache_key=CacheKey(value=str(document["cache_key"])),
        )
        page_number = document.get("page_number")
        return DocumentChunk(
            chunk_text=str(document["chunk_text"]),
            chunk_index=int(document["chunk_index"]),
            metadata=metadata,
            page_number=None if page_number is None else int(page_number),
            section_title=None,
        )

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        return datetime.fromisoformat(normalized)
