from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.core.config import VectorStoreProvider
from backend.core.models import (
    CacheKey,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    generate_chunk_id,
)
from backend.vector_store.chroma import ChromaVectorStore
from backend.vector_store.vertex import VertexVectorStore


class StubEmbedder:
    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        anchor_index = 0 if "voltage" in text.lower() else 1
        vector[anchor_index] = 1.0
        return vector


class _FakeDocumentReference:
    def __init__(self, store: dict[str, dict], document_id: str) -> None:
        self._store = store
        self.id = document_id


class _FakeDocumentSnapshot:
    def __init__(self, reference: _FakeDocumentReference, data: dict | None) -> None:
        self.reference = reference
        self.id = reference.id
        self._data = data
        self.exists = data is not None

    def to_dict(self) -> dict | None:
        if self._data is None:
            return None
        return dict(self._data)


class _FakeBatch:
    def __init__(self) -> None:
        self._operations: list[tuple[str, _FakeDocumentReference, dict | None]] = []

    def set(self, reference: _FakeDocumentReference, data: dict) -> None:
        self._operations.append(("set", reference, dict(data)))

    def delete(self, reference: _FakeDocumentReference) -> None:
        self._operations.append(("delete", reference, None))

    def commit(self) -> None:
        for operation, reference, data in self._operations:
            if operation == "set" and data is not None:
                reference._store[reference.id] = data
            elif operation == "delete":
                reference._store.pop(reference.id, None)
        self._operations.clear()


class _FakeQuery:
    def __init__(
        self,
        store: dict[str, dict],
        field_name: str | None = None,
        field_value: str | None = None,
        limit_count: int | None = None,
    ) -> None:
        self._store = store
        self._field_name = field_name
        self._field_value = field_value
        self._limit_count = limit_count

    def limit(self, count: int) -> "_FakeQuery":
        return _FakeQuery(
            self._store,
            field_name=self._field_name,
            field_value=self._field_value,
            limit_count=count,
        )

    def get(self) -> list[_FakeDocumentSnapshot]:
        snapshots: list[_FakeDocumentSnapshot] = []
        for document_id, data in sorted(self._store.items()):
            if self._field_name is not None and data.get(self._field_name) != self._field_value:
                continue
            reference = _FakeDocumentReference(self._store, document_id)
            snapshots.append(_FakeDocumentSnapshot(reference, data))
        if self._limit_count is not None:
            return snapshots[: self._limit_count]
        return snapshots


class _FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.store: dict[str, dict] = {}

    def document(self, document_id: str) -> _FakeDocumentReference:
        return _FakeDocumentReference(self.store, document_id)

    def where(self, field_name: str, operator: str, value: str) -> _FakeQuery:
        assert operator == "=="
        return _FakeQuery(self.store, field_name=field_name, field_value=value)

    def limit(self, count: int) -> _FakeQuery:
        return _FakeQuery(self.store, limit_count=count)


class _FakeFirestoreClient:
    def __init__(self, *args, **kwargs) -> None:
        del args
        del kwargs
        self.collections: dict[str, _FakeCollection] = {}
        self.requested_collection: str | None = None

    def collection(self, name: str) -> _FakeCollection:
        self.requested_collection = name
        if name not in self.collections:
            self.collections[name] = _FakeCollection(name)
        return self.collections[name]

    def batch(self) -> _FakeBatch:
        return _FakeBatch()

    def get_all(self, references: list[_FakeDocumentReference]) -> list[_FakeDocumentSnapshot]:
        return [
            _FakeDocumentSnapshot(reference, reference._store.get(reference.id))
            for reference in references
        ]


def test_chroma_vector_store_round_trip(test_settings) -> None:
    cache_key = CacheKey.from_parts(
        manufacturer="Siemens",
        model_number="SIMATIC S7-1200",
        part_number="6ES7214-1AG40-0XB0",
    )
    metadata = DocumentMetadata(
        source_url="https://example.com/siemens-s7.pdf",
        source_title="SIMATIC S7 Datasheet",
        manufacturer="Siemens",
        model_number="SIMATIC S7-1200",
        part_number="6ES7214-1AG40-0XB0",
        document_type=DocumentType.DATASHEET,
        revision="A",
        page_map={0: 1, 1: 2},
        content_hash="hash-1",
        cache_key=cache_key,
    )
    chunks = [
        DocumentChunk(
            chunk_text="Input voltage is 24 V DC and current draw is 0.5 A.",
            chunk_index=0,
            metadata=metadata,
            page_number=1,
            section_title=None,
        ),
        DocumentChunk(
            chunk_text="Mount on DIN rail and use shielded wiring in noisy environments.",
            chunk_index=1,
            metadata=metadata,
            page_number=2,
            section_title=None,
        ),
    ]

    store = ChromaVectorStore(settings=test_settings, embedder=StubEmbedder(dimension=64))
    store.add_documents(chunks=chunks, metadata=metadata)

    assert store.health_check() is True
    assert store.key_exists(cache_key) is True

    results = store.similarity_search(
        query="What is the input voltage?",
        k=1,
        filter_key=cache_key,
    )

    assert len(results) == 1
    assert "voltage" in results[0].chunk.chunk_text.lower()
    assert results[0].chunk.metadata.source_title == "SIMATIC S7 Datasheet"

    store.delete_by_key(cache_key)

    assert store.key_exists(cache_key) is False


class TestVertexVectorStore:
    def test_chunk_id_is_deterministic(self) -> None:
        cache_key = CacheKey.from_parts("HUAWEI", "WS7200", None)

        chunk_id_1 = generate_chunk_id(cache_key, chunk_index=0, content_hash="abc123")
        chunk_id_2 = generate_chunk_id(cache_key, chunk_index=0, content_hash="abc123")

        assert chunk_id_1 == chunk_id_2

    def test_chunk_id_differs_by_index(self) -> None:
        cache_key = CacheKey.from_parts("HUAWEI", "WS7200", None)

        chunk_id_1 = generate_chunk_id(cache_key, chunk_index=0, content_hash="abc123")
        chunk_id_2 = generate_chunk_id(cache_key, chunk_index=1, content_hash="abc123")

        assert chunk_id_1 != chunk_id_2

    @patch("google.cloud.firestore.Client")
    @patch("google.cloud.aiplatform.MatchingEngineIndex")
    @patch("google.cloud.aiplatform.MatchingEngineIndexEndpoint")
    @patch("google.cloud.aiplatform.init")
    def test_round_trip(
        self,
        mock_aiplatform_init,
        mock_index_endpoint_cls,
        mock_index_cls,
        mock_firestore_client_cls,
        test_settings,
    ) -> None:
        del mock_aiplatform_init
        settings = test_settings.model_copy(
            update={
                "vector_store": VectorStoreProvider.VERTEX,
                "embedding_provider": "hashing",
                "google_cloud_project": "gen-lang-client-0680686270",
                "google_cloud_location": "us-central1",
                "vertex_project_id": "jobhunt-490400",
                "vertex_ai_location": "us-central1",
                "vertex_index_endpoint_id": "projects/jobhunt-490400/locations/us-central1/indexEndpoints/endpoint-456",
                "vertex_deployed_index_id": "component_docs_v1",
                "firestore_collection": "component_chunks",
                "firestore_project_id": "jobhunt-490400",
                "embedding_dimension": 64,
            }
        )
        cache_key = CacheKey.from_parts(
            manufacturer="HUAWEI",
            model_number="WS7200",
            part_number=None,
        )
        metadata = DocumentMetadata(
            source_url="https://consumer.huawei.com/en/routers/ax3/specs/",
            source_title="HUAWEI WiFi AX3 Specifications",
            manufacturer="HUAWEI",
            model_number="WS7200",
            part_number=None,
            document_type=DocumentType.MANUAL,
            revision=None,
            page_map={3: 12},
            retrieved_at=datetime(2026, 3, 22, 21, 0, 0, tzinfo=timezone.utc),
            content_hash="b7e2",
            cache_key=cache_key,
        )
        chunk = DocumentChunk(
            chunk_text="The input voltage range is 100-240V AC.",
            chunk_index=3,
            metadata=metadata,
            page_number=12,
            section_title="Electrical",
        )
        chunk_id = generate_chunk_id(cache_key, chunk.chunk_index, metadata.content_hash)

        endpoint_instance = MagicMock()
        endpoint_instance.deployed_indexes = [
            SimpleNamespace(
                id="component_docs_v1",
                index="projects/jobhunt-490400/locations/us-central1/indexes/index-123",
            )
        ]
        endpoint_instance.find_neighbors.return_value = [
            [SimpleNamespace(id=chunk_id, distance=0.2)]
        ]
        mock_index_endpoint_cls.return_value = endpoint_instance

        index_instance = MagicMock()
        mock_index_cls.return_value = index_instance

        firestore_client = _FakeFirestoreClient()
        mock_firestore_client_cls.return_value = firestore_client

        store = VertexVectorStore(
            settings=settings,
            embedder=StubEmbedder(dimension=64),
        )

        store.add_documents(chunks=[chunk], metadata=metadata)

        assert store.key_exists(cache_key) is True
        assert store.vertex_project_id == "jobhunt-490400"
        assert store.vertex_location == "us-central1"
        assert firestore_client.requested_collection == "component_chunks"
        assert index_instance.upsert_datapoints.call_count == 1
        upsert_datapoints = index_instance.upsert_datapoints.call_args.kwargs["datapoints"]
        assert len(upsert_datapoints) == 1
        assert upsert_datapoints[0].datapoint_id == chunk_id
        assert len(upsert_datapoints[0].restricts) == 1
        assert upsert_datapoints[0].restricts[0].namespace == "cache_key"
        assert list(upsert_datapoints[0].restricts[0].allow_list) == [cache_key.normalized]

        expected_document = {
            "chunk_id": chunk_id,
            "chunk_text": "The input voltage range is 100-240V AC.",
            "chunk_index": 3,
            "page_number": 12,
            "cache_key": cache_key.normalized,
            "source_url": "https://consumer.huawei.com/en/routers/ax3/specs/",
            "manufacturer": "HUAWEI",
            "model_number": "WS7200",
            "document_type": "manual",
            "revision": None,
            "retrieved_at": "2026-03-22T21:00:00Z",
            "content_hash": "b7e2",
        }
        assert firestore_client.collection("component_chunks").store[chunk_id] == expected_document

        results = store.similarity_search(
            query="What is the input voltage?",
            k=1,
            filter_key=cache_key,
        )

        assert len(results) == 1
        assert results[0].chunk.chunk_text == chunk.chunk_text
        assert results[0].chunk.metadata.source_url == metadata.source_url
        assert results[0].similarity_score == 0.2

        find_neighbors_kwargs = endpoint_instance.find_neighbors.call_args.kwargs
        assert find_neighbors_kwargs["deployed_index_id"] == "component_docs_v1"
        assert find_neighbors_kwargs["num_neighbors"] == 1
        assert find_neighbors_kwargs["queries"] == [[1.0] + [0.0] * 63]
        assert find_neighbors_kwargs["filter"][0].name == "cache_key"
        assert find_neighbors_kwargs["filter"][0].allow_tokens == [cache_key.normalized]

        assert store.health_check() is True

        store.delete_by_key(cache_key)

        index_instance.remove_datapoints.assert_called_once_with(datapoint_ids=[chunk_id])
        assert store.key_exists(cache_key) is False
        assert firestore_client.collection("component_chunks").store == {}

