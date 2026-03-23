from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
import importlib
import json
import os
import sys

from backend.core.config import Settings, get_settings
from backend.core.models import (
    CacheKey,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    RetrievedChunk,
)
from backend.services.embedder import TextEmbedder, get_embedder
from backend.vector_store.base import VectorStore

NO_OP_TELEMETRY_IMPL = "backend.vector_store.chroma_telemetry.NoOpProductTelemetryClient"


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: TextEmbedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self._configure_runtime_environment()

        chromadb, chroma_settings_type = _import_chromadb_with_suppressed_stderr()
        client_settings = chroma_settings_type(
            anonymized_telemetry=self.settings.chroma_anonymized_telemetry,
            chroma_product_telemetry_impl=NO_OP_TELEMETRY_IMPL,
            chroma_telemetry_impl=NO_OP_TELEMETRY_IMPL,
            is_persistent=True,
            persist_directory=str(self.settings.chroma_persist_directory),
        )
        self.client = chromadb.PersistentClient(
            path=str(self.settings.chroma_persist_directory),
            settings=client_settings,
        )
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"hnsw:space": self.settings.chroma_distance_metric},
        )

    def add_documents(
        self,
        chunks: Sequence[DocumentChunk],
        metadata: DocumentMetadata,
    ) -> None:
        if not chunks:
            return

        texts = [chunk.chunk_text for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        metadatas = [self._serialize_metadata(chunk, metadata) for chunk in chunks]
        ids = [
            self._chunk_id(metadata.cache_key, metadata.content_hash, chunk.chunk_index)
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query: str,
        k: int,
        filter_key: CacheKey | None = None,
    ) -> list[RetrievedChunk]:
        results = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=k,
            where={"cache_key": filter_key.value} if filter_key else None,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata_model = self._deserialize_metadata(metadata)
            page_number = int(metadata.get("page_number", -1))
            section_title = str(metadata.get("section_title") or "") or None
            chunk = DocumentChunk(
                chunk_text=document,
                chunk_index=int(metadata["chunk_index"]),
                page_number=None if page_number < 0 else page_number,
                section_title=section_title,
                metadata=metadata_model,
            )
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity_score=max(0.0, 1.0 - float(distance)),
                )
            )

        return retrieved

    def key_exists(self, cache_key: CacheKey) -> bool:
        results = self.collection.get(where={"cache_key": cache_key.value}, limit=1)
        return bool(results.get("ids"))

    def delete_by_key(self, cache_key: CacheKey) -> None:
        results = self.collection.get(where={"cache_key": cache_key.value})
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def health_check(self) -> bool:
        try:
            self.collection.count()
        except Exception:
            return False
        return True

    def _configure_runtime_environment(self) -> None:
        os.environ["ANONYMIZED_TELEMETRY"] = (
            "TRUE" if self.settings.chroma_anonymized_telemetry else "FALSE"
        )
        os.environ["ORT_LOG_SEVERITY_LEVEL"] = str(
            self.settings.onnxruntime_log_severity_level
        )
        os.environ.setdefault("ORT_LOG_VERBOSITY_LEVEL", "0")

    @staticmethod
    def _chunk_id(cache_key: CacheKey, content_hash: str, chunk_index: int) -> str:
        return f"{cache_key.value}:{content_hash}:{chunk_index}"

    @staticmethod
    def _serialize_metadata(
        chunk: DocumentChunk,
        metadata: DocumentMetadata,
    ) -> dict[str, str | int | float | bool]:
        return {
            "cache_key": metadata.cache_key.value,
            "source_url": metadata.source_url,
            "source_title": metadata.source_title or "",
            "manufacturer": metadata.manufacturer or "",
            "model_number": metadata.model_number or "",
            "part_number": metadata.part_number or "",
            "document_type": metadata.document_type.value,
            "revision": metadata.revision or "",
            "page_map": json.dumps(metadata.page_map, sort_keys=True),
            "retrieved_at": metadata.retrieved_at.isoformat(),
            "content_hash": metadata.content_hash,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number if chunk.page_number is not None else -1,
            "section_title": chunk.section_title or "",
        }

    @staticmethod
    def _deserialize_metadata(raw_metadata: dict[str, object]) -> DocumentMetadata:
        return DocumentMetadata(
            source_url=str(raw_metadata["source_url"]),
            source_title=str(raw_metadata.get("source_title") or "") or None,
            manufacturer=str(raw_metadata["manufacturer"]) or None,
            model_number=str(raw_metadata["model_number"]) or None,
            part_number=str(raw_metadata["part_number"]) or None,
            document_type=DocumentType(str(raw_metadata["document_type"])),
            revision=str(raw_metadata["revision"]) or None,
            page_map={
                int(chunk_index): int(page_number)
                for chunk_index, page_number in json.loads(
                    str(raw_metadata["page_map"])
                ).items()
            },
            retrieved_at=datetime.fromisoformat(str(raw_metadata["retrieved_at"])),
            content_hash=str(raw_metadata["content_hash"]),
            cache_key=CacheKey(value=str(raw_metadata["cache_key"])),
        )


def _import_chromadb_with_suppressed_stderr():
    with _suppress_native_stderr():
        chromadb = importlib.import_module("chromadb")
        chroma_config = importlib.import_module("chromadb.config")
    return chromadb, chroma_config.Settings


@contextmanager
def _suppress_native_stderr() -> Iterator[None]:
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return

    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stderr.flush()
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
