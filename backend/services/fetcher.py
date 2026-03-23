from __future__ import annotations

from hashlib import sha256
import io

import requests
from pypdf import PdfReader
import trafilatura

from backend.core.config import Settings, get_settings
from backend.core.models import (
    CacheKey,
    ComponentIdentification,
    DocumentationCandidate,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
)


class FetcherError(RuntimeError):
    """Raised when a documentation source cannot be fetched or parsed."""


class DocumentFetcher:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def fetch(
        self,
        candidate: DocumentationCandidate,
        identification: ComponentIdentification,
        cache_key: CacheKey,
    ) -> list[DocumentChunk]:
        response = requests.get(
            candidate.url,
            headers={"User-Agent": "VisualTechnicalAssistant/1.0"},
            timeout=self.settings.http_timeout_seconds,
        )
        response.raise_for_status()
        raw_content = response.content
        content_hash = sha256(raw_content).hexdigest()
        content_type = response.headers.get("content-type", "").lower()
        resolved_source_url = str(response.url)
        is_pdf = "pdf" in content_type or candidate.url.lower().endswith(".pdf")

        if is_pdf:
            return self._parse_pdf(
                raw_content=raw_content,
                candidate=candidate,
                identification=identification,
                cache_key=cache_key,
                content_hash=content_hash,
                source_url=resolved_source_url,
            )
        return self._parse_html(
            raw_content=raw_content,
            candidate=candidate,
            identification=identification,
            cache_key=cache_key,
            content_hash=content_hash,
            source_url=resolved_source_url,
        )

    def _parse_pdf(
        self,
        raw_content: bytes,
        candidate: DocumentationCandidate,
        identification: ComponentIdentification,
        cache_key: CacheKey,
        content_hash: str,
        source_url: str,
    ) -> list[DocumentChunk]:
        reader = PdfReader(io.BytesIO(raw_content))
        metadata = DocumentMetadata(
            source_url=source_url,
            source_title=candidate.title,
            manufacturer=identification.manufacturer,
            model_number=identification.model_number,
            part_number=identification.part_number,
            document_type=candidate.document_type,
            content_hash=content_hash,
            cache_key=cache_key,
        )
        chunks: list[DocumentChunk] = []
        page_map: dict[int, int] = {}

        chunk_index = 0
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            for chunk_text in split_text(
                page_text,
                self.settings.document_chunk_size,
                self.settings.document_chunk_overlap,
            ):
                page_map[chunk_index] = page_number
                chunks.append(
                    DocumentChunk(
                        chunk_text=chunk_text,
                        chunk_index=chunk_index,
                        metadata=metadata,
                        page_number=page_number,
                        section_title=None,
                    )
                )
                chunk_index += 1

        if not chunks:
            raise FetcherError(f"No extractable PDF text found for {candidate.url}")

        metadata.page_map = page_map
        return chunks

    def _parse_html(
        self,
        raw_content: bytes,
        candidate: DocumentationCandidate,
        identification: ComponentIdentification,
        cache_key: CacheKey,
        content_hash: str,
        source_url: str,
    ) -> list[DocumentChunk]:
        html = raw_content.decode("utf-8", errors="ignore")
        extracted_text = trafilatura.extract(
            html,
            url=source_url,
            include_comments=False,
            include_tables=True,
        )
        if not extracted_text:
            raise FetcherError(f"No extractable HTML text found for {candidate.url}")

        metadata = DocumentMetadata(
            source_url=source_url,
            source_title=candidate.title,
            manufacturer=identification.manufacturer,
            model_number=identification.model_number,
            part_number=identification.part_number,
            document_type=(
                candidate.document_type
                if candidate.document_type is not DocumentType.UNKNOWN
                else DocumentType.MANUAL
            ),
            content_hash=content_hash,
            cache_key=cache_key,
        )
        chunks: list[DocumentChunk] = []
        for chunk_index, chunk_text in enumerate(
            split_text(
                extracted_text,
                self.settings.document_chunk_size,
                self.settings.document_chunk_overlap,
            )
        ):
            chunks.append(
                DocumentChunk(
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                    metadata=metadata,
                    page_number=None,
                    section_title=candidate.title or f"HTML section {chunk_index + 1}",
                )
            )

        if not chunks:
            raise FetcherError(f"No HTML chunks could be built for {candidate.url}")

        metadata.page_map = {}
        return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks
