from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum, IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundingBox(BaseModel):
    left: int
    top: int
    width: int
    height: int


class OCRTextObservation(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox | None = None


class OCRResult(BaseModel):
    raw_text: str = ""
    average_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    observations: list[OCRTextObservation] = Field(default_factory=list)
    detected_part_numbers: list[str] = Field(default_factory=list)
    provider: str | None = None

    @property
    def has_text(self) -> bool:
        return bool(self.raw_text.strip())


class FallbackTier(IntEnum):
    OCR_CONFIRMED = 1
    OCR_PARTIAL = 2
    VISION_ONLY = 3
    MANUAL_INPUT = 4


class CacheKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str

    @property
    def normalized(self) -> str:
        return self.value

    @classmethod
    def from_identification(cls, identification: "ComponentIdentification") -> "CacheKey":
        return cls.from_parts(
            manufacturer=identification.manufacturer,
            model_number=identification.model_number,
            part_number=identification.part_number,
        )

    @classmethod
    def from_parts(
        cls,
        manufacturer: str | None,
        model_number: str | None,
        part_number: str | None,
    ) -> "CacheKey":
        manufacturer_segment = cls._normalize_segment(
            manufacturer,
            fallback="unknown-manufacturer",
        )
        model_segment = cls._normalize_segment(
            model_number or part_number,
            fallback="unknown-model",
        )
        part_segment = cls._normalize_segment(
            part_number or model_number,
            fallback="unknown-part",
        )
        return cls(value=f"{manufacturer_segment}::{model_segment}::{part_segment}")

    @staticmethod
    def _normalize_segment(value: str | None, fallback: str) -> str:
        if not value:
            return fallback
        normalized = re.sub(r"[^a-z0-9-]+", "-", value.strip().lower())
        normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
        return normalized or fallback

    def __str__(self) -> str:
        return self.value


def generate_chunk_id(cache_key: CacheKey, chunk_index: int, content_hash: str) -> str:
    payload = f"{cache_key.normalized}::{chunk_index}::{content_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


class ComponentIdentification(BaseModel):
    manufacturer: str | None = None
    model_number: str | None = None
    part_number: str | None = None
    component_type: str | None = None
    component_serial: str | None = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fallback_tier: FallbackTier = FallbackTier.VISION_ONLY
    raw_ocr_text: str = ""
    visual_description: str | None = None
    ocr_result: OCRResult | None = None
    should_attempt_document_lookup: bool = False
    requires_manual_input: bool = False
    error_details: str | None = None

    @model_validator(mode="after")
    def populate_component_serial(self) -> "ComponentIdentification":
        if self.component_serial:
            return self
        self.component_serial = self.build_component_serial()
        return self

    def to_cache_key(self) -> CacheKey:
        return CacheKey.from_identification(self)

    def build_component_serial(self) -> str | None:
        values = [self.manufacturer, self.model_number, self.part_number]
        if not any(values):
            return None
        parts: list[str] = []
        for value in values:
            if not value:
                continue
            normalized = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
            if normalized and normalized not in parts:
                parts.append(normalized)
        return "_".join(parts) or None


class DocumentType(str, Enum):
    DATASHEET = "datasheet"
    MANUAL = "manual"
    WIRING = "wiring"
    SAFETY = "safety"
    UNKNOWN = "unknown"


class DocumentationCandidate(BaseModel):
    url: str
    title: str | None = None
    document_type: DocumentType = DocumentType.UNKNOWN
    score: float = Field(default=0.0, ge=0.0)


class DocumentMetadata(BaseModel):
    source_url: str
    source_title: str | None = None
    manufacturer: str | None = None
    model_number: str | None = None
    part_number: str | None = None
    document_type: DocumentType
    revision: str | None = None
    page_map: dict[int, int] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str
    cache_key: CacheKey


class DocumentChunk(BaseModel):
    chunk_text: str
    chunk_index: int
    metadata: DocumentMetadata
    page_number: int | None = None
    section_title: str | None = None


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    similarity_score: float = Field(ge=0.0)


class AnswerWithCitations(BaseModel):
    answer_text: str
    citations: list[RetrievedChunk] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    has_citations: bool = False

    @model_validator(mode="after")
    def sync_has_citations(self) -> "AnswerWithCitations":
        self.has_citations = bool(self.citations)
        return self


class QueryLogSource(str, Enum):
    WEB = "web"
    PRIVATE_DOC = "private_doc"
    CACHE = "cache"


class QueryLogEntry(BaseModel):
    id: str | None = None
    tenant_id: str
    component_serial: str
    component_model: str
    question: str
    answer: str
    source: QueryLogSource
    confidence: int = Field(ge=0, le=100)
    timestamp: datetime
    doc_source: str | None = None


class VisionExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manufacturer: str | None = None
    model_number: str | None = None
    part_number: str | None = None
    component_type: str | None = None
    visual_description: str | None = None
    extracted_text: str = ""
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    part_number_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class HealthStatus(BaseModel):
    status: str
    vector_store_healthy: bool


class ServiceIndex(BaseModel):
    name: str
    description: str
    docs_url: str
    openapi_url: str
    health_url: str
    identify_endpoint: str
    query_endpoint: str
    history_endpoint: str



