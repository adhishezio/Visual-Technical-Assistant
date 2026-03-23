from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    LOCAL = "local"
    TEST = "test"
    PRODUCTION = "production"


class VectorStoreProvider(str, Enum):
    CHROMA = "chroma"
    VERTEX = "vertex"


class EmbeddingProvider(str, Enum):
    VERTEX = "vertex"
    HASHING = "hashing"


class VisionProvider(str, Enum):
    GEMINI = "gemini"
    NONE = "none"


class OCRProvider(str, Enum):
    TROCR = "trocr"
    NONE = "none"


class SearchProvider(str, Enum):
    TAVILY = "tavily"
    NONE = "none"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = Field(default="Visual Technical Assistant API", alias="APP_NAME")
    environment: Environment = Field(default=Environment.LOCAL, alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")

    vector_store: VectorStoreProvider = Field(
        default=VectorStoreProvider.CHROMA,
        alias="VECTOR_STORE",
    )
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.VERTEX,
        alias="EMBEDDING_PROVIDER",
    )
    embedding_model: str = Field(
        default="text-embedding-005",
        alias="EMBEDDING_MODEL",
    )
    chroma_persist_directory: Path = Field(
        default=Path("db"),
        alias="CHROMA_PERSIST_DIRECTORY",
    )
    chroma_collection_name: str = Field(
        default="component_documentation",
        alias="CHROMA_COLLECTION_NAME",
    )
    chroma_distance_metric: str = Field(
        default="cosine",
        alias="CHROMA_DISTANCE_METRIC",
    )
    chroma_anonymized_telemetry: bool = Field(
        default=False,
        alias="CHROMA_ANONYMIZED_TELEMETRY",
    )
    onnxruntime_log_severity_level: int = Field(
        default=4,
        alias="ONNXRUNTIME_LOG_SEVERITY_LEVEL",
    )
    embedding_dimension: int = Field(default=768, alias="EMBEDDING_DIMENSION")

    vision_provider: VisionProvider = Field(
        default=VisionProvider.GEMINI,
        alias="VISION_PROVIDER",
    )
    ocr_provider: OCRProvider = Field(
        default=OCRProvider.TROCR,
        alias="OCR_PROVIDER",
    )
    identification_confidence_threshold: float = Field(
        default=0.6,
        alias="IDENTIFICATION_CONFIDENCE_THRESHOLD",
    )
    part_number_confidence_threshold: float = Field(
        default=0.75,
        alias="PART_NUMBER_CONFIDENCE_THRESHOLD",
    )

    google_cloud_project: str | None = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(
        default="us-central1",
        alias="GOOGLE_CLOUD_LOCATION",
    )
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")

    vertex_project_id: str | None = Field(
        default=None,
        alias="VERTEX_PROJECT_ID",
    )
    vertex_ai_location: str | None = Field(
        default=None,
        alias="VERTEX_AI_LOCATION",
    )
    vertex_index_endpoint_id: str | None = Field(
        default=None,
        alias="VERTEX_INDEX_ENDPOINT_ID",
    )
    vertex_deployed_index_id: str | None = Field(
        default=None,
        alias="VERTEX_DEPLOYED_INDEX_ID",
    )
    firestore_collection: str = Field(
        default="component_chunks",
        alias="FIRESTORE_COLLECTION",
    )
    firestore_project_id: str | None = Field(
        default=None,
        alias="FIRESTORE_PROJECT_ID",
    )

    trocr_model: str = Field(
        default="microsoft/trocr-large-printed",
        alias="TROCR_MODEL",
    )
    trocr_device: str | None = Field(default=None, alias="TROCR_DEVICE")
    trocr_max_new_tokens: int = Field(default=64, alias="TROCR_MAX_NEW_TOKENS")

    search_provider: SearchProvider = Field(
        default=SearchProvider.TAVILY,
        alias="SEARCH_PROVIDER",
    )
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    tavily_max_results: int = Field(default=5, alias="TAVILY_MAX_RESULTS")

    http_timeout_seconds: int = Field(default=20, alias="HTTP_TIMEOUT_SECONDS")
    document_chunk_size: int = Field(default=1200, alias="DOCUMENT_CHUNK_SIZE")
    document_chunk_overlap: int = Field(default=200, alias="DOCUMENT_CHUNK_OVERLAP")
    max_fetch_attempts: int = Field(default=2, alias="MAX_FETCH_ATTEMPTS")
    similarity_search_k: int = Field(default=4, alias="SIMILARITY_SEARCH_K")

    @model_validator(mode="after")
    def validate_vertex_runtime_config(self) -> "Settings":
        requires_vertex_ai = (
            self.embedding_provider is EmbeddingProvider.VERTEX
            or self.vector_store is VectorStoreProvider.VERTEX
        )
        if requires_vertex_ai:
            vertex_ai_missing: list[str] = []
            if not self.vertex_project_id:
                vertex_ai_missing.append("VERTEX_PROJECT_ID")
            if not self.vertex_ai_location:
                vertex_ai_missing.append("VERTEX_AI_LOCATION")
            if vertex_ai_missing:
                missing_values = ", ".join(vertex_ai_missing)
                raise ValueError(
                    "Vertex AI features require the following settings: "
                    f"{missing_values}."
                )

        if self.vector_store is not VectorStoreProvider.VERTEX:
            return self

        vector_store_missing: list[str] = []
        if not self.vertex_index_endpoint_id:
            vector_store_missing.append("VERTEX_INDEX_ENDPOINT_ID")
        if not self.vertex_deployed_index_id:
            vector_store_missing.append("VERTEX_DEPLOYED_INDEX_ID")
        if not self.firestore_collection:
            vector_store_missing.append("FIRESTORE_COLLECTION")
        if not self.firestore_project_id:
            vector_store_missing.append("FIRESTORE_PROJECT_ID")

        if vector_store_missing:
            missing_values = ", ".join(vector_store_missing)
            raise ValueError(
                "VECTOR_STORE=vertex requires the following settings: "
                f"{missing_values}."
            )

        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", self.vertex_deployed_index_id or ""):
            raise ValueError(
                "VERTEX_DEPLOYED_INDEX_ID must start with a letter and contain only letters, numbers, and underscores."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
