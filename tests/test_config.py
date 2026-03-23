from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from backend.core.config import (
    EmbeddingProvider,
    Environment,
    Settings,
    VectorStoreProvider,
)


class SettingsForTest(Settings):
    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding='utf-8',
        extra='ignore',
        populate_by_name=True,
    )


def build_settings(**overrides: Any) -> SettingsForTest:
    payload: dict[str, Any] = {
        'environment': Environment.TEST,
        'vector_store': VectorStoreProvider.CHROMA,
        'embedding_provider': EmbeddingProvider.HASHING,
    }
    payload.update(overrides)
    return SettingsForTest(**payload)


def test_vertex_features_require_dedicated_vertex_project_settings() -> None:
    with pytest.raises(ValidationError, match='VERTEX_PROJECT_ID, VERTEX_AI_LOCATION'):
        build_settings(
            embedding_provider=EmbeddingProvider.VERTEX,
        )


def test_vertex_and_gemini_projects_can_be_configured_independently() -> None:
    settings = build_settings(
        embedding_provider=EmbeddingProvider.VERTEX,
        google_cloud_project='gen-lang-client-0680686270',
        google_cloud_location='us-central1',
        vertex_project_id='jobhunt-490400',
        vertex_ai_location='us-central1',
        firestore_project_id='jobhunt-490400',
    )

    assert settings.google_cloud_project == 'gen-lang-client-0680686270'
    assert settings.vertex_project_id == 'jobhunt-490400'
    assert settings.vertex_ai_location == 'us-central1'
    assert settings.firestore_project_id == 'jobhunt-490400'


def test_vertex_deployed_index_id_rejects_hyphens() -> None:
    with pytest.raises(ValidationError, match='VERTEX_DEPLOYED_INDEX_ID'):
        build_settings(
            vector_store=VectorStoreProvider.VERTEX,
            vertex_project_id='jobhunt-490400',
            vertex_ai_location='us-central1',
            vertex_index_endpoint_id='projects/jobhunt-490400/locations/us-central1/indexEndpoints/123',
            vertex_deployed_index_id='component-docs-v1',
            firestore_collection='component_chunks',
            firestore_project_id='jobhunt-490400',
        )

