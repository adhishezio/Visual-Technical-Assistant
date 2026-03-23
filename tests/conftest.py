from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from backend.core.config import (
    EmbeddingProvider,
    Environment,
    OCRProvider,
    SearchProvider,
    Settings,
    VectorStoreProvider,
    VisionProvider,
)
from backend.main import app


class _BlockedModelsClient:
    def generate_content(self, *args, **kwargs):
        del args
        del kwargs
        raise AssertionError("Tests must not call the real Gemini API.")

    def embed_content(self, *args, **kwargs):
        del args
        del kwargs
        raise AssertionError("Tests must not call the real embedding API.")


class _BlockedGeminiClient:
    models = _BlockedModelsClient()


class _FakeTensor:
    def to(self, device: str) -> "_FakeTensor":
        del device
        return self


class _FakeProcessor:
    def __call__(self, images, return_tensors: str):
        del images
        del return_tensors
        return type("FakeBatch", (), {"pixel_values": _FakeTensor()})()

    def batch_decode(self, generated_ids, skip_special_tokens: bool):
        del generated_ids
        del skip_special_tokens
        return [""]


class _FakeModel:
    def to(self, device: str) -> "_FakeModel":
        del device
        return self

    def eval(self) -> "_FakeModel":
        return self

    def generate(self, pixel_values, max_new_tokens: int):
        del pixel_values
        del max_new_tokens
        return [[0]]


class _FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type
        del exc
        del traceback
        return None


class _FakeTorch:
    @staticmethod
    def no_grad() -> _FakeNoGrad:
        return _FakeNoGrad()


class SettingsForTest(Settings):
    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    app.dependency_overrides = {}
    yield
    app.dependency_overrides = {}


@pytest.fixture(autouse=True)
def block_external_services(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.services.gemini.build_gemini_client",
        lambda settings=None: _BlockedGeminiClient(),
    )
    monkeypatch.setattr(
        "backend.services.vision._load_trocr_artifacts",
        lambda model_name, requested_device: (
            _FakeProcessor(),
            _FakeModel(),
            requested_device or "cpu",
            _FakeTorch(),
        ),
    )


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    return SettingsForTest.model_validate(
        {
            "environment": Environment.TEST,
            "vector_store": VectorStoreProvider.CHROMA,
            "embedding_provider": EmbeddingProvider.HASHING,
            "chroma_persist_directory": tmp_path / "chroma",
            "embedding_dimension": 64,
            "vision_provider": VisionProvider.NONE,
            "ocr_provider": OCRProvider.NONE,
            "search_provider": SearchProvider.NONE,
            "identification_confidence_threshold": 0.5,
            "part_number_confidence_threshold": 0.75,
            "similarity_search_k": 2,
        }
    )
