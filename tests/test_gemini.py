from __future__ import annotations

import json

from backend.core.config import Environment
from backend.core.models import VisionExtraction
from backend.services.gemini import build_gemini_client, generate_structured_content


class _FakeResponse:
    def __init__(self, parsed, text: str = "") -> None:
        self.parsed = parsed
        self.text = text


class _FakeModelsClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def generate_content(self, *args, **kwargs) -> _FakeResponse:
        del args
        del kwargs
        return self._response


class _FakeGeminiClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.models = _FakeModelsClient(response)


class _RetryingModelsClient:
    def __init__(self, response: _FakeResponse, failures_before_success: int) -> None:
        self._response = response
        self._failures_remaining = failures_before_success
        self.calls = 0

    def generate_content(self, *args, **kwargs) -> _FakeResponse:
        del args
        del kwargs
        self.calls += 1
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise RuntimeError("temporary provider failure")
        return self._response


class _RetryingGeminiClient:
    def __init__(self, response: _FakeResponse, failures_before_success: int) -> None:
        self.models = _RetryingModelsClient(response, failures_before_success)


class _RecordingClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs



def test_generate_structured_content_falls_back_to_response_text(monkeypatch, test_settings) -> None:
    payload = {
        "manufacturer": "Siemens",
        "model_number": "SIMATIC S7-1200",
        "part_number": "6ES7214-1AG40-0XB0",
        "component_type": "PLC",
        "visual_description": "Compact PLC",
        "extracted_text": "SIEMENS 6ES7214-1AG40-0XB0",
        "confidence_score": 0.92,
        "part_number_confidence": 0.97,
    }
    monkeypatch.setattr(
        "backend.services.gemini.build_gemini_client",
        lambda settings=None: _FakeGeminiClient(
            _FakeResponse(parsed=None, text=json.dumps(payload))
        ),
    )

    result = generate_structured_content(
        prompt="Identify this component.",
        response_schema=VisionExtraction,
        settings=test_settings,
    )

    assert result.part_number == "6ES7214-1AG40-0XB0"
    assert result.manufacturer == "Siemens"

def test_generate_structured_content_retries_transient_failures(
    monkeypatch,
    test_settings,
) -> None:
    payload = {
        "manufacturer": "ABB",
        "model_number": "M2BAX 160MLA 4",
        "part_number": "3GBA162410-ADF",
        "component_type": "Motor",
        "visual_description": "Motor nameplate",
        "extracted_text": "ABB 3GBA162410-ADF",
        "confidence_score": 0.97,
        "part_number_confidence": 0.99,
    }
    client = _RetryingGeminiClient(
        _FakeResponse(parsed=None, text=json.dumps(payload)),
        failures_before_success=2,
    )
    monkeypatch.setattr(
        "backend.services.gemini.build_gemini_client",
        lambda settings=None: client,
    )
    monkeypatch.setattr("backend.services.gemini.time.sleep", lambda seconds: None)

    result = generate_structured_content(
        prompt="Identify this component.",
        response_schema=VisionExtraction,
        settings=test_settings.model_copy(
            update={
                "gemini_max_retries": 3,
                "gemini_retry_backoff_seconds": 0.01,
            }
        ),
    )

    assert result.part_number == "3GBA162410-ADF"
    assert client.models.calls == 3


def test_build_gemini_client_prefers_api_key_for_local_runs(monkeypatch, test_settings) -> None:
    import google.genai

    recorded: list[dict[str, object]] = []

    def fake_client(**kwargs):
        recorded.append(kwargs)
        return _RecordingClient(**kwargs)

    monkeypatch.setattr(google.genai, "Client", fake_client)

    settings = test_settings.model_copy(
        update={
            "environment": Environment.LOCAL,
            "google_cloud_project": "demo-project",
            "google_cloud_location": "us-central1",
            "google_api_key": "demo-key",
        }
    )

    build_gemini_client(settings)

    assert recorded == [{"api_key": "demo-key"}]



def test_build_gemini_client_prefers_api_key_even_in_production(monkeypatch, test_settings) -> None:
    import google.genai

    recorded: list[dict[str, object]] = []

    def fake_client(**kwargs):
        recorded.append(kwargs)
        return _RecordingClient(**kwargs)

    monkeypatch.setattr(google.genai, "Client", fake_client)

    settings = test_settings.model_copy(
        update={
            "google_cloud_project": "demo-project",
            "google_cloud_location": "us-central1",
            "google_api_key": "demo-key",
        }
    )

    build_gemini_client(settings)

    assert recorded == [{"api_key": "demo-key"}]

