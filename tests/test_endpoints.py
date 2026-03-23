from __future__ import annotations

from fastapi.testclient import TestClient

from backend.agent.graph import VisualTechnicalAssistantAgent
from backend.agent.nodes import build_safe_answer
from backend.api.routes.identify import get_vision_service
from backend.api.routes.query import get_agent_runner
from backend.core.models import AnswerWithCitations, ComponentIdentification
from backend.main import app, get_health_vector_store
from backend.services.vision import VisionIdentificationService

client = TestClient(app)


class FakeVisionService(VisionIdentificationService):
    def identify_component(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> ComponentIdentification:
        del image_bytes
        del mime_type
        return ComponentIdentification(
            manufacturer="Siemens",
            model_number="SIMATIC S7-1200",
            part_number="6ES7214-1AG40-0XB0",
            component_type="PLC",
            confidence_score=0.92,
            raw_ocr_text="SIEMENS 6ES7214-1AG40-0XB0",
            should_attempt_document_lookup=True,
        )


class FakeAgent(VisualTechnicalAssistantAgent):
    def __init__(self, answer: AnswerWithCitations) -> None:
        self._answer = answer

    def run(self, image_bytes: bytes, question: str, mime_type: str = "image/jpeg") -> AnswerWithCitations:
        del image_bytes
        del question
        del mime_type
        return self._answer


class RaisingAgent:
    def run(self, image_bytes: bytes, question: str, mime_type: str = "image/jpeg") -> AnswerWithCitations:
        del image_bytes
        del question
        del mime_type
        raise RuntimeError("provider failure")


class FakeVectorStore:
    def __init__(self, healthy: bool) -> None:
        self.healthy = healthy

    def health_check(self) -> bool:
        return self.healthy



def test_identify_endpoint_returns_typed_identification() -> None:
    app.dependency_overrides[get_vision_service] = lambda: FakeVisionService()

    response = client.post(
        "/identify",
        files={"image": ("component.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["manufacturer"] == "Siemens"
    assert body["part_number"] == "6ES7214-1AG40-0XB0"
    assert body["should_attempt_document_lookup"] is True



def test_query_endpoint_replaces_uncited_answer_with_safe_message() -> None:
    app.dependency_overrides[get_agent_runner] = lambda: FakeAgent(
        AnswerWithCitations(
            answer_text="Uncited raw answer",
            citations=[],
            confidence=0.88,
        )
    )

    response = client.post(
        "/query",
        data={"question": "What is the input voltage?"},
        files={"image": ("component.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer_text"] == build_safe_answer().answer_text
    assert body["has_citations"] is False
    assert body["citations"] == []



def test_query_endpoint_returns_safe_message_on_runtime_failure() -> None:
    app.dependency_overrides[get_agent_runner] = lambda: RaisingAgent()

    response = client.post(
        "/query",
        data={"question": "What is the battery capacity?"},
        files={"image": ("component.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["answer_text"] == build_safe_answer().answer_text



def test_health_endpoint_reports_vector_store_status() -> None:
    app.dependency_overrides[get_health_vector_store] = lambda: FakeVectorStore(True)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "vector_store_healthy": True}
