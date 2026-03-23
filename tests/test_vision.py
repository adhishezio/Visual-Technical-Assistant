from __future__ import annotations

from backend.core.models import FallbackTier, OCRResult, VisionExtraction
from backend.services.vision import VisionIdentificationService


class FakeVisionClient:
    def __init__(self, extraction: VisionExtraction) -> None:
        self.extraction = extraction

    def identify(self, image_bytes: bytes, mime_type: str) -> VisionExtraction:
        del image_bytes
        del mime_type
        return self.extraction


class FakeOCRClient:
    def __init__(self, result: OCRResult) -> None:
        self.result = result
        self.calls = 0

    def extract(self, image_bytes: bytes, mime_type: str) -> OCRResult:
        del image_bytes
        del mime_type
        self.calls += 1
        return self.result


class FailingVisionClient:
    def identify(self, image_bytes: bytes, mime_type: str) -> VisionExtraction:
        del image_bytes
        del mime_type
        raise RuntimeError("gemini parse failure")


def test_high_confidence_gemini_part_number_skips_trocr(test_settings) -> None:
    ocr_client = FakeOCRClient(OCRResult(raw_text="unused", average_confidence=0.0))
    service = VisionIdentificationService(
        settings=test_settings,
        vision_client=FakeVisionClient(
            VisionExtraction(
                manufacturer="Siemens",
                model_number="SIMATIC S7-1200",
                part_number="6ES7214-1AG40-0XB0",
                component_type="PLC",
                visual_description="Compact PLC with screw terminals.",
                extracted_text="SIEMENS 6ES7214-1AG40-0XB0",
                confidence_score=0.8,
                part_number_confidence=0.95,
            )
        ),
        ocr_client=ocr_client,
    )

    result = service.identify_component(image_bytes=b"test-image")

    assert result.part_number == "6ES7214-1AG40-0XB0"
    assert result.fallback_tier is FallbackTier.OCR_CONFIRMED
    assert result.should_attempt_document_lookup is True
    assert ocr_client.calls == 0


def test_trocr_fallback_overrides_low_confidence_gemini_part_number(test_settings) -> None:
    service = VisionIdentificationService(
        settings=test_settings,
        vision_client=FakeVisionClient(
            VisionExtraction(
                manufacturer="Siemens",
                model_number="SIMATIC S7-1200",
                part_number="GUESS-123",
                component_type="PLC",
                visual_description="Compact PLC with screw terminals.",
                extracted_text="SIEMENS GUESS-123",
                confidence_score=0.72,
                part_number_confidence=0.2,
            )
        ),
        ocr_client=FakeOCRClient(
            OCRResult(
                raw_text="SIEMENS 6ES7214-1AG40-0XB0",
                average_confidence=0.91,
                detected_part_numbers=["6ES7214-1AG40-0XB0"],
                provider="trocr",
            )
        ),
    )

    result = service.identify_component(image_bytes=b"test-image")

    assert result.part_number == "6ES7214-1AG40-0XB0"
    assert result.fallback_tier is FallbackTier.OCR_CONFIRMED
    assert result.should_attempt_document_lookup is True


def test_visual_only_result_stays_in_fallback_tier_three(test_settings) -> None:
    service = VisionIdentificationService(
        settings=test_settings,
        vision_client=FakeVisionClient(
            VisionExtraction(
                manufacturer=None,
                model_number=None,
                part_number=None,
                component_type="relay",
                visual_description="Blue relay module with screw terminals.",
                extracted_text="",
                confidence_score=0.7,
                part_number_confidence=0.0,
            )
        ),
        ocr_client=FakeOCRClient(OCRResult()),
    )

    result = service.identify_component(image_bytes=b"test-image")

    assert result.fallback_tier is FallbackTier.VISION_ONLY
    assert result.should_attempt_document_lookup is True


def test_vision_failures_are_exposed_in_error_details(test_settings) -> None:
    service = VisionIdentificationService(
        settings=test_settings,
        vision_client=FailingVisionClient(),
        ocr_client=FakeOCRClient(OCRResult()),
    )

    result = service.identify_component(image_bytes=b"test-image")

    assert result.error_details == "Vision extraction failed: gemini parse failure"
    assert result.requires_manual_input is True
