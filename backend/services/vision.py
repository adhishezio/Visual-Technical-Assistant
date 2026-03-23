from __future__ import annotations

import io
from functools import lru_cache
import re
from typing import Protocol

from PIL import Image

from backend.agent.prompts import VISION_EXTRACTION_PROMPT
from backend.core.config import OCRProvider, Settings, VisionProvider, get_settings
from backend.core.models import (
    ComponentIdentification,
    FallbackTier,
    OCRResult,
    VisionExtraction,
)
from backend.services.gemini import GeminiServiceError, generate_structured_content

PART_NUMBER_PATTERNS = (
    re.compile(r"\b[A-Z0-9]{2,}(?:[-/][A-Z0-9]{2,})+\b"),
    re.compile(r"\b[A-Z]{1,5}\d[A-Z0-9-]{4,}\b"),
    re.compile(r"\b\d[A-Z0-9-]{5,}\b"),
)


class VisionServiceError(RuntimeError):
    """Raised when the identification pipeline cannot call an external provider."""


class VisionModelClient(Protocol):
    def identify(self, image_bytes: bytes, mime_type: str) -> VisionExtraction:
        """Return a structured vision extraction."""


class OCRClient(Protocol):
    def extract(self, image_bytes: bytes, mime_type: str) -> OCRResult:
        """Return OCR output with candidate part numbers."""


class NullOCRClient:
    def extract(self, image_bytes: bytes, mime_type: str) -> OCRResult:
        del image_bytes
        del mime_type
        return OCRResult(provider="none")


class GeminiVisionClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def identify(self, image_bytes: bytes, mime_type: str) -> VisionExtraction:
        try:
            return generate_structured_content(
                prompt=VISION_EXTRACTION_PROMPT,
                response_schema=VisionExtraction,
                settings=self.settings,
                image_bytes=image_bytes,
                mime_type=mime_type,
            )
        except GeminiServiceError as exc:
            raise VisionServiceError(str(exc)) from exc


class TrOCRClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        _load_trocr_artifacts(self.settings.trocr_model, self.settings.trocr_device)

    def extract(self, image_bytes: bytes, mime_type: str) -> OCRResult:
        del mime_type

        processor, model, device, torch_module = _load_trocr_artifacts(
            self.settings.trocr_model,
            self.settings.trocr_device,
        )
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch_module.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=self.settings.trocr_max_new_tokens,
            )

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        candidates = extract_part_number_candidates(text, [text])
        average_confidence = estimate_text_confidence(text, candidates)

        return OCRResult(
            raw_text=text,
            average_confidence=average_confidence,
            observations=[],
            detected_part_numbers=candidates,
            provider="trocr",
        )


@lru_cache(maxsize=4)
def _load_trocr_artifacts(model_name: str, requested_device: str | None):
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        raise VisionServiceError(
            "transformers and torch are required for TrOCR extraction."
        ) from exc

    device_name = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(device_name)
    model.eval()
    return processor, model, device_name, torch


class VisionIdentificationService:
    def __init__(
        self,
        settings: Settings | None = None,
        vision_client: VisionModelClient | None = None,
        ocr_client: OCRClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vision_client = vision_client or self._build_vision_client()
        self.ocr_client = ocr_client or self._build_ocr_client()

    def identify_component(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
    ) -> ComponentIdentification:
        vision_extraction, vision_error = self._run_vision(image_bytes, mime_type)
        trocr_required = self._should_run_trocr(vision_extraction)
        ocr_error: str | None = None
        if trocr_required:
            ocr_result, ocr_error = self._run_ocr(image_bytes, mime_type)
        else:
            ocr_result = OCRResult(
                raw_text=vision_extraction.extracted_text,
                average_confidence=vision_extraction.part_number_confidence,
                detected_part_numbers=[vision_extraction.part_number]
                if vision_extraction.part_number
                else [],
                provider="gemini",
            )

        ocr_part_number = pick_best_part_number(ocr_result.detected_part_numbers)
        part_number = vision_extraction.part_number
        if ocr_part_number:
            part_number = ocr_part_number

        fallback_tier = determine_fallback_tier(
            vision_extraction=vision_extraction,
            ocr_result=ocr_result,
            trocr_required=trocr_required,
            resolved_part_number=part_number,
            confidence_threshold=self.settings.part_number_confidence_threshold,
        )
        confidence = calculate_identification_confidence(
            vision_confidence=vision_extraction.confidence_score,
            part_number_confidence=max(
                vision_extraction.part_number_confidence,
                ocr_result.average_confidence,
            ),
            fallback_tier=fallback_tier,
            has_part_number=bool(part_number),
        )
        should_attempt_document_lookup = (
            confidence >= self.settings.identification_confidence_threshold
            and fallback_tier is not FallbackTier.MANUAL_INPUT
            and any(
                [
                    vision_extraction.manufacturer,
                    vision_extraction.model_number,
                    part_number,
                    vision_extraction.component_type,
                ]
            )
        )
        requires_manual_input = fallback_tier is FallbackTier.MANUAL_INPUT or (
            not should_attempt_document_lookup
            and not any([vision_extraction.model_number, part_number])
        )
        error_details = " | ".join(
            error for error in [vision_error, ocr_error] if error
        ) or None

        return ComponentIdentification(
            manufacturer=vision_extraction.manufacturer,
            model_number=vision_extraction.model_number,
            part_number=part_number,
            component_type=vision_extraction.component_type,
            confidence_score=confidence,
            fallback_tier=fallback_tier,
            raw_ocr_text=merge_extracted_texts(
                vision_extraction.extracted_text,
                ocr_result.raw_text if trocr_required else "",
            ),
            visual_description=vision_extraction.visual_description,
            ocr_result=ocr_result,
            should_attempt_document_lookup=should_attempt_document_lookup,
            requires_manual_input=requires_manual_input,
            error_details=error_details,
        )

    def _build_vision_client(self) -> VisionModelClient:
        if self.settings.vision_provider is VisionProvider.NONE:
            return _StaticVisionClient(VisionExtraction())
        return GeminiVisionClient(settings=self.settings)

    def _build_ocr_client(self) -> OCRClient:
        if self.settings.ocr_provider is OCRProvider.NONE:
            return NullOCRClient()
        return TrOCRClient(settings=self.settings)

    def _run_vision(self, image_bytes: bytes, mime_type: str) -> tuple[VisionExtraction, str | None]:
        try:
            return self.vision_client.identify(image_bytes=image_bytes, mime_type=mime_type), None
        except Exception as exc:
            return VisionExtraction(), f"Vision extraction failed: {exc}"

    def _run_ocr(self, image_bytes: bytes, mime_type: str) -> tuple[OCRResult, str | None]:
        try:
            return self.ocr_client.extract(image_bytes=image_bytes, mime_type=mime_type), None
        except Exception as exc:
            return OCRResult(provider="trocr"), f"TrOCR fallback failed: {exc}"

    def _should_run_trocr(self, vision_extraction: VisionExtraction) -> bool:
        return (
            not vision_extraction.part_number
            or vision_extraction.part_number_confidence
            < self.settings.part_number_confidence_threshold
        )


class _StaticVisionClient:
    def __init__(self, extraction: VisionExtraction) -> None:
        self.extraction = extraction

    def identify(self, image_bytes: bytes, mime_type: str) -> VisionExtraction:
        del image_bytes
        del mime_type
        return self.extraction


def extract_part_number_candidates(raw_text: str, observation_texts: list[str]) -> list[str]:
    search_text = " ".join([raw_text, *observation_texts]).upper()
    candidates: set[str] = set()
    for pattern in PART_NUMBER_PATTERNS:
        for match in pattern.findall(search_text):
            normalized = normalize_part_number(match)
            if normalized:
                candidates.add(normalized)
    return sorted(candidates, key=score_part_number, reverse=True)


def normalize_part_number(value: str) -> str:
    normalized = re.sub(r"\s+", "", value.upper())
    normalized = re.sub(r"[^A-Z0-9/-]", "", normalized)
    return normalized.strip("-/")


def score_part_number(candidate: str) -> int:
    score = len(candidate)
    if any(character.isalpha() for character in candidate) and any(
        character.isdigit() for character in candidate
    ):
        score += 10
    if "-" in candidate or "/" in candidate:
        score += 5
    return score


def pick_best_part_number(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    return sorted(candidates, key=score_part_number, reverse=True)[0]


def estimate_text_confidence(raw_text: str, candidates: list[str]) -> float:
    if candidates:
        best_candidate = candidates[0]
        confidence = 0.65 + min(len(best_candidate), 20) * 0.01
        if any(character.isalpha() for character in best_candidate) and any(
            character.isdigit() for character in best_candidate
        ):
            confidence += 0.1
        return min(confidence, 0.95)
    if raw_text.strip():
        return 0.45
    return 0.0


def determine_fallback_tier(
    vision_extraction: VisionExtraction,
    ocr_result: OCRResult,
    trocr_required: bool,
    resolved_part_number: str | None,
    confidence_threshold: float,
) -> FallbackTier:
    if resolved_part_number:
        if trocr_required and ocr_result.detected_part_numbers:
            return FallbackTier.OCR_CONFIRMED
        if (
            vision_extraction.part_number
            and vision_extraction.part_number_confidence >= confidence_threshold
        ):
            return FallbackTier.OCR_CONFIRMED
    if merge_extracted_texts(vision_extraction.extracted_text, ocr_result.raw_text):
        return FallbackTier.OCR_PARTIAL
    if any(
        [
            vision_extraction.manufacturer,
            vision_extraction.model_number,
            vision_extraction.component_type,
            vision_extraction.visual_description,
        ]
    ):
        return FallbackTier.VISION_ONLY
    return FallbackTier.MANUAL_INPUT


def merge_extracted_texts(*texts: str) -> str:
    unique_texts: list[str] = []
    for text in texts:
        cleaned = text.strip()
        if cleaned and cleaned not in unique_texts:
            unique_texts.append(cleaned)
    return "\n".join(unique_texts)


def calculate_identification_confidence(
    vision_confidence: float,
    part_number_confidence: float,
    fallback_tier: FallbackTier,
    has_part_number: bool,
) -> float:
    if fallback_tier is FallbackTier.OCR_CONFIRMED:
        score = (vision_confidence * 0.45) + (part_number_confidence * 0.55)
        if has_part_number:
            score += 0.1
        return min(score, 1.0)
    if fallback_tier is FallbackTier.OCR_PARTIAL:
        return min((vision_confidence * 0.6) + (part_number_confidence * 0.4), 1.0)
    if fallback_tier is FallbackTier.VISION_ONLY:
        return min(vision_confidence * 0.85, 1.0)
    return 0.0
