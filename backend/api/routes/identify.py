from __future__ import annotations

import logging
from threading import Thread
from functools import lru_cache

from fastapi import APIRouter, Depends, File, UploadFile

from backend.agent.graph import VisualTechnicalAssistantAgent
from backend.core.models import ComponentIdentification
from backend.services.vision import VisionIdentificationService

router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vision_service() -> VisionIdentificationService:
    return VisionIdentificationService()


@lru_cache(maxsize=1)
def get_agent_runner() -> VisualTechnicalAssistantAgent | None:
    try:
        return VisualTechnicalAssistantAgent()
    except Exception:
        logger.exception("[identify_route] failed_to_initialize_agent")
        return None


@router.post("", response_model=ComponentIdentification)
async def identify_component(
    image: UploadFile = File(...),
    vision_service: VisionIdentificationService = Depends(get_vision_service),
    agent: VisualTechnicalAssistantAgent | None = Depends(get_agent_runner),
) -> ComponentIdentification:
    image_bytes = await image.read()
    identification = vision_service.identify_component(
        image_bytes=image_bytes,
        mime_type=image.content_type or "image/jpeg",
    )
    if identification.should_attempt_document_lookup and agent is not None:
        _launch_cache_prime(agent, identification)
    return identification


def _launch_cache_prime(
    agent: VisualTechnicalAssistantAgent,
    identification: ComponentIdentification,
) -> None:
    Thread(
        target=_prime_cache_safely,
        args=(agent, identification),
        daemon=True,
        name="prime-cache",
    ).start()


def _prime_cache_safely(
    agent: VisualTechnicalAssistantAgent,
    identification: ComponentIdentification,
) -> None:
    try:
        primed = agent.prime_cache(identification)
        logger.info(
            "[identify_route] prime_cache manufacturer=%r model=%r primed=%s",
            identification.manufacturer,
            identification.model_number,
            primed,
        )
    except Exception:
        logger.exception(
            "[identify_route] prime_cache_failed manufacturer=%r model=%r",
            identification.manufacturer,
            identification.model_number,
        )
