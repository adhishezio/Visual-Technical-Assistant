from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, File, UploadFile

from backend.core.models import ComponentIdentification
from backend.services.vision import VisionIdentificationService

router = APIRouter()


@lru_cache(maxsize=1)
def get_vision_service() -> VisionIdentificationService:
    return VisionIdentificationService()


@router.post("", response_model=ComponentIdentification)
async def identify_component(
    image: UploadFile = File(...),
    vision_service: VisionIdentificationService = Depends(get_vision_service),
) -> ComponentIdentification:
    image_bytes = await image.read()
    return vision_service.identify_component(
        image_bytes=image_bytes,
        mime_type=image.content_type or "image/jpeg",
    )
