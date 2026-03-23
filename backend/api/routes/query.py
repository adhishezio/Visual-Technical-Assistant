from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.agent.graph import VisualTechnicalAssistantAgent
from backend.agent.nodes import build_safe_answer, enforce_cited_answer
from backend.core.models import AnswerWithCitations, ComponentIdentification

router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_agent_runner() -> VisualTechnicalAssistantAgent | None:
    try:
        return VisualTechnicalAssistantAgent()
    except Exception:
        logger.exception("[query_route] failed_to_initialize_agent")
        return None


@router.post("", response_model=AnswerWithCitations)
async def query_component(
    image: UploadFile = File(...),
    question: str = Form(...),
    identification: str | None = Form(default=None),
    agent: VisualTechnicalAssistantAgent | None = Depends(get_agent_runner),
) -> AnswerWithCitations:
    if agent is None:
        return build_safe_answer()

    image_bytes = await image.read()
    parsed_identification: ComponentIdentification | None = None
    if identification:
        try:
            parsed_identification = ComponentIdentification.model_validate_json(
                identification
            )
        except Exception:
            logger.exception("[query_route] invalid_identification_payload")

    try:
        answer = await run_in_threadpool(
            agent.run,
            image_bytes,
            question,
            image.content_type or "image/jpeg",
            parsed_identification,
        )
    except Exception:
        logger.exception("[query_route] agent_run_failed")
        return build_safe_answer()
    return enforce_cited_answer(answer)
