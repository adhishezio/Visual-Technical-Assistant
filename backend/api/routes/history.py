from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, Query

from backend.core.models import QueryLogEntry
from backend.services.history import QueryHistoryService

router = APIRouter()


@lru_cache(maxsize=1)
def get_history_service() -> QueryHistoryService:
    return QueryHistoryService()


@router.get('', response_model=list[QueryLogEntry])
async def get_component_history(
    component_serial: str = Query(..., min_length=1),
    tenant_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    history_service: QueryHistoryService = Depends(get_history_service),
) -> list[QueryLogEntry]:
    return history_service.get_component_history(
        component_serial=component_serial,
        tenant_id=tenant_id,
        limit=limit,
    )
