from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI

from backend.api.routes.identify import router as identify_router
from backend.api.routes.query import router as query_router
from backend.core.config import get_settings
from backend.core.models import HealthStatus, ServiceIndex
from backend.vector_store import get_vector_store
from backend.vector_store.base import VectorStore

backend_logger = logging.getLogger("backend")
if not backend_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    backend_logger.addHandler(handler)
backend_logger.setLevel(logging.INFO)
backend_logger.propagate = False

settings = get_settings()
app = FastAPI(title=settings.app_name, debug=settings.debug)
app.include_router(identify_router, prefix="/identify", tags=["identify"])
app.include_router(query_router, prefix="/query", tags=["query"])


@lru_cache(maxsize=1)
def get_health_vector_store() -> VectorStore | None:
    try:
        return get_vector_store()
    except Exception:
        return None


@app.get("/", response_model=ServiceIndex)
async def service_index() -> ServiceIndex:
    return ServiceIndex(
        name=settings.app_name,
        description=(
            "Production-grade API for component identification and cited technical question answering."
        ),
        docs_url="/docs",
        openapi_url="/openapi.json",
        health_url="/health",
        identify_endpoint="/identify",
        query_endpoint="/query",
    )


@app.get("/health", response_model=HealthStatus)
async def health_check(
    vector_store: VectorStore | None = Depends(get_health_vector_store),
) -> HealthStatus:
    is_healthy = bool(vector_store and vector_store.health_check())
    status = "ok" if is_healthy else "degraded"
    return HealthStatus(status=status, vector_store_healthy=is_healthy)
