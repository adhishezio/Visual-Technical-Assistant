from __future__ import annotations

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetryClient(ProductTelemetryClient):
    """Disables Chroma product telemetry for local development, tests, and Docker logs."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        del event
