"""Integration-benchmark evaluation stage package."""

from __future__ import annotations

from cellquorum.integration_benchmark.scib_benchmark import ScibBenchmarkMethod
from cellquorum.integration_benchmark.stage import IntegrationBenchmarkStage
from cellquorum.methods.registry import METHOD_REGISTRY

# Register the scib-metrics benchmark method.
if not METHOD_REGISTRY.has("integration_benchmark", "scib_benchmark"):
    METHOD_REGISTRY.register(ScibBenchmarkMethod)

__all__ = ["IntegrationBenchmarkStage", "ScibBenchmarkMethod"]
