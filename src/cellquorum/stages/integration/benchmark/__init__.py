"""Integration-benchmark evaluation stage package."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.integration.benchmark.scib_benchmark import ScibBenchmarkMethod
from cellquorum.stages.integration.benchmark.stage import IntegrationBenchmarkStage

# Register the scib-metrics benchmark method.
if not METHOD_REGISTRY.has("integration_benchmark", "scib_benchmark"):
    METHOD_REGISTRY.register(ScibBenchmarkMethod)

__all__ = ["IntegrationBenchmarkStage", "ScibBenchmarkMethod"]
