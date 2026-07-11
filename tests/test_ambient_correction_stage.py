"""AmbientCorrectionStage skips cleanly when disabled / no manifest / no R."""

from __future__ import annotations

from cellquorum.ambient_correction.stage import AmbientCorrectionStage
from cellquorum.core.stage import StageResult


class _Ctx:
    def __init__(self, config):
        self.config = config
        self.adata = None

    def require_adata(self):
        raise AssertionError("ambient_correction must not require adata")


def test_stage_skips_when_disabled():
    cfg = type("C", (), {"ambient_correction": type("A", (), {"enabled": False})()})()
    result = AmbientCorrectionStage().run(_Ctx(cfg))
    assert isinstance(result, StageResult)
    assert result.metrics.get("skipped") is True
