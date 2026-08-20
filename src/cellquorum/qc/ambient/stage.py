"""Backward-compatible re-export shim — see ``cellquorum.ambient_correction.stage``.

Importing this module triggers import of the moved stage module, which is where
the ``@register_stage`` decorator fires. Re-importing here does not re-register:
Python caches the already-imported module.
"""

from __future__ import annotations

from cellquorum.ambient_correction.stage import AmbientCorrectionStage

__all__ = ["AmbientCorrectionStage"]
