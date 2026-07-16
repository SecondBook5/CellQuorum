"""Evidence namespace: ``cq.evidence.*`` (planned).

The unified biological-evidence graph (see ``docs/SCIENTIFIC_ENGINEERING_PLAN.md``
Phase I) is not implemented yet. This module reserves the namespace and fails
loudly with a clear "planned" message rather than silently missing, so callers
know the capability is on the roadmap but not available.
"""

from __future__ import annotations

from typing import Any

_PLANNED = (
    "cq.evidence is planned but not implemented yet. The unified biological "
    "evidence graph is a later roadmap phase (see docs/SCIENTIFIC_ENGINEERING_PLAN.md)."
)


def build(*_args: Any, **_kwargs: Any) -> None:
    """Planned: build a unified biological evidence graph. Not yet implemented."""

    raise NotImplementedError(_PLANNED)


__all__ = ["build"]
