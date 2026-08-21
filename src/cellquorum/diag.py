"""Compatibility shim — the diagnostics namespace moved to
:mod:`cellquorum.api.diag`.

Kept so pre-move imports (``import cellquorum.diag``,
``from cellquorum.diag import ...``) keep working unchanged. New code should
import from :mod:`cellquorum.api.diag` (or use the ``cq.diag`` re-export).
"""

from __future__ import annotations

from cellquorum.api.diag import *  # noqa: F401,F403
from cellquorum.api.diag import __all__  # noqa: F401
