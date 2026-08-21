"""Compatibility shim — the preprocessing namespace moved to
:mod:`cellquorum.api.pp`.

Kept so pre-move imports (``import cellquorum.pp``,
``from cellquorum.pp import ...``) keep working unchanged. New code should
import from :mod:`cellquorum.api.pp` (or use the ``cq.pp`` re-export).
"""

from __future__ import annotations

from cellquorum.api.pp import *  # noqa: F401,F403
from cellquorum.api.pp import __all__  # noqa: F401
