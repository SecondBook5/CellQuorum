"""Compatibility shim — the evidence namespace moved to
:mod:`cellquorum.api.evidence`.

Kept so pre-move imports (``import cellquorum.evidence``,
``from cellquorum.evidence import ...``) keep working unchanged. New code
should import from :mod:`cellquorum.api.evidence` (or use the ``cq.evidence``
re-export).
"""

from __future__ import annotations

from cellquorum.api.evidence import *  # noqa: F401,F403
from cellquorum.api.evidence import __all__  # noqa: F401
