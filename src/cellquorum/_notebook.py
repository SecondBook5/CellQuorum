"""Compatibility shim — the notebook adapter moved to
:mod:`cellquorum.api._notebook`.

Kept so pre-move imports (``from cellquorum._notebook import ...``) keep
working unchanged. New code should import from
:mod:`cellquorum.api._notebook`.
"""

from __future__ import annotations

from cellquorum.api._notebook import *  # noqa: F401,F403
from cellquorum.api._notebook import __all__  # noqa: F401
