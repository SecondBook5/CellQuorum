"""Public API for CellQuorum."""

from __future__ import annotations

# Import the notebook-facing namespaces (thin wrappers over registered stages).
from cellquorum import diag, evidence, pp, tl

# Import the main public pipeline entry point.
from cellquorum.api import run_pipeline

# Import the canonical package version for users and downstream tools.
from cellquorum.version import __version__

# Define the public symbols exposed by `from cellquorum import *`.
__all__: list[str] = ["__version__", "diag", "evidence", "pp", "run_pipeline", "tl"]
