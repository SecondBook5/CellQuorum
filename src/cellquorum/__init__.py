"""Public API for CellQuorum."""

from __future__ import annotations

# Import the reusable analytical building blocks (documented, versioned surface
# for power-user scripts) so `cq.utils.*` resolves without a separate import.
from cellquorum import utils

# Import the public surface from the api package: the notebook-facing
# namespaces (thin wrappers over registered stages) and the pipeline entry.
from cellquorum.api import diag, evidence, pp, run_pipeline, tl

# Import the canonical package version for users and downstream tools.
from cellquorum.version import __version__

# Define the public symbols exposed by `from cellquorum import *`.
__all__: list[str] = [
    "__version__",
    "diag",
    "evidence",
    "pp",
    "run_pipeline",
    "tl",
    "utils",
]
