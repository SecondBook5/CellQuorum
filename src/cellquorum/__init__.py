"""Public API for CellQuorum."""

from __future__ import annotations

# Import the canonical package version for users and downstream tools.
from cellquorum.version import __version__

# Define the public symbols exposed by `from cellquorum import *`.
__all__: list[str] = ["__version__"]