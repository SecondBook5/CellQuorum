"""Where the repository is, independent of how deep a test file sits.

Tests live in folders that mirror the source tree, so ``Path(__file__).parents[N]`` would change
whenever a file moves. Find the root by the file that defines it instead.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
