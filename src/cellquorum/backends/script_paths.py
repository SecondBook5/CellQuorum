"""Resolution of bundled backend script paths.

Bundled R scripts live in ``cellquorum/backends/r_scripts/``. Stage modules used
to reach them with a relative ``Path(__file__).parent.parent...`` chain, which
encodes the stage module's *nesting depth* in a constant. When stages were
regrouped into category packages (``stages/differential_abundance`` ->
``stages/comparative/differential_abundance``) every one of those chains became
off by one and silently resolved to a ``stages/backends/r_scripts`` directory
that has never existed. Each affected method then failed at run time with
"R script not found", one stage at a time.

Resolving from the backends package instead means the path cannot depend on where
the caller lives, so moving or regrouping stages can never break it again.
"""

from __future__ import annotations

from pathlib import Path

# Anchored to THIS module, which sits in the same package as the scripts.
R_SCRIPTS_DIR = Path(__file__).parent / "r_scripts"


def r_script_path(name: str) -> Path:
    """Absolute path to the bundled R script ``name`` (e.g. ``"choir.R"``).

    Returns the path whether or not it exists: callers report a missing script as
    a skip with their own stage context, which is more useful than an exception
    raised from here. Use :func:`r_script_exists` to check.
    """
    return R_SCRIPTS_DIR / name


def r_script_exists(name: str) -> bool:
    """Is the bundled R script ``name`` present?"""
    return r_script_path(name).is_file()


def available_r_scripts() -> list[str]:
    """Names of every bundled R script, sorted. Empty if the directory is absent."""
    if not R_SCRIPTS_DIR.is_dir():
        return []
    return sorted(p.name for p in R_SCRIPTS_DIR.glob("*.R"))


__all__ = [
    "R_SCRIPTS_DIR",
    "available_r_scripts",
    "r_script_exists",
    "r_script_path",
]
