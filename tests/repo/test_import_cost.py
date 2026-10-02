"""Guard the lazy-import invariant that keeps the package and CLI fast to start.

CellQuorum's engine reaches :mod:`cellquorum.config.models`, which aggregates all
30 stage config modules, and some of those stages sit on optional heavyweight
stacks (scvi/torch, celltypist, decoupler). Nothing stops a future one-line
``import scvi`` at module scope in a stage package — and when it happens, the cost
does not surface as a test failure. It surfaces as ``cellquorum --version`` taking
five seconds and the test suite taking four minutes to *collect*.

That is exactly what had happened: ``reference_mapping/__init__.py`` probed for scvi
with ``import scvi`` inside a ``try/except ImportError``. The guard stopped the
crash but not the import, so a plain ``import cellquorum`` loaded 7312 modules
(torch, lightning, jax, scvi) in 3.7s, and ``cq --version`` took 4.7s. Probing with
:func:`importlib.util.find_spec` instead answers the same question without
executing the module.

These tests run in a subprocess because import cost is only observable on a cold
interpreter — by the time pytest has imported the rest of the suite, every heavy
module is already in ``sys.modules`` and an in-process assertion would pass
vacuously.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

# Top-level modules that must never be pulled in by a bare `import cellquorum`.
# Each one costs hundreds of milliseconds and belongs to an OPTIONAL backend, so
# importing it eagerly both slows every entry point and defeats the engine-wide
# skip-not-crash invariant (a missing optional dep must skip a stage, not change
# what importing the package costs).
FORBIDDEN_EAGER_IMPORTS = (
    "torch",
    "scvi",
    "lightning",
    "pytorch_lightning",
    "jax",
    "tensorflow",
    "celltypist",
    "decoupler",
)

# Ceiling on modules loaded by a bare `import cellquorum`. The lazy surface loads
# ~4; this is set well above that so ordinary refactors do not trip it, while still
# catching the class of regression that re-imports the analysis stack (~7300).
MAX_MODULES_ON_BARE_IMPORT = 400


def _run_python(code: str) -> str:
    """
    Execute Python source in a cold subprocess and return its stdout.

    Args:
        code: Source to execute with the current interpreter.

    Returns:
        Captured stdout, stripped.

    Raises:
        AssertionError: If the subprocess exits non-zero.
    """

    # Run the snippet with the same interpreter running the tests.
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    # Surface the child's stderr on failure; a bare CalledProcessError hides it.
    assert (
        completed.returncode == 0
    ), f"subprocess failed (exit {completed.returncode})\nstderr:\n{completed.stderr}"

    # Return the trimmed stdout for the caller to parse.
    return completed.stdout.strip()


def _bare_import_report() -> dict[str, object]:
    """
    Import cellquorum in a cold subprocess and report what that cost.

    Returns:
        Mapping with the loaded-module count and which forbidden modules appeared.
    """

    # Measure sys.modules before and after so only cellquorum's own cost counts.
    payload = _run_python(
        "import json, sys\n"
        "before = set(sys.modules)\n"
        "import cellquorum\n"
        "loaded = set(sys.modules) - before\n"
        f"forbidden = {FORBIDDEN_EAGER_IMPORTS!r}\n"
        "print(json.dumps({\n"
        "    'count': len(loaded),\n"
        "    'forbidden': sorted(m for m in forbidden if m in sys.modules),\n"
        "}))\n"
    )

    # Parse the JSON report emitted by the child.
    report: dict[str, object] = json.loads(payload)

    # Hand the parsed report back to the assertions.
    return report


def test_bare_import_pulls_no_optional_heavy_backend() -> None:
    """A plain `import cellquorum` must not import any optional heavy backend."""

    # Collect what the cold import actually loaded.
    report = _bare_import_report()

    # Fail with the offending module names, which point straight at the regression.
    assert report["forbidden"] == [], (
        f"`import cellquorum` eagerly imported optional heavy backends: "
        f"{report['forbidden']}. Probe for optional dependencies with "
        f"importlib.util.find_spec(...) instead of importing them, and keep "
        f"stage-level imports inside the method body."
    )


def test_bare_import_stays_cheap() -> None:
    """A plain `import cellquorum` must stay far below the analysis-stack cost."""

    # Collect the module count from the cold import.
    report = _bare_import_report()

    # Compare against the ceiling, reporting the real number for easy triage.
    assert report["count"] <= MAX_MODULES_ON_BARE_IMPORT, (
        f"`import cellquorum` loaded {report['count']} modules, over the "
        f"{MAX_MODULES_ON_BARE_IMPORT} ceiling. Something on the public surface "
        f"stopped being lazy — check cellquorum/__init__.py and the stage package "
        f"__init__ files."
    )


@pytest.mark.parametrize("name", ["diag", "evidence", "pp", "run_pipeline", "tl", "utils"])
def test_lazy_public_names_still_resolve(name: str) -> None:
    """Every advertised public name must resolve despite being lazily bound.

    Args:
        name: Public attribute promised by ``cellquorum.__all__``.
    """

    # Import the package fresh and resolve the name through __getattr__.
    import cellquorum

    # getattr is what `from cellquorum import <name>` ultimately calls.
    assert getattr(cellquorum, name) is not None


def test_public_surface_matches_lazy_export_table() -> None:
    """``__all__`` and the lazy-export table must not drift apart."""

    # Import the package to read both sides of the contract.
    import cellquorum

    # __all__ carries __version__ (bound eagerly), so exclude it from the compare.
    advertised = set(cellquorum.__all__) - {"__version__"}

    # Every advertised name needs a resolution target, and vice versa.
    assert advertised == set(cellquorum._LAZY_EXPORTS), (
        "cellquorum.__all__ and cellquorum._LAZY_EXPORTS disagree; a public name "
        "was added to one but not the other, so it will raise AttributeError."
    )


def test_utils_public_surface_matches_lazy_export_table() -> None:
    """``cellquorum.utils.__all__`` and its lazy-export table must not drift."""

    # Import the utils surface to read both sides of the contract.
    from cellquorum import utils

    # Compare the advertised surface against the resolution table.
    assert set(utils.__all__) == set(utils._LAZY_EXPORTS), (
        "cellquorum.utils.__all__ and _LAZY_EXPORTS disagree; a re-export was "
        "added to one but not the other."
    )


def test_unknown_attribute_raises_attribute_error() -> None:
    """A lazy module must still reject unknown names like a normal module."""

    # Import the package to probe a name that does not exist.
    import cellquorum

    # PEP 562 __getattr__ must raise AttributeError, not KeyError or ImportError,
    # so `hasattr` probes and typo diagnostics behave as they always did. getattr is
    # used rather than plain attribute access because a bare expression statement is
    # a lint error (ruff B018) even when raising is the point.
    with pytest.raises(AttributeError):
        getattr(cellquorum, "definitely_not_a_real_attribute")  # noqa: B009


def test_cli_help_does_not_import_the_engine() -> None:
    """`--help` and `--version` must not pay for the config/stage aggregation.

    The CLI module imports the planner, config loader, and pipeline API inside each
    command body rather than at module scope. This asserts that stays true: the
    first thing a new user runs is ``--help``, and it should be instant.
    """

    # Import only the CLI module, then report whether the engine came with it.
    payload = _run_python(
        "import json, sys\n"
        "import cellquorum.cli.app\n"
        "print(json.dumps({\n"
        "    'models': 'cellquorum.config.models' in sys.modules,\n"
        "    'planner': 'cellquorum.core.planner' in sys.modules,\n"
        "    'api': 'cellquorum.api' in sys.modules,\n"
        "}))\n"
    )

    # Parse the child's report.
    loaded = json.loads(payload)

    # None of the engine modules should be present after importing the CLI alone.
    assert not any(loaded.values()), (
        f"importing cellquorum.cli.app pulled in the engine: {loaded}. Move the "
        f"engine imports back inside the command bodies."
    )
