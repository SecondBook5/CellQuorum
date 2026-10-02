"""Every stage writes figures through ONE hardened save_figure.

This file exists because four stage families each carried their own private
``save_figure`` — a bare loop over ``fig.savefig`` — while
``figstyle.save_figure`` was hardened against a real failure the bare loop had
already caused. On the LEC arm the velocity stream figure raised "Can only
output finite numbers in PDF" partway through writing, and the trajectory copy
left behind a 38 KB truncated ``velocity_stream.pdf`` and never attempted the
PNG. Three of the four copies would have done the same thing on the next figure
that raised.

The tests below pin the consolidation itself rather than re-testing the write
(``test_figstyle_contract.py`` owns the atomicity and all-formats-attempted
properties): a future stage that hand-rolls its own writer, or a re-export that
silently stops pointing at the shared one, fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cellquorum.visualization import figio, figstyle

# The installed ``cellquorum`` package root, so paths below read as
# ``visualization/figstyle.py`` regardless of where the checkout lives.
SRC = Path(figio.__file__).resolve().parents[1]

# The four stage-facing modules that used to hold a copy. Named explicitly so a
# module dropping its re-export is a failure, not a silently smaller test.
RE_EXPORTING_MODULES = (
    "cellquorum.stages.trajectory.viz._helpers",
    "cellquorum.stages.comparative.enrichment.viz.io",
    "cellquorum.stages.cell_cell_communication.viz._io",
    "cellquorum.stages.integration.embeddings.plots",
)


@pytest.mark.parametrize("module_path", RE_EXPORTING_MODULES)
def test_stage_viz_modules_re_export_the_shared_writer(module_path: str):
    import importlib

    module = importlib.import_module(module_path)
    # Identity, not equality: a re-implementation that happens to behave the same
    # today is the situation this consolidation removed.
    assert module.save_figure is figstyle.save_figure
    assert module.figure_artifacts is figio.figure_artifacts


def test_no_module_outside_figstyle_defines_save_figure():
    """The whole package, parsed — not just the four modules known to have copied.

    AST rather than grep so a definition inside a class or an ``if`` block still
    counts, and so a mere mention of the name in a comment or string does not.
    """
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in (
                "save_figure",
                "figure_artifacts",
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno} defines {node.name}")

    allowed = {
        "visualization/figstyle.py": {"save_figure"},
        "visualization/figio.py": {"figure_artifacts"},
    }
    unexpected = [
        entry
        for entry in offenders
        if entry.split(" defines ")[1] not in allowed.get(entry.split(":")[0], set())
    ]
    assert not unexpected, "hand-rolled figure writer(s): " + "; ".join(unexpected)


def test_figure_artifacts_reports_only_the_paths_that_were_written(tmp_path: Path):
    """It must describe what is on disk, not what was requested.

    ``save_figure`` returns fewer paths than formats when a format fails, and the
    artifact list feeds the run's provenance — claiming a PDF that failed to
    render would put a nonexistent file in the manifest.
    """
    written = [tmp_path / "fig.png"]
    artifacts = figio.figure_artifacts(written, name="trajectory_figure", description="velocity")

    assert len(artifacts) == 1
    assert artifacts[0].path == written[0]
    assert artifacts[0].kind == "figure"
    assert artifacts[0].name == "trajectory_figure"
    assert figio.figure_artifacts([], name="x", description="y") == []


def test_no_module_outside_figstyle_calls_savefig_directly():
    """``atomic_savefig`` is the only place a figure reaches the filesystem.

    The four hand-rolled ``save_figure`` copies were the loud half of this
    problem. The quiet half was five *by-path* writers scattered across the
    package — the PCA scree plot, the QC publication panels, the QC table
    typesetter, the two reference-mapping diagnostics — each a bare
    ``fig.savefig``, each therefore unable to leave anything but a truncated
    file behind on a mid-write failure, and each shipping PNG with no vector
    twin. A run's ``figures/`` directory held 60 PNGs and 57 PDFs; the three
    missing PDFs were exactly the ones written this way.

    AST rather than grep, for the same reasons as the test above: a call inside
    a nested function still counts, and the word appearing in a docstring does
    not.
    """
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "savefig"
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")

    # figstyle owns the one write mechanic, so it is the one place allowed to
    # touch savefig. Everything else goes through it.
    unexpected = [
        entry for entry in offenders if not entry.startswith("visualization/figstyle.py:")
    ]
    assert not unexpected, (
        "bare fig.savefig outside figstyle: "
        + "; ".join(unexpected)
        + " — use figstyle.save_figure (directory + stem), save_cellquorum_figure "
        "(full path), or atomic_savefig (custom savefig kwargs)."
    )


def test_only_one_savefig_call_remains_in_figstyle():
    """And within figstyle, only ``atomic_savefig`` writes.

    ``save_figure``, ``save_cellquorum_figure`` and ``save_publication_figure``
    were three independent implementations of the same write: three
    ``mkdir(parents=True)`` calls, three tight-layout decisions, three white
    backgrounds. Hardening one reached only one. They now differ in their
    *interface* — directory+stem, full path, caller-set facecolor — and share
    their mechanic.
    """
    figstyle_source = (SRC / "visualization" / "figstyle.py").read_text(encoding="utf-8")
    tree = ast.parse(figstyle_source)

    writers = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "savefig"
    ]
    assert len(writers) == 1, f"expected one savefig in figstyle, found {writers}"

    # ...and it is inside atomic_savefig, not beside it.
    enclosing = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.lineno <= writers[0] <= (node.end_lineno or node.lineno)
    ]
    assert enclosing == ["atomic_savefig"], enclosing
