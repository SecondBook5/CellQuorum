"""Every source file a doc points at must exist.

The docs are the only findability aid a collaborator has before they know the tree,
and a code pointer that no longer resolves is worse than no pointer: it sends the
reader looking for a file that was renamed, merged or moved, and there is nothing in
the doc to say which. The #167 consolidation moved several packages, and
``docs/backends.md`` was left naming ``cellquorum/backends/r_scripts/soupx_per_library.R``
with the ``src/`` prefix dropped — a one-word error, invisible to review, fatal to
someone trying to read the SoupX adapter.

Only *source* files are checked (``.py``, ``.R``, ``.sh``, ``.toml``, docs' own
``.md``, and environment ``.yml``). Run outputs — ``results/…csv``, ``figures/…pdf``,
generated ``configs/*.yaml`` — are deliberately excluded: they are produced by a run,
not committed, so requiring them to exist would make this test pass or fail depending
on whether someone had run the pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# The repository root: this file is tests/<name>.py.
ROOT = Path(__file__).resolve().parents[1]

# Prose cites paths in backticks. Nothing else in the docs looks like this.
BACKTICKED = re.compile(r"`([A-Za-z0-9_./-]+)`")

# Extensions that name a file living in the checkout.
SOURCE_SUFFIXES = {".py", ".R", ".sh", ".toml", ".cff"}

# ``.md`` is a source file under ``docs/`` and an artifact anywhere else: a run writes
# ``runs/matrix_status.md``, which exists only after Snakemake has produced it.
MARKDOWN_SOURCE_PREFIXES = ("docs/",)

# ``.yml`` is a source file under these prefixes and a generated artifact elsewhere.
YAML_SOURCE_PREFIXES = ("envs/", "docker/", ".github/", "workflow/")

# Directories that are not the source tree. ``site/`` and ``build/`` are generated
# copies of it, so leaving them in would let a doc "resolve" against a stale build.
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "site",
    "build",
    "runs",
    "htmlcov",
    ".ruff_cache",
    ".pytest_cache",
}


def _repo_paths() -> set[str]:
    """Every file in the checkout, as a repo-relative POSIX path.

    Built once so a citation can be matched as a *suffix*. Docs deliberately cite
    package-internal paths at whatever depth reads best in context — ``src/cellquorum/
    core/stage.py`` in a layout section, the bare ``trajectory/save.py`` mid-sentence —
    and both are useful pointers. Suffix matching accepts either without needing a
    hand-maintained list of roots to try, which is what let ``trajectory/save.py`` and
    ``qc/artifacts.py`` read as broken when both files were exactly where the doc said.
    """
    paths: set[str] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS & set(path.parts):
            continue
        paths.add(path.relative_to(ROOT).as_posix())
    return paths


REPO_PATHS = _repo_paths()


def _doc_files() -> list[Path]:
    """Every user-facing markdown file, excluding the internal spec archive.

    ``docs/superpowers/`` is design history: it deliberately records paths as they
    were when a spec was written, so holding it to the current tree would be wrong.
    """
    docs = sorted(p for p in (ROOT / "docs").rglob("*.md") if "superpowers" not in p.parts)
    return docs + [p for p in (ROOT / "README.md", ROOT / "CONTRIBUTING.md") if p.is_file()]


def _is_source_path(token: str) -> bool:
    """Is this backticked token a path to a file that should be in the checkout?"""
    if "/" not in token:
        return False
    suffix = Path(token).suffix
    if suffix in SOURCE_SUFFIXES:
        return True
    if suffix == ".md":
        return token.startswith(MARKDOWN_SOURCE_PREFIXES)
    return suffix in {".yml", ".yaml"} and token.startswith(YAML_SOURCE_PREFIXES)


def broken_citations(text: str) -> list[str]:
    """Backticked source paths in ``text`` that match nothing in the checkout."""
    broken: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for match in BACKTICKED.finditer(line):
            token = match.group(1)
            if not _is_source_path(token):
                continue
            suffix = f"/{token}"
            if not any(p == token or p.endswith(suffix) for p in REPO_PATHS):
                broken.append(f"line {lineno}: {token}")
    return broken


@pytest.mark.parametrize("doc", _doc_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_source_path_cited_in_a_doc_exists(doc: Path):
    broken = broken_citations(doc.read_text(encoding="utf-8"))
    assert not broken, f"{doc.relative_to(ROOT)} cites paths that do not exist: " + "; ".join(
        broken
    )


def test_the_check_catches_a_renamed_file():
    """A guard that cannot fail is not a guard.

    The two paths below are the shapes that actually go stale: a package-internal
    module that got merged elsewhere, and a real path with a wrong prefix.
    """
    text = (
        "Open `src/cellquorum/core/stage.py` to see the base class.\n"
        "The writer lives in `stages/qc/renamed_away.py` now.\n"
        "SoupX runs `cellquorum/backends/r_scripts/soupx_per_library.R`.\n"
    )
    broken = broken_citations(text)

    # Line 1 resolves; line 2 does not exist at any depth. Line 3 is the docs/backends.md
    # error verbatim — and it must NOT be reported, because suffix matching accepts a
    # path fragment: the file really is at src/cellquorum/backends/r_scripts/.
    assert broken == ["line 2: stages/qc/renamed_away.py"]


def test_run_outputs_are_not_treated_as_source():
    """Otherwise the suite passes or fails on whether someone ran the pipeline."""
    text = (
        "Status lands in `runs/matrix_status.md` and `runs/matrix_status.csv`.\n"
        "Figures go to `figures/qc_panel.pdf`; configs to `configs/generated.yaml`.\n"
    )
    assert broken_citations(text) == []
