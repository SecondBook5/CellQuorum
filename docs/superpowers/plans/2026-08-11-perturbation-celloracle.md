# Perturbation Stage (in-silico KO via CellOracle) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generalizable `perturbation` stage that performs in-silico transcription-factor knockouts with CellOracle — infers its own GRN from observational scRNA + a built-in promoter base GRN, simulates each KO, and ranks knockouts by disease→healthy shift — running end-to-end from one config and skipping cleanly when its isolated env is absent.

**Architecture:** One-for-one mirror of the shipped `grn`/pySCENIC stage: a `MethodDispatchStage` subclass + `StrictBaseModel` config + `AnalysisMethod` + isolated-env subprocess backend + a bundled in-env CLI script, wired through the four canonical seams (config/models, executor registry, planner canonical order, backend registry). Heavy CellOracle GRN-fit + KO simulation runs in an isolated `celloracle_env` via `micromamba run`; publication figures render in-process (cellquorum env) via `figstyle`. Design spec: `docs/superpowers/specs/2026-08-11-perturbation-celloracle-design.md`.

**Tech Stack:** Python 3.12 (cellquorum env), pandas, anndata, matplotlib/seaborn via `cellquorum.visualization.figstyle`; CellOracle in a frozen isolated micromamba env (`celloracle_env`); pytest with fakes (no real CellOracle in CI).

## Global Constraints

- **Zero study-specific biology in `src`.** No hardcoded gene lists, no "lymphedema"/"disease" strings, no organism baked into code. Generic obs-key fallbacks only: `cluster_key` = config → `cell_type` → `leiden` → `"all"`; `rep_key` = config → `X_pca` → `X_pca_harmony`; `embedding_key` = config → `X_umap`.
- **Skip-not-crash.** Every failure path returns `MethodSkip` — the method `_run` never raises. The in-env script writes a sentinel + `sys.exit(0)` on a harmless skip, exits non-zero only on a real CLI failure. Absence of `condition_key`/`healthy_label` is NOT a skip — degrade to direction-agnostic magnitude outputs.
- **Determinism.** Everything seeded (`seed`, default 0): kNN imputation, signal propagation, any sampling.
- **Dual-format figures** via `figstyle.save_figure` (writes PDF+PNG, closes the fig); palette from `figstyle.CATEGORICAL_PALETTE`. Never import a `theme` module.
- **Subprocess isolation.** `micromamba run -n <env> python <script> [args]`; `shutil.which` launcher probe; `subprocess.run(check=False)`; never import `celloracle` into the cellquorum process.
- **Method/stage identity.** stage name `perturbation`, `stage_category = "perturbation"`, method `name = "celloracle"`, backend `name = "celloracle"`, env `celloracle_env`.
- **Placement:** planner canonical order — after `grn`, before `trajectory`.
- **Do NOT push to any remote. Do NOT add any "Co-Authored-By" / "Generated with Claude" trailer to commits.** `docs/superpowers/` is committed locally only (it is gitignored; force-add).
- **Do NOT touch** the pre-existing dirty files (`configs/le_global.yaml`, `src/cellquorum/qc/visualization.py`, `src/cellquorum/reference_mapping/diagnostics.py`) or the pre-existing untracked scripts (`scripts/plot_integration_benchmark.py`, `scripts/run_annotation_diagnostics.py`, `scripts/run_integration_benchmark.py`).
- **TDD:** every task writes the failing test first, watches it fail, implements minimally, watches it pass, commits.
- **Commit message convention:** conventional-commit prefix (`feat:`, `test:`, `docs:`), no trailer.

---

### Task 1: `PerturbationConfig` model

**Files:**
- Create: `src/cellquorum/perturbation/__init__.py` (empty placeholder for now — Task 8 fills it; create the directory)
- Create: `src/cellquorum/perturbation/config.py`
- Test: `tests/test_perturbation_config.py`

**Interfaces:**
- Produces: `PerturbationConfig(StrictBaseModel)` with the exact fields below; imported by later tasks as `from cellquorum.perturbation.config import PerturbationConfig`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the PerturbationConfig model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.perturbation.config import PerturbationConfig


def test_defaults() -> None:
    c = PerturbationConfig()
    assert c.enabled is True
    assert c.method == "celloracle"
    assert c.layer == "counts"
    assert c.organism == "human"
    assert c.cluster_key is None
    assert c.embedding_key is None
    assert c.rep_key is None
    assert c.condition_key is None
    assert c.healthy_label is None
    assert c.tf_list is None
    assert c.n_top_targets == 20
    assert c.knn_n_neighbors == 200
    assert c.n_propagation == 3
    assert c.min_cells_total == 200
    assert c.seed == 0
    assert c.env_name == "celloracle_env"
    assert c.launcher == "micromamba"
    assert c.timeout_seconds == 10800


def test_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        PerturbationConfig(bogus=1)


def test_tf_list_accepts_list() -> None:
    c = PerturbationConfig(tf_list=["PROX1", "PIEZO1"])
    assert c.tf_list == ["PROX1", "PIEZO1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_perturbation_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cellquorum.perturbation'`

- [ ] **Step 3: Create the package + config**

Create `src/cellquorum/perturbation/__init__.py` as an empty file (one line: `"""In-silico perturbation (CellOracle) stage."""`).

Create `src/cellquorum/perturbation/config.py`:

```python
"""Configuration for the perturbation (in-silico KO) stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class PerturbationConfig(StrictBaseModel):
    """In-silico transcription-factor knockout via CellOracle in an isolated env.

    Infers a simulation-ready GRN from observational counts + a built-in promoter
    base GRN, simulates each TF knockout by zeroing the TF and propagating the
    signal, and ranks knockouts by how strongly they shift disease cells toward the
    healthy state. Skips cleanly when the env or base GRN is unavailable.

    Attributes:
        enabled: Whether the stage runs (enabled by default).
        method: Perturbation method registry key (celloracle).
        layer: Layer holding raw counts for CellOracle.
        organism: Built-in base GRN organism (human/mouse).
        cluster_key: GRN cluster grouping (falls back cell_type -> leiden -> "all").
        embedding_key: Shift-vector embedding (falls back X_umap).
        rep_key: PCA/kNN representation (falls back X_pca -> X_pca_harmony).
        condition_key: Disease/healthy obs column; absent -> direction-agnostic.
        healthy_label: Target condition value; absent -> direction-agnostic.
        tf_list: TFs to knock out; None -> systematic screen of all fitted TFs.
        n_top_targets: Ranked-table / figure cutoff.
        knn_n_neighbors: Neighbors for CellOracle kNN imputation.
        n_propagation: Signal-propagation iterations for the KO simulation.
        min_cells_total: Minimum total cells required to attempt inference.
        seed: Random seed for reproducibility.
        env_name: Name of the isolated micromamba environment.
        launcher: Environment launcher (micromamba).
        timeout_seconds: CellOracle execution timeout in seconds.
    """

    # Whether this stage runs.
    enabled: bool = True

    # Selected perturbation method (registry key under stage_category 'perturbation').
    method: str = "celloracle"

    # Layer holding raw counts for CellOracle.
    layer: str = "counts"

    # Built-in base GRN organism (human/mouse).
    organism: str = "human"

    # GRN cluster grouping (falls back cell_type -> leiden -> "all").
    cluster_key: str | None = None

    # Shift-vector embedding (falls back X_umap).
    embedding_key: str | None = None

    # PCA/kNN representation (falls back X_pca -> X_pca_harmony).
    rep_key: str | None = None

    # Disease/healthy obs column; absent -> direction-agnostic.
    condition_key: str | None = None

    # Target condition value; absent -> direction-agnostic.
    healthy_label: str | None = None

    # TFs to knock out; None -> systematic screen of all fitted TFs.
    tf_list: list[str] | None = None

    # Ranked-table / figure cutoff.
    n_top_targets: int = 20

    # Neighbors for CellOracle kNN imputation.
    knn_n_neighbors: int = 200

    # Signal-propagation iterations for the KO simulation.
    n_propagation: int = 3

    # Minimum total cells required to attempt inference.
    min_cells_total: int = 200

    # Random seed for reproducibility.
    seed: int = 0

    # Name of the isolated micromamba environment.
    env_name: str = "celloracle_env"

    # Environment launcher (micromamba).
    launcher: str = "micromamba"

    # CellOracle execution timeout (seconds).
    timeout_seconds: int = 10800


__all__ = ["PerturbationConfig"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_perturbation_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/perturbation/__init__.py src/cellquorum/perturbation/config.py tests/test_perturbation_config.py
git commit -m "feat: add PerturbationConfig model for in-silico KO stage"
```

---

### Task 2: `CellOracleBackend` subprocess backend

**Files:**
- Create: `src/cellquorum/backends/celloracle_backend.py`
- Create: `src/cellquorum/backends/celloracle_scripts/__init__.py` (empty package marker so the scripts dir is importable/packaged)
- Test: `tests/test_celloracle_backend.py`

**Interfaces:**
- Consumes: `cellquorum.backends.base.{BackendRequirement, BackendStatus, BaseBackend}` (same as `PyscenicBackend`).
- Produces:
  - `CellOracleBackend(BaseBackend)` dataclass — `name="celloracle"`, `kind="external"`, `env_name="celloracle_env"`, `launcher="micromamba"`, `timeout_seconds=60`, `script_timeout_seconds=10800`; methods `status()`, `run_script(script_path, args=None, *, timeout=None)`, `_launcher_available()`, `_py_module_available(module_name)`, static `_valid_module_name(module_name)`.
  - `build_celloracle_backend(*, env_name="celloracle_env", launcher="micromamba", timeout_seconds=60) -> CellOracleBackend`.
  - `CELLORACLE_KO_PY: Path` — path to the bundled in-env script (created in Task 3; referenced here as `_CELLORACLE_SCRIPTS_DIR / "celloracle_ko.py"`, so it resolves even before the script file exists).

This backend is a near-verbatim copy of `src/cellquorum/backends/pyscenic_backend.py` with pySCENIC→CellOracle substitutions. Copy that file's structure exactly (the `run_script`, `_launcher_available`, `_py_module_available`, `_valid_module_name` bodies are identical).

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the CellOracle subprocess backend (no real micromamba/celloracle)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cellquorum.backends.celloracle_backend import (
    CELLORACLE_KO_PY,
    CellOracleBackend,
    build_celloracle_backend,
)


def test_build_defaults() -> None:
    b = build_celloracle_backend()
    assert b.name == "celloracle"
    assert b.kind == "external"
    assert b.env_name == "celloracle_env"
    assert b.launcher == "micromamba"
    assert b.script_timeout_seconds == 10800


def test_run_script_builds_micromamba_argv(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text("print('hi')\n")
    b = build_celloracle_backend()

    captured = {}

    def fake_which(_name):  # noqa: ANN001, ANN202
        return "/usr/bin/micromamba"

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003, ANN202
        captured["cmd"] = cmd
        captured["check"] = kwargs.get("check")

        class R:  # noqa: D401
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr("cellquorum.backends.celloracle_backend.shutil.which", fake_which)
    monkeypatch.setattr("cellquorum.backends.celloracle_backend.subprocess.run", fake_run)

    b.run_script(script, ["--h5ad", "x.h5ad"], timeout=123)
    assert captured["cmd"][:5] == ["micromamba", "run", "-n", "celloracle_env", "python"]
    assert str(script) in captured["cmd"]
    assert captured["cmd"][-2:] == ["--h5ad", "x.h5ad"]
    assert captured["check"] is False


def test_run_script_missing_script_raises(tmp_path: Path) -> None:
    b = build_celloracle_backend()
    with pytest.raises(FileNotFoundError):
        b.run_script(tmp_path / "does_not_exist.py", [])


def test_run_script_missing_launcher_raises(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text("x=1\n")
    b = build_celloracle_backend()
    monkeypatch.setattr(
        "cellquorum.backends.celloracle_backend.shutil.which", lambda _n: None
    )
    with pytest.raises(FileNotFoundError):
        b.run_script(script, [])


def test_invalid_module_name_rejected() -> None:
    b = build_celloracle_backend()
    with pytest.raises(ValueError):
        b._py_module_available("bad name; rm -rf")


def test_ko_script_path_points_into_scripts_dir() -> None:
    assert CELLORACLE_KO_PY.name == "celloracle_ko.py"
    assert CELLORACLE_KO_PY.parent.name == "celloracle_scripts"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_celloracle_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cellquorum.backends.celloracle_backend'`

- [ ] **Step 3: Create the backend**

Create `src/cellquorum/backends/celloracle_scripts/__init__.py` (one line: `"""In-env CellOracle helper scripts."""`).

Create `src/cellquorum/backends/celloracle_backend.py` by copying `src/cellquorum/backends/pyscenic_backend.py` and applying these substitutions (keep every method body — `run_script`, `_launcher_available`, `_py_module_available`, `_valid_module_name` — byte-identical to the pySCENIC original except the names/strings below):

- Module docstring: describe CellOracle instead of pySCENIC.
- `_PYSCENIC_SCRIPTS_DIR` → `_CELLORACLE_SCRIPTS_DIR = Path(__file__).parent / "celloracle_scripts"`.
- Class `PyscenicBackend` → `CellOracleBackend`.
- Defaults: `name: str = "celloracle"`, `env_name: str = "celloracle_env"`, `script_timeout_seconds: int = 10800`. Keep `kind="external"`, `launcher="micromamba"`, `timeout_seconds=60`.
- `requirement_list`: two requirements —
  - `BackendRequirement(name="micromamba", requirement_type="executable", required=True, install_hint="Install micromamba (or set launcher to conda/mamba).")`
  - `BackendRequirement(name="celloracle", requirement_type="other", required=True, install_hint="Create a frozen isolated env with CellOracle: `micromamba create -n celloracle_env -c conda-forge celloracle`. The promoter base GRN (hg38/mm10) ships with CellOracle.")`
- `status()`: probe `self._py_module_available("celloracle")` (append `"celloracle"` to `missing` when absent). Body otherwise identical.
- Every error string / helper-script message: replace "pyscenic" with "celloracle".
- `build_pyscenic_backend` → `build_celloracle_backend` (same signature/defaults, returns `CellOracleBackend`).
- Bottom-of-module script paths: replace the two `PYSCENIC_*_PY` constants with a single `CELLORACLE_KO_PY = _CELLORACLE_SCRIPTS_DIR / "celloracle_ko.py"`.
- `__all__ = ["CELLORACLE_KO_PY", "CellOracleBackend", "build_celloracle_backend"]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_celloracle_backend.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/backends/celloracle_backend.py src/cellquorum/backends/celloracle_scripts/__init__.py tests/test_celloracle_backend.py
git commit -m "feat: add CellOracle isolated-env subprocess backend"
```

---

### Task 3: In-env CellOracle KO script

**Files:**
- Create: `src/cellquorum/backends/celloracle_scripts/celloracle_ko.py`
- Test: `tests/test_celloracle_ko_script.py`

**Interfaces:**
- Produces (module-level, importable without a real CellOracle install): `build_parser() -> argparse.ArgumentParser`; `write_skip(out_dir: Path, tag: str, reason: str) -> None` (writes empty `perturbation_ranking.csv` schema + `perturbation_SKIPPED_{tag}.txt`); `write_ranking(out_dir: Path, tag: str, rows: list[dict]) -> Path` (writes `perturbation_ranking.csv` with columns `tf,score,n_cells,direction`); `main() -> None`.
- The heavy CellOracle import happens **inside `main()`** (after arg-parse, inside a try/except that calls `write_skip` + `sys.exit(0)` on `ImportError`/`Exception`), so the module imports fine in CI where CellOracle is absent. This mirrors `pyscenic_grn.py`'s "import inside main, graceful skip" structure.

**Structure (mirror `pyscenic_grn.py`'s two-failure-regime contract):**
1. `build_parser()` adds: `--h5ad` (required), `--out-dir` (required), `--tag` (required), `--organism` (default `human`), `--cluster-key` (default `""`), `--rep-key` (default `""`), `--embedding-key` (default `""`), `--condition-key` (default `""`), `--healthy-label` (default `""`), `--tf-list` (default `""`, space/comma-separated), `--n-top-targets` (int, default 20), `--knn-n-neighbors` (int, default 200), `--n-propagation` (int, default 3), `--seed` (int, default 0).
2. `main()`:
   - Parse args; `out_dir = Path(args.out_dir)`.
   - `try: import celloracle` (+ numpy/pandas/scanpy as CellOracle needs) `except Exception as e: write_skip(out_dir, args.tag, f"celloracle import failed: {type(e).__name__}: {e}"); sys.exit(0)`.
   - Load the base GRN for `--organism`; if unavailable → `write_skip(...); sys.exit(0)`.
   - GRN inference → KO simulation → scoring, exactly as the design's three steps. On a real CellOracle failure (raised exception after imports succeeded), write `perturbation_FAILED_{tag}.txt` + `sys.exit(1)` (fail-loud). On the harmless "nothing fit" case, `write_skip` + `sys.exit(0)`.
   - Scoring: if `args.condition_key` and `args.healthy_label` are both non-empty → directional (project per-cell shift onto the diseased→healthy centroid axis in the embedding), `direction="directional"`; else magnitude, `direction="magnitude"`. Write via `write_ranking`.
   - Also write `shift_vectors_<TF>.parquet` (index = obs_names, columns e.g. `dx,dy`) per TF, `grn_summary.csv`, and per-cluster `links_<cluster>.parquet` when produced.

The CI test exercises ONLY the pure helpers + the graceful-skip path (no real CellOracle). Keep all CellOracle-specific calls behind the in-`main` import so this file imports cleanly in `cellquorum-dev`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the in-env CellOracle KO script's pure helpers + skip path.

No real CellOracle in CI — only argument parsing, the ranking-schema writer, and
the graceful-skip writer are exercised.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "cellquorum"
    / "backends"
    / "celloracle_scripts"
    / "celloracle_ko.py"
)


def _load():
    return runpy.run_path(str(SCRIPT))


def test_module_imports_without_celloracle() -> None:
    ns = _load()
    assert "main" in ns
    assert "build_parser" in ns
    assert "write_skip" in ns
    assert "write_ranking" in ns


def test_parser_has_expected_args() -> None:
    ns = _load()
    parser = ns["build_parser"]()
    args = parser.parse_args(
        ["--h5ad", "a.h5ad", "--out-dir", "o", "--tag", "t"]
    )
    assert args.h5ad == "a.h5ad"
    assert args.out_dir == "o"
    assert args.tag == "t"
    assert args.organism == "human"
    assert args.n_top_targets == 20
    assert args.knn_n_neighbors == 200
    assert args.n_propagation == 3
    assert args.seed == 0


def test_write_skip_creates_empty_ranking_and_marker(tmp_path: Path) -> None:
    ns = _load()
    ns["write_skip"](tmp_path, "t", "no base GRN")
    ranking = tmp_path / "perturbation_ranking.csv"
    marker = tmp_path / "perturbation_SKIPPED_t.txt"
    assert ranking.exists()
    assert ranking.read_text().splitlines()[0] == "tf,score,n_cells,direction"
    # header only, no data rows
    assert len(ranking.read_text().strip().splitlines()) == 1
    assert marker.exists()
    assert "no base GRN" in marker.read_text()


def test_write_ranking_writes_rows(tmp_path: Path) -> None:
    ns = _load()
    rows = [
        {"tf": "PROX1", "score": 0.9, "n_cells": 100, "direction": "directional"},
        {"tf": "PIEZO1", "score": 0.5, "n_cells": 100, "direction": "directional"},
    ]
    out = ns["write_ranking"](tmp_path, "t", rows)
    assert Path(out).exists()
    text = Path(out).read_text()
    assert text.splitlines()[0] == "tf,score,n_cells,direction"
    assert "PROX1" in text and "PIEZO1" in text


def test_script_skips_gracefully_when_celloracle_absent(tmp_path: Path) -> None:
    # Running end-to-end with no celloracle installed must exit 0 and write a skip.
    h5ad = tmp_path / "in.h5ad"
    h5ad.write_text("")  # never read — import gate trips first
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--h5ad",
            str(h5ad),
            "--out-dir",
            str(out_dir),
            "--tag",
            "t",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert (out_dir / "perturbation_ranking.csv").exists()
    assert (out_dir / "perturbation_SKIPPED_t.txt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_celloracle_ko_script.py -v`
Expected: FAIL — script file does not exist (`runpy` raises `FileNotFoundError`).

- [ ] **Step 3: Write the script**

Create `src/cellquorum/backends/celloracle_scripts/celloracle_ko.py`. Header + helpers (the pure, CI-tested parts) exactly:

```python
#!/usr/bin/env python
"""CellQuorum in-env CellOracle in-silico KO backend.

Runs CellOracle's three-phase workflow in an isolated frozen env: (1) infer a
simulation-ready GRN from counts + the built-in promoter base GRN, (2) simulate
each TF knockout by zeroing the TF and propagating the signal, (3) score/rank the
knockouts by disease->healthy shift.

Two failure regimes, deliberately distinct (mirrors the pySCENIC backend):
  - NOT CONFIGURED (celloracle absent, base GRN unavailable, nothing fit): writes
    an empty perturbation_ranking.csv + perturbation_SKIPPED_{tag}.txt and exits 0
    -> harmless skip, nothing else affected.
  - REAL FAILURE (celloracle present but the GRN fit / simulation itself raises):
    writes perturbation_FAILED_{tag}.txt and exits NON-ZERO.

Scoring: with --condition-key AND --healthy-label, each TF's score is the mean
projection of its per-cell shift vectors onto the unit diseased->healthy centroid
axis in the embedding (direction="directional"); otherwise the mean shift
magnitude (direction="magnitude").
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_RANKING_HEADER = ["tf", "score", "n_cells", "direction"]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the CellOracle KO script."""
    p = argparse.ArgumentParser(description="CellOracle in-silico TF knockout")
    p.add_argument("--h5ad", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--organism", default="human")
    p.add_argument("--cluster-key", default="")
    p.add_argument("--rep-key", default="")
    p.add_argument("--embedding-key", default="")
    p.add_argument("--condition-key", default="")
    p.add_argument("--healthy-label", default="")
    p.add_argument("--tf-list", default="", help="space/comma-separated; empty -> screen all")
    p.add_argument("--n-top-targets", type=int, default=20)
    p.add_argument("--knn-n-neighbors", type=int, default=200)
    p.add_argument("--n-propagation", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p


def write_skip(out_dir: Path, tag: str, reason: str) -> None:
    """Write the empty ranking schema + a skip marker (harmless configuration skip)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "perturbation_ranking.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_RANKING_HEADER)
    (out_dir / f"perturbation_SKIPPED_{tag}.txt").write_text(
        f"CellOracle skipped (isolated backend, no downstream effect): {reason}\n",
        encoding="utf-8",
    )
    print(f"[celloracle] SKIPPED gracefully: {reason}")


def write_ranking(out_dir: Path, tag: str, rows: list[dict]) -> Path:
    """Write the ranked-target table. Returns the written path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "perturbation_ranking.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_RANKING_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _RANKING_HEADER})
    return path


def _parse_tf_list(raw: str) -> list[str]:
    """Parse a space/comma-separated TF list; empty -> [] (means screen all)."""
    if not raw:
        return []
    return [t for t in raw.replace(",", " ").split() if t]


def main() -> None:
    """Run the CellOracle KO workflow, or skip gracefully."""
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)

    # Heavy import gate: absence of CellOracle is a harmless skip, not a failure.
    try:
        import celloracle as co  # noqa: F401
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import scanpy as sc  # noqa: F401
    except Exception as e:  # pragma: no cover - exercised only where celloracle absent
        write_skip(out_dir, args.tag, f"celloracle import failed: {type(e).__name__}: {e}")
        sys.exit(0)

    try:
        _run_celloracle(args)  # pragma: no cover - requires a real celloracle env
    except Exception as e:  # pragma: no cover
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"perturbation_FAILED_{args.tag}.txt").write_text(
            f"celloracle KO failed: {type(e).__name__}: {e}\n", encoding="utf-8"
        )
        print(f"[celloracle] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

Then add the CellOracle-dependent worker `_run_celloracle(args)` (marked `# pragma: no cover` — not exercised in CI). It must, in order:
1. Read the h5ad (`anndata.read_h5ad`); resolve `cluster_key`/`rep_key`/`embedding_key` (use the arg if non-empty, else the CellOracle defaults). If the base GRN for `args.organism` is unavailable, call `write_skip(out_dir, args.tag, "base GRN unavailable for organism <organism>")` and `return`.
2. Build the `co.Oracle`, import the AnnData, `perform_PCA()`, `knn_imputation(n_neighbors=args.knn_n_neighbors, ...)` seeded with `args.seed`; fit cluster-specific `Links`. Write `grn_summary.csv` and per-cluster `links_<cluster>.parquet`.
3. TF set = `_parse_tf_list(args.tf_list)` if non-empty, else all TFs present in both the fitted GRN and `adata.var_names`. If empty → `write_skip(out_dir, args.tag, "no TFs to screen")`; `return`.
4. For each TF: `oracle.simulate_shift(perturb_condition={tf: 0.0}, n_propagation=args.n_propagation)`, compute per-cell shift vectors on the embedding, write `shift_vectors_<tf>.parquet` (index = obs_names).
5. Score each TF (directional vs magnitude per the docstring), assemble `rows` sorted by `score` descending, and `write_ranking(out_dir, args.tag, rows)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_celloracle_ko_script.py -v`
Expected: PASS (5 tests) — the skip path runs because `celloracle` is absent in `cellquorum-dev`.

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/backends/celloracle_scripts/celloracle_ko.py tests/test_celloracle_ko_script.py
git commit -m "feat: add in-env CellOracle KO script (parser, ranking writer, skip path)"
```

---

### Task 4: `perturbation_figures.py` (house-styled figures)

**Files:**
- Create: `src/cellquorum/perturbation/perturbation_figures.py`
- Test: `tests/test_perturbation_figures.py`

**Interfaces:**
- Consumes: `cellquorum.visualization.figstyle` (`set_style()`, `save_figure(fig, out_dir, stem) -> list[Path]`, `CATEGORICAL_PALETTE`).
- Produces four functions, each returning `list[Path]` (PNG+PDF) and returning `[]` on empty/degenerate input, never raising to the caller:
  - `plot_target_ranking(ranking_df: pd.DataFrame, out_dir, *, n_top: int = 20, name: str = "perturbation_target_ranking") -> list[Path]` — horizontal lollipop/bar of the top-N TFs by `score` (columns `tf`, `score`).
  - `plot_ko_shift_field(shift_df: pd.DataFrame, embedding_df: pd.DataFrame, out_dir, *, tf: str, groups: pd.Series | None = None, name: str | None = None) -> list[Path]` — quiver of per-cell shift vectors (`shift_df` columns `dx,dy`, index = cell) on the 2-D embedding (`embedding_df` columns like `DIM1,DIM2`, index = cell), aligned by index intersection.
  - `plot_ko_fate_summary(fate_df: pd.DataFrame, out_dir, *, tf: str, name: str | None = None) -> list[Path]` — bar of per-cluster net transition-probability change for one KO (`fate_df` columns `cluster`, `delta`).
  - `plot_grn_connectivity(grn_summary_df: pd.DataFrame, out_dir, *, n_top: int = 20, name: str = "perturbation_grn_connectivity") -> list[Path]` — bar of top regulators by degree/connectivity (`grn_summary_df` columns `tf`, `degree`).

Model the empty-guard + `figstyle` usage on `regulon_figures.py` (each `plot_*` starts with `if df is None or df.shape[0] == 0: return []`, then `figstyle.set_style()`, builds a fig, ends with `return figstyle.save_figure(fig, out_dir, name)`). Colors come from `figstyle.CATEGORICAL_PALETTE`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for perturbation figures (synthetic input; PNG+PDF; empty -> [])."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cellquorum.perturbation import perturbation_figures as pf


def _ranking(n: int = 25) -> pd.DataFrame:
    return pd.DataFrame(
        {"tf": [f"TF{i}" for i in range(n)], "score": np.linspace(1.0, 0.1, n),
         "n_cells": 100, "direction": "directional"}
    )


def test_target_ranking_writes_png_and_pdf(tmp_path: Path) -> None:
    paths = pf.plot_target_ranking(_ranking(), tmp_path, n_top=10)
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes == [".pdf", ".png"]
    assert all(p.exists() for p in paths)


def test_target_ranking_empty_returns_empty(tmp_path: Path) -> None:
    assert pf.plot_target_ranking(pd.DataFrame(columns=["tf", "score"]), tmp_path) == []


def test_shift_field_writes_and_aligns(tmp_path: Path) -> None:
    idx = [f"c{i}" for i in range(50)]
    shift = pd.DataFrame({"dx": np.random.rand(50), "dy": np.random.rand(50)}, index=idx)
    emb = pd.DataFrame({"DIM1": np.random.rand(50), "DIM2": np.random.rand(50)}, index=idx)
    paths = pf.plot_ko_shift_field(shift, emb, tmp_path, tf="PROX1")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]


def test_shift_field_no_overlap_returns_empty(tmp_path: Path) -> None:
    shift = pd.DataFrame({"dx": [1.0], "dy": [1.0]}, index=["a"])
    emb = pd.DataFrame({"DIM1": [1.0], "DIM2": [1.0]}, index=["b"])
    assert pf.plot_ko_shift_field(shift, emb, tmp_path, tf="X") == []


def test_fate_summary_writes(tmp_path: Path) -> None:
    fate = pd.DataFrame({"cluster": ["A", "B", "C"], "delta": [0.1, -0.2, 0.05]})
    paths = pf.plot_ko_fate_summary(fate, tmp_path, tf="PROX1")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]


def test_grn_connectivity_writes(tmp_path: Path) -> None:
    grn = pd.DataFrame({"tf": [f"TF{i}" for i in range(30)], "degree": range(30)})
    paths = pf.plot_grn_connectivity(grn, tmp_path, n_top=15)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]


def test_grn_connectivity_empty_returns_empty(tmp_path: Path) -> None:
    assert pf.plot_grn_connectivity(pd.DataFrame(columns=["tf", "degree"]), tmp_path) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_perturbation_figures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cellquorum.perturbation.perturbation_figures'`

- [ ] **Step 3: Implement the figures**

Create `src/cellquorum/perturbation/perturbation_figures.py`. Start exactly:

```python
"""Publication figures for the in-silico KO (CellOracle) stage.

House-styled on cellquorum.visualization.figstyle. Every plot returns the written
PNG+PDF paths and returns [] on empty/degenerate input — never raises to the caller,
so a single failed figure never sinks the stage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cellquorum.visualization import figstyle


def plot_target_ranking(
    ranking_df: pd.DataFrame,
    out_dir: Path | str,
    *,
    n_top: int = 20,
    name: str = "perturbation_target_ranking",
) -> list[Path]:
    """Horizontal lollipop of the top-N knockout targets by shift score."""
    if ranking_df is None or ranking_df.shape[0] == 0 or "score" not in ranking_df:
        return []
    figstyle.set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = ranking_df.sort_values("score", ascending=False).head(n_top).iloc[::-1]
    color = figstyle.CATEGORICAL_PALETTE[0]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.32 * len(df))))
    y = np.arange(len(df))
    ax.hlines(y, 0, df["score"].to_numpy(), color=color, linewidth=2)
    ax.plot(df["score"].to_numpy(), y, "o", color=color, markersize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["tf"].astype(str).tolist(), fontsize=8)
    ax.set_xlabel("KO shift score")
    ax.set_title(f"Top {min(n_top, len(df))} in-silico knockout targets", fontweight="bold")
    return figstyle.save_figure(fig, out_dir, name)
```

Implement the other three (`plot_ko_shift_field`, `plot_ko_fate_summary`, `plot_grn_connectivity`) following the same pattern: empty-guard first (`plot_ko_shift_field` guards `shift_df`/`embedding_df` empty AND returns `[]` when `shift_df.index.intersection(embedding_df.index)` is empty), then `figstyle.set_style()`, build the figure with `figstyle.CATEGORICAL_PALETTE` colors, and `return figstyle.save_figure(fig, out_dir, name or f"perturbation_ko_shift_{tf}")`. `plot_ko_shift_field` uses `ax.quiver` on the aligned coords; `plot_ko_fate_summary` a bar over `fate_df["cluster"]` vs `fate_df["delta"]`; `plot_grn_connectivity` a bar of the top-N `grn_summary_df` by `degree`.

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_perturbation_figures.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/perturbation/perturbation_figures.py tests/test_perturbation_figures.py
git commit -m "feat: add house-styled perturbation figures (ranking, shift field, fate, GRN)"
```

---

### Task 5: `CellOracleMethod` orchestration

**Files:**
- Create: `src/cellquorum/perturbation/celloracle_method.py`
- Test: `tests/test_celloracle_method.py`

**Interfaces:**
- Consumes: `cellquorum.backends.celloracle_backend.CELLORACLE_KO_PY`; `cellquorum.contracts.DataContract`; `cellquorum.core.stage.{StageArtifact, StageResult}`; `cellquorum.methods.base.{AnalysisMethod, MethodSkip}`; `cellquorum.perturbation.perturbation_figures` (as `pfig`).
- Produces: `CellOracleMethod(AnalysisMethod)` — `name="celloracle"`, `stage_category="perturbation"`, `backend="celloracle"`; `input_contract(config)`, `requires_obs(config)`, `_run(adata, config, context)`.

Model `_run` on `PyscenicMethod._run` (same skeleton: resolve config → resolve generic keys → guards → write scratch h5ad → run script → parse outputs → figures → artifacts → metrics → `StageResult`). Differences from pySCENIC:
- Backend key `"celloracle"`, module probe `_py_module_available("celloracle")`, script `CELLORACLE_KO_PY`.
- **No cisTarget resource gate** (base GRN is built in).
- `condition_key`/`healthy_label`/`cluster_key`/`embedding_key`/`rep_key` resolved generically; passed as CLI args (empty string when unset).
- Output parsing keyed on `out_dir / "perturbation_ranking.csv"` + `perturbation_SKIPPED_{tag}.txt` / `perturbation_FAILED_{tag}.txt`. Empty ranking (header only) → still return `StageResult` (skip figures), NOT a `MethodSkip` (the run succeeded; there was simply nothing to rank — consistent with "condition absent is not a skip").
- Metrics: `n_tfs_screened`, `n_top_targets`, `condition_scored` (bool: `direction == "directional"` present in the table), `cluster_key`, `n_obs`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the CellOracle perturbation method (fake backend; no real celloracle)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.methods.base import MethodSkip
from cellquorum.perturbation.celloracle_method import CellOracleMethod


def _adata(n: int = 300, g: int = 40, condition: bool = False) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    obs = {"cell_type": ["A" if i % 2 else "B" for i in range(n)]}
    if condition:
        obs["condition"] = ["disease" if i % 2 else "healthy" for i in range(n)]
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame(obs, index=[f"cell_{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(g)]),
    )
    a.layers["counts"] = X.copy()
    a.obsm["X_umap"] = rng.random((n, 2)).astype("float32")
    return a


def _ctx(tmp_path: Path, backend):
    paths = SimpleNamespace(results=tmp_path / "res", scratch=tmp_path / "scr")
    reg = SimpleNamespace(get=lambda name: backend)
    return SimpleNamespace(paths=paths, backend_registry=reg, config=None)


def test_skips_when_too_few_cells(tmp_path: Path) -> None:
    res = CellOracleMethod()._run(
        _adata(n=10), {"min_cells_total": 200}, _ctx(tmp_path, object())
    )
    assert isinstance(res, MethodSkip)
    assert "too few cells" in res.reason.lower()


def test_skips_when_backend_missing(tmp_path: Path) -> None:
    cfg = {"launcher": "python"}  # resolves on PATH
    res = CellOracleMethod()._run(_adata(), cfg, _ctx(tmp_path, None))
    assert isinstance(res, MethodSkip)


def test_skips_on_timeout(tmp_path: Path) -> None:
    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _s, _a, *, timeout=None):  # noqa: ANN001, ANN003
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    res = CellOracleMethod()._run(_adata(), {"launcher": "python"}, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)


def test_input_contract_does_not_require_condition_or_cluster() -> None:
    m = CellOracleMethod()
    contract = m.input_contract({})
    assert contract.required_obs == []
    assert m.requires_obs({}) == []


def _fake_success_backend(direction: str):
    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _script, args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(args[args.index("--out-dir") + 1])
            tag = args[args.index("--tag") + 1]
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"tf": ["PROX1", "PIEZO1"], "score": [0.9, 0.4],
                 "n_cells": [300, 300], "direction": [direction, direction]}
            ).to_csv(out_dir / "perturbation_ranking.csv", index=False)
            # a shift-vector parquet so the shift-field figure has input
            idx = [f"cell_{i}" for i in range(300)]
            pd.DataFrame(
                {"dx": np.random.rand(300), "dy": np.random.rand(300)}, index=idx
            ).to_parquet(out_dir / "shift_vectors_PROX1.parquet")
            _ = tag
            return SimpleNamespace(returncode=0, stderr="")

    return FakeBackend()


def test_success_directional_builds_artifacts_and_metrics(tmp_path: Path) -> None:
    res = CellOracleMethod()._run(
        _adata(condition=True),
        {"launcher": "python", "condition_key": "condition", "healthy_label": "healthy"},
        _ctx(tmp_path, _fake_success_backend("directional")),
    )
    assert not isinstance(res, MethodSkip)
    assert res.metrics["n_tfs_screened"] == 2
    assert res.metrics["condition_scored"] is True
    assert res.metrics["cluster_key"] == "cell_type"
    assert res.metrics["n_obs"] == 300
    names = {a.name for a in res.artifacts}
    assert "ranking" in names


def test_success_magnitude_when_no_condition(tmp_path: Path) -> None:
    res = CellOracleMethod()._run(
        _adata(),
        {"launcher": "python"},
        _ctx(tmp_path, _fake_success_backend("magnitude")),
    )
    assert not isinstance(res, MethodSkip)
    assert res.metrics["condition_scored"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_celloracle_method.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cellquorum.perturbation.celloracle_method'`

- [ ] **Step 3: Implement the method**

Create `src/cellquorum/perturbation/celloracle_method.py`:

```python
"""In-silico transcription-factor knockout via CellOracle in an isolated env."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from cellquorum.backends.celloracle_backend import CELLORACLE_KO_PY
from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.perturbation import perturbation_figures as pfig


class CellOracleMethod(AnalysisMethod):
    """Directed in-silico TF knockout + fate-shift ranking via CellOracle."""

    name = "celloracle"
    stage_category = "perturbation"
    backend = "celloracle"

    def input_contract(self, config: dict) -> DataContract:
        """Require the counts layer; condition/cluster keys are NOT hard obs reqs.

        They fall back generically, so requiring them here would hard-fail the
        contract instead of allowing the fallback (the grn/hdWGCNA lesson).
        """
        layer = config.get("layer", "counts")
        return DataContract(
            required_layers=[layer],
            required_obs=[],
            expression_layer=layer,
            expected_kind="counts",
        )

    def requires_obs(self, config: dict) -> list[str]:
        return []

    def _run(self, adata, config, context):  # noqa: ANN001
        # 1. Resolve config
        layer = config.get("layer", "counts")
        organism = config.get("organism", "human")
        min_cells_total = int(config.get("min_cells_total", 200))
        n_top_targets = int(config.get("n_top_targets", 20))
        knn_n_neighbors = int(config.get("knn_n_neighbors", 200))
        n_propagation = int(config.get("n_propagation", 3))
        seed = int(config.get("seed", 0))
        launcher = config.get("launcher", "micromamba")
        timeout_seconds = int(config.get("timeout_seconds", 10800))
        condition_key = config.get("condition_key")
        healthy_label = config.get("healthy_label")
        tf_list = config.get("tf_list")

        # 2. Resolve generic keys
        cluster_key = config.get("cluster_key")
        if not cluster_key:
            cluster_key = (
                "cell_type" if "cell_type" in adata.obs.columns
                else "leiden" if "leiden" in adata.obs.columns
                else "all"
            )
        rep_key = config.get("rep_key")
        if not rep_key:
            rep_key = "X_pca" if "X_pca" in adata.obsm else (
                "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
            )
        embedding_key = config.get("embedding_key") or "X_umap"

        # 3. Guards -> MethodSkip
        if adata.n_obs < min_cells_total:
            return MethodSkip(
                reason=f"celloracle skipped: too few cells ({adata.n_obs} < {min_cells_total})",
                details={"method": self.name, "n_obs": int(adata.n_obs)},
            )
        if shutil.which(launcher) is None:
            return MethodSkip(
                reason=f"celloracle skipped: launcher '{launcher}' not found on PATH",
                details={"method": self.name, "launcher": launcher},
            )
        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("celloracle")
            except Exception:
                backend = None
        if backend is None:
            return MethodSkip(
                reason="celloracle skipped: celloracle backend unavailable",
                details={"method": self.name},
            )
        try:
            module_ok = backend._py_module_available("celloracle")
        except Exception:
            module_ok = False
        if not module_ok:
            return MethodSkip(
                reason="celloracle skipped: celloracle module unavailable in env",
                details={"method": self.name},
            )

        # 4. Write counts h5ad to scratch
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        h5ad = scratch / "perturbation_input.h5ad"
        if layer and layer != "X" and layer in adata.layers:
            a2 = adata.copy()
            a2.X = a2.layers[layer]
            a2.write_h5ad(h5ad)
        else:
            adata.write_h5ad(h5ad)

        out_dir = Path(getattr(context.paths, "results", ".")) / "perturbation"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "perturbation"

        # 5. Run the in-env KO script
        ko_args = [
            "--h5ad", str(h5ad),
            "--out-dir", str(out_dir),
            "--tag", tag,
            "--organism", str(organism),
            "--cluster-key", str(cluster_key),
            "--rep-key", str(rep_key),
            "--embedding-key", str(embedding_key),
            "--condition-key", str(condition_key or ""),
            "--healthy-label", str(healthy_label or ""),
            "--tf-list", " ".join(tf_list) if tf_list else "",
            "--n-top-targets", str(n_top_targets),
            "--knn-n-neighbors", str(knn_n_neighbors),
            "--n-propagation", str(n_propagation),
            "--seed", str(seed),
        ]
        try:
            proc = backend.run_script(CELLORACLE_KO_PY, ko_args, timeout=timeout_seconds)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            return MethodSkip(
                reason="celloracle skipped: KO script execution failed or timed out",
                details={"method": self.name, "error": str(exc)[:500]},
            )
        if proc.returncode != 0:
            return MethodSkip(
                reason="celloracle skipped: KO script failed",
                details={"method": self.name, "stderr": str(getattr(proc, "stderr", "")).strip()[:500]},
            )

        ranking_csv = out_dir / "perturbation_ranking.csv"
        failed_marker = out_dir / f"perturbation_FAILED_{tag}.txt"
        skip_marker = out_dir / f"perturbation_SKIPPED_{tag}.txt"
        if failed_marker.exists():
            return MethodSkip(
                reason=f"celloracle skipped: {failed_marker.read_text().strip()[:300]}",
                details={"method": self.name},
            )
        if not ranking_csv.exists() and skip_marker.exists():
            return MethodSkip(
                reason=f"celloracle skipped: {skip_marker.read_text().strip()[:300]}",
                details={"method": self.name},
            )
        if not ranking_csv.exists():
            return MethodSkip(
                reason="celloracle skipped: no ranking produced",
                details={"method": self.name},
            )

        try:
            ranking = pd.read_csv(ranking_csv)
        except Exception as exc:
            return MethodSkip(
                reason="celloracle skipped: could not read ranking CSV",
                details={"method": self.name, "error": str(exc)[:500]},
            )

        # 6. Figures (in cellquorum env) — never let one failure sink the stage
        notes: list[str] = []
        figs: list[Path] = []
        if len(ranking) > 0:
            try:
                figs.extend(pfig.plot_target_ranking(ranking, out_dir, n_top=n_top_targets))
            except Exception as exc:
                notes.append(f"target-ranking figure failed: {str(exc)[:150]}")
            grn_summary = out_dir / "grn_summary.csv"
            if grn_summary.exists():
                try:
                    figs.extend(
                        pfig.plot_grn_connectivity(pd.read_csv(grn_summary), out_dir, n_top=n_top_targets)
                    )
                except Exception as exc:
                    notes.append(f"grn-connectivity figure failed: {str(exc)[:150]}")
            # shift-field for the top TF, if its shift vectors + embedding are present
            if embedding_key in adata.obsm and len(ranking) > 0:
                top_tf = str(ranking.sort_values("score", ascending=False).iloc[0]["tf"])
                shift_pq = out_dir / f"shift_vectors_{top_tf}.parquet"
                if shift_pq.exists():
                    try:
                        emb = pd.DataFrame(
                            adata.obsm[embedding_key][:, :2],
                            index=adata.obs_names,
                            columns=["DIM1", "DIM2"],
                        )
                        groups = (
                            adata.obs[cluster_key].astype(str)
                            if cluster_key in adata.obs.columns
                            else None
                        )
                        figs.extend(
                            pfig.plot_ko_shift_field(
                                pd.read_parquet(shift_pq), emb, out_dir, tf=top_tf, groups=groups
                            )
                        )
                    except Exception as exc:
                        notes.append(f"shift-field figure failed: {str(exc)[:150]}")

        # 7. Artifacts
        artifacts: list[StageArtifact] = [
            StageArtifact(
                name="ranking",
                path=ranking_csv,
                kind="csv",
                description="Ranked in-silico knockout targets (disease->healthy shift)",
            )
        ]
        grn_summary = out_dir / "grn_summary.csv"
        if grn_summary.exists():
            artifacts.append(
                StageArtifact(
                    name="grn_summary", path=grn_summary, kind="csv",
                    description="CellOracle fitted GRN per-cluster top regulators",
                )
            )
        for shift_pq in sorted(out_dir.glob("shift_vectors_*.parquet")):
            artifacts.append(
                StageArtifact(
                    name=f"shift_{shift_pq.stem}", path=shift_pq, kind="parquet",
                    description="Per-cell KO shift vectors on the embedding",
                )
            )
        for fig_path in figs:
            artifacts.append(
                StageArtifact(
                    name=f"figure_{fig_path.stem}", path=fig_path, kind="figure",
                    description="CellOracle in-silico KO figure",
                )
            )

        # 8. Metrics
        condition_scored = bool(
            "direction" in ranking.columns and (ranking["direction"] == "directional").any()
        )
        metrics = {
            "n_tfs_screened": int(len(ranking)),
            "n_top_targets": int(min(n_top_targets, len(ranking))),
            "condition_scored": condition_scored,
            "cluster_key": cluster_key,
            "n_obs": int(adata.n_obs),
        }

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics=metrics,
            backend="celloracle",
        )


__all__ = ["CellOracleMethod"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_celloracle_method.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/perturbation/celloracle_method.py tests/test_celloracle_method.py
git commit -m "feat: add CellOracleMethod orchestration for the perturbation stage"
```

---

### Task 6: `PerturbationStage` + method registration

**Files:**
- Create: `src/cellquorum/perturbation/stage.py`
- Modify: `src/cellquorum/perturbation/__init__.py`
- Test: `tests/test_perturbation_stage.py`

**Interfaces:**
- Consumes: `cellquorum.methods.stage_base.MethodDispatchStage`; `cellquorum.core.stage.StageResult`; `cellquorum.methods.registry.METHOD_REGISTRY`.
- Produces: `PerturbationStage(MethodDispatchStage)` — `name="perturbation"`, `stage_category="perturbation"`, `_select_method_name(config) -> config.get("method", "celloracle")`, no-op `_validate_output`. Importing `cellquorum.perturbation` registers `CellOracleMethod` under `("perturbation", "celloracle")` as a side effect (guarded by `METHOD_REGISTRY.has`). Package exports `PerturbationConfig`, `CellOracleMethod`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the perturbation stage + method registration."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.perturbation.stage import PerturbationStage


def test_stage_identity() -> None:
    s = PerturbationStage()
    assert s.name == "perturbation"
    assert s.stage_category == "perturbation"


def test_selects_celloracle_by_default() -> None:
    s = PerturbationStage()
    assert s._select_method_name({}) == "celloracle"
    assert s._select_method_name({"method": "celloracle"}) == "celloracle"


def test_method_is_registered_on_import() -> None:
    import cellquorum.perturbation  # noqa: F401

    assert METHOD_REGISTRY.has("perturbation", "celloracle")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_perturbation_stage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cellquorum.perturbation.stage'`

- [ ] **Step 3: Create the stage + wire the package**

Create `src/cellquorum/perturbation/stage.py`:

```python
"""Perturbation stage: dispatches to the configured perturbation method."""

from __future__ import annotations

import cellquorum.perturbation  # noqa: F401  (registers the method as a side effect)
from cellquorum.core.stage import StageResult
from cellquorum.methods.stage_base import MethodDispatchStage


class PerturbationStage(MethodDispatchStage):
    """Run the configured in-silico perturbation (CellOracle) method."""

    name = "perturbation"
    stage_category = "perturbation"

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "celloracle")

    def _validate_output(self, result: StageResult) -> None:
        """No structural postcondition; the stage writes tables + figures."""


__all__ = ["PerturbationStage"]
```

Replace `src/cellquorum/perturbation/__init__.py` with (mirror `grn/__init__.py`):

```python
"""In-silico perturbation (CellOracle) stage."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.perturbation.celloracle_method import CellOracleMethod
from cellquorum.perturbation.config import PerturbationConfig

if not METHOD_REGISTRY.has("perturbation", "celloracle"):
    METHOD_REGISTRY.register(CellOracleMethod)

__all__ = ["CellOracleMethod", "PerturbationConfig"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_perturbation_stage.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/perturbation/stage.py src/cellquorum/perturbation/__init__.py tests/test_perturbation_stage.py
git commit -m "feat: add PerturbationStage and register CellOracleMethod"
```

---

### Task 7: Wire config seam (config/models.py)

**Files:**
- Modify: `src/cellquorum/config/models.py` (import at ~line 73 area; `StageSelectionConfig` flag near line 611; `CellQuorumConfig` sub-block near line 760; docstrings near lines 541 and 542)
- Test: `tests/test_perturbation_config_wiring.py`

**Interfaces:**
- Consumes: `cellquorum.perturbation.config.PerturbationConfig`.
- Produces: `CellQuorumConfig().stages.perturbation is True`; `CellQuorumConfig().perturbation` is a `PerturbationConfig`.

- [ ] **Step 1: Write the failing test**

```python
"""Verify the perturbation stage is wired into the top-level config."""

from __future__ import annotations

from cellquorum.config.models import CellQuorumConfig
from cellquorum.perturbation.config import PerturbationConfig


def test_stage_flag_present_and_default_true() -> None:
    cfg = CellQuorumConfig()
    assert cfg.stages.perturbation is True


def test_perturbation_sub_block_is_perturbation_config() -> None:
    cfg = CellQuorumConfig()
    assert isinstance(cfg.perturbation, PerturbationConfig)
    assert cfg.perturbation.method == "celloracle"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_perturbation_config_wiring.py -v`
Expected: FAIL — `AttributeError` (no `stages.perturbation` / no `perturbation` field).

- [ ] **Step 3: Edit `config/models.py`**

1. Add the import beside the GRN import (after `from cellquorum.grn.config import GrnConfig` at line 73):

```python
from cellquorum.perturbation.config import PerturbationConfig
```

2. In `StageSelectionConfig`, add the flag right after the `grn` flag (after line 611 `grn: bool = True`):

```python

    # Store whether in-silico perturbation (CellOracle) is enabled.
    perturbation: bool = True
```

3. In that class's docstring (near line 541), add a line after the `grn:` entry:

```
        perturbation: Whether in-silico perturbation (CellOracle) is enabled.
```

4. In `CellQuorumConfig`, add the sub-block right after the `grn` sub-block (after line 760):

```python

    # Store in-silico perturbation (CellOracle) settings.
    perturbation: PerturbationConfig = Field(default_factory=PerturbationConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_perturbation_config_wiring.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/config/models.py tests/test_perturbation_config_wiring.py
git commit -m "feat: wire perturbation stage into top-level config"
```

---

### Task 8: Wire executor + planner seams

**Files:**
- Modify: `src/cellquorum/core/executor.py` (import near line 80; registry entry near line 284)
- Modify: `src/cellquorum/core/planner.py` (canonical `stage_flags` list near line 230)
- Test: `tests/test_perturbation_planner_wiring.py` (new); extend `tests/test_pipeline_executor.py`

**Interfaces:**
- Consumes: `cellquorum.perturbation.stage.PerturbationStage`.
- Produces: `"perturbation"` present in `build_default_stage_registry()`; planner canonical order emits `perturbation` after `grn`, before `trajectory`.

- [ ] **Step 1: Write the failing tests**

New file `tests/test_perturbation_planner_wiring.py`:

```python
"""The perturbation stage must be planned in canonical order (after grn, before trajectory)."""

from __future__ import annotations

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.executor import build_default_stage_registry
from cellquorum.core.planner import build_pipeline_plan


def test_perturbation_stage_is_planned_in_canonical_order() -> None:
    order = build_pipeline_plan(CellQuorumConfig()).enabled_stage_names()
    assert "perturbation" in order
    assert order.index("grn") < order.index("perturbation")
    assert order.index("perturbation") < order.index("trajectory")


def test_perturbation_stage_is_registered() -> None:
    registry = build_default_stage_registry()
    assert "perturbation" in registry.registered_stage_names()
```

(If `registered_stage_names()` is not the exact accessor, mirror whatever `tests/test_pipeline_executor.py` already uses to assert `"grn"` is registered — check that file and copy its idiom.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_perturbation_planner_wiring.py -v`
Expected: FAIL — `"perturbation"` absent from plan/registry.

- [ ] **Step 3: Edit executor + planner**

In `src/cellquorum/core/executor.py`:
1. Add the import beside the GRN stage import (after line 80 `from cellquorum.grn.stage import GrnStage`):

```python
from cellquorum.perturbation.stage import PerturbationStage
```

2. Add the registry entry right after `"grn": GrnStage(),` (line 284):

```python
            "perturbation": PerturbationStage(),
```

In `src/cellquorum/core/planner.py`, insert into the `stage_flags` list right after the `("grn", ...)` tuple (line 230), before `("molecular_inference", ...)`:

```python
            # In-silico perturbation (CellOracle) is a discovery-tail stage: it
            # infers its OWN GRN from counts + a built-in base GRN and simulates
            # TF knockouts, so it slots in right after grn and before the
            # trajectory/CCC tracks.
            ("perturbation", self.config.stages.perturbation),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_perturbation_planner_wiring.py tests/test_pipeline_executor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/core/executor.py src/cellquorum/core/planner.py tests/test_perturbation_planner_wiring.py tests/test_pipeline_executor.py
git commit -m "feat: wire perturbation stage into executor registry and planner order"
```

---

### Task 9: Wire backend-registry seam

**Files:**
- Modify: `src/cellquorum/backends/registry.py` (import near line 10; registration near line 322)
- Modify: `tests/test_backend_registry.py` (the two expected-backend assertions near lines 481 and 514)
- Test: covered by the edited `tests/test_backend_registry.py`

**Interfaces:**
- Consumes: `cellquorum.backends.celloracle_backend.build_celloracle_backend`.
- Produces: `"celloracle"` in `build_default_backend_registry().names()` and in the status-table row names.

- [ ] **Step 1: Update the failing assertions**

In `tests/test_backend_registry.py`, add `"celloracle"` to the sorted `registry.names()` list (line ~481) and to the `row_names` set (line ~514). `names()` returns sorted order, so `"celloracle"` sorts first:

```python
    assert registry.names() == [
        "celloracle",
        "gpu",
        "hdwgcna_r",
        "pyscenic",
        "python",
        "python_optional",
        "r",
        "rapids",
        "rscript",
        "scclr",
        "sccoda",
    ]
```

and the set assertion:

```python
    assert row_names == {
        "celloracle",
        "gpu",
        "hdwgcna_r",
        "pyscenic",
        "python",
        "python_optional",
        "r",
        "rapids",
        "rscript",
        "scclr",
        "sccoda",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_backend_registry.py -k "build_default" -v`
Expected: FAIL — `"celloracle"` not yet registered (assertion mismatch).

- [ ] **Step 3: Register the backend**

In `src/cellquorum/backends/registry.py`:
1. Add the import beside the pySCENIC backend import (after line 10 `from cellquorum.backends.pyscenic_backend import build_pyscenic_backend`):

```python
from cellquorum.backends.celloracle_backend import build_celloracle_backend
```

2. Register it right after `registry.register(build_pyscenic_backend())` (line 322):

```python
    registry.register(build_celloracle_backend())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_backend_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/backends/registry.py tests/test_backend_registry.py
git commit -m "feat: register CellOracle backend in the default backend registry"
```

---

### Task 10: Trajectory-e2e cascade guard

**Files:**
- Modify: `tests/test_trajectory_track_e2e.py` (disabled-stages dict, near line 119 where `"grn": False` lives)
- Test: the edited `tests/test_trajectory_track_e2e.py`

**Interfaces:**
- Produces: the trajectory-only e2e run disables `perturbation` too, so its counts-layer contract does not halt the trajectory-isolating run (the same cascade guard applied for `coexpression`/`grn`).

- [ ] **Step 1: Add the disabled flag**

In `tests/test_trajectory_track_e2e.py`, in the `stages={...}` dict, add right after `"grn": False,` (line 119):

```python
            "perturbation": False,
```

- [ ] **Step 2: Run the e2e test to verify it still passes**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_trajectory_track_e2e.py -v`
Expected: PASS (the trajectory track runs end-to-end with perturbation disabled). If any optional-dep skip pre-exists here, it must be identical to the pre-branch behavior — do not "fix" unrelated skips.

- [ ] **Step 3: Commit**

```bash
git add tests/test_trajectory_track_e2e.py
git commit -m "test: disable perturbation stage in trajectory-track e2e fixture"
```

---

### Task 11: README docs

**Files:**
- Modify: `README.md` (badge line 15; backbone diagram line 54; prose lines 47 + 65-70; stage table after line 125; workflow spine line 444)
- Test: none (docs). Verify no test references the old stage count.

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update the badge and counts**

- Line 15: `stages-26%20implemented` → `stages-27%20implemented`.
- Line 47: `twenty-six registered stages` → `twenty-seven registered stages`.

- [ ] **Step 2: Update the backbone diagram (line 54)**

Change:

```
    → grn → trajectory → cell_cell_communication → ccc_network → ccc_viz
```

to:

```
    → grn → perturbation → trajectory → cell_cell_communication → ccc_network → ccc_viz
```

- [ ] **Step 3: Update the prose (lines 65-70)**

Replace the "Remaining discovery stages (in-silico perturbation and cellular potency) are planned slots not yet implemented." sentence so in-silico perturbation is described as implemented, e.g. append after the `grn` sentence:

```
The `perturbation` stage runs in-silico transcription-factor knockouts with
CellOracle in an isolated frozen environment: it infers its own simulation-ready
GRN from counts + a built-in promoter base GRN, simulates each knockout, and emits
a ranked therapeutic-target table (disease→healthy shift) plus KO shift-field,
fate-redistribution, and GRN-connectivity figures. The remaining discovery slot
(cellular potency) is planned but not yet implemented.
```

- [ ] **Step 4: Add the stage-table row (after line 125)**

```
| `perturbation` | in-silico TF-knockout with CellOracle (own GRN + KO simulation, isolated env) — ranked therapeutic-target table + KO shift-field, fate-redistribution, and GRN-connectivity figures | Implemented |
```

- [ ] **Step 5: Update the workflow spine (line 444)**

Change:

```
◐ gene-regulatory networks — ✅ co-expression modules (hdWGCNA); ✅ regulon/GRN inference (pySCENIC); ⏳ in-silico perturbation
```

to:

```
◐ gene-regulatory networks — ✅ co-expression modules (hdWGCNA); ✅ regulon/GRN inference (pySCENIC); ✅ in-silico perturbation (CellOracle; ranked-target + KO shift-field figures)
```

- [ ] **Step 6: Sanity check + commit**

Run: `micromamba run -n cellquorum-dev python -m pytest tests/test_planner.py tests/test_perturbation_planner_wiring.py -v`
Expected: PASS (no test asserts the old count in a way this breaks; if one does, it belongs to the stage-count contract and should be updated to 27).

```bash
git add README.md
git commit -m "docs: document the perturbation (in-silico KO / CellOracle) stage"
```

---

## Self-Review

**1. Spec coverage:**
- Backend (isolated env, subprocess, module probe) → Task 2. ✓
- In-env 3-step script (GRN → KO sim → scoring; directional vs magnitude; two failure regimes) → Task 3. ✓
- Config (`PerturbationConfig`, all fields, StrictBaseModel) → Task 1. ✓
- Method orchestration (generic fallbacks, guards, scratch h5ad, figures, artifacts, metrics) → Task 5. ✓
- Figures (4 house-styled, empty→[]) → Task 4. ✓
- Stage + registration → Task 6. ✓
- Four wiring seams: config/models → Task 7; executor + planner → Task 8; backend registry → Task 9. ✓
- Trajectory-e2e cascade → Task 10. ✓
- README → Task 11. ✓
- Skip matrix rows: too-few-cells / launcher / backend / module (Task 5 guards); base-GRN / import / nothing-fit (Task 3 write_skip + exit 0); real failure (Task 3 FAILED + non-zero → Task 5 MethodSkip); timeout/OSError (Task 5); condition absent → not a skip (Task 3 magnitude path + Task 5 metrics); empty ranking → StageResult not skip (Task 5); single figure raises → note + continue (Task 5 per-figure try). ✓
- Metrics `n_tfs_screened, n_top_targets, condition_scored, cluster_key, n_obs` → Task 5. ✓
- UDE extension seam → documented in spec, no code (correctly out of scope; no task). ✓

**2. Placeholder scan:** No TBD/TODO. Every code step carries real code; the only "implementer confirms against the real env" note is the `_run_celloracle` worker body (Task 3), which is correct — that code is `# pragma: no cover` and cannot be pinned without a live CellOracle env, but its contract (inputs, output filenames, exit behavior) is fully specified.

**3. Type/name consistency:** `PerturbationConfig` (T1) ← used T5, T7. `CellOracleBackend`/`build_celloracle_backend`/`CELLORACLE_KO_PY` (T2) ← used T5, T9. `celloracle_ko.py` filename + `perturbation_ranking.csv`/`perturbation_SKIPPED_{tag}.txt`/`perturbation_FAILED_{tag}.txt` + columns `tf,score,n_cells,direction` (T3) ← consumed T5. `perturbation_figures` functions (T4) ← called T5. `CellOracleMethod` (T5) ← registered T6, exported T6. `PerturbationStage` (T6) ← registered T8. Planner placement after `grn` before `trajectory` (T8) matches the spec's corrected seam-3 and the planner test. Backend `names()` sorted order places `celloracle` first (T9). All consistent.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-11-perturbation-celloracle.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session with checkpoints for review.

Which approach?
