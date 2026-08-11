# CellQuorum Docker + Snakemake Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the `cellquorum` engine a reproducible Docker image that bakes every backend environment and a hypothesis-keyed Snakemake layer that expands a manifest into per-hypothesis, publication-ready output bundles.

**Architecture:** Two layered artifacts over the existing engine. (1) A multi-stage Dockerfile builds `cellquorum:<ver>` (CPU-complete) and `-gpu`, each creating micromamba envs with the exact names the subprocess backends hardcode. (2) A Snakemake workflow reads `workflow/hypotheses.yaml`, uses a pure `gen_configs()` function to expand it into one validated `cellquorum` config per `(hypothesis, cell_type)` — methods encoded as engine stage flags — runs each via `cellquorum run` inside the image, then bundles each hypothesis and aggregates a status matrix. No new analysis methods; the engine already runs and reports a single analysis.

**Tech Stack:** Python 3.12, Pydantic v2 config models, Typer CLI, pytest, Snakemake, Docker (multi-stage), micromamba, conda-lock, GNU Make.

## Global Constraints

- **No publishing.** Build + tag locally with GHCR-ready naming; never push to any registry or index.
- **Keep-local.** Never push the branch to a remote. `docs/superpowers/` is gitignored — commit its files local-only with `git add -f`.
- **No commit trailers.** Never add `Co-Authored-By`, `Generated with Claude`, or `Generated with Claude Code` to any commit message.
- **Do NOT touch pre-existing dirty/untracked files:** `configs/le_global.yaml`, `src/cellquorum/qc/visualization.py`, `src/cellquorum/reference_mapping/diagnostics.py`, `scripts/plot_integration_benchmark.py`, `scripts/run_annotation_diagnostics.py`, `scripts/run_integration_benchmark.py`.
- **Zero new analysis methods.** Orchestration + packaging only. Track-sheet methods with no engine support are declared `blocked:` in the manifest — never faked.
- **Bake ALL backends — do not trim.** The image bakes core + R + GPU + all five isolated backend envs. Completeness over image size (~15GB acceptable; not published).
- **Exact env names (source of truth = `src/cellquorum/backends/*_backend.py` `env_name`):** `celloracle_env`, `pyscenic_env`, `hdwgcna_env`, `scclr` (no `_env` suffix), `sccoda_env`, plus the primary env for core/R/GPU. A wrong name silently breaks the subprocess backends.
- **Reproducibility.** Images build from pinned lockfiles, not floating `>=`. `snakemake -n` must expand to a stable, inspectable target set with no hidden targets.
- **Repo facts (verified):** package name `cellquorum`, version `0.1.0` in both `src/cellquorum/version.py` (`__version__: str = "0.1.0"`) and `pyproject.toml`. `cellquorum --version` exists. Console scripts: `cellquorum` and `cq` → `cellquorum.cli.app:main`. Config CLI: `cellquorum run --config/-c <path> --output-dir/-o <dir> [--json] [--quiet/-q]`; `cellquorum plan --config/-c <path> [--json]`. Tests run with `python -m pytest` (config in `pyproject.toml`: `testpaths=["tests"]`, `addopts="-q"`). Ruff `line-length=100`, `target-version=py312`, lint select `["E","F","I","B","UP","ANN"]`; test files exempt from `ANN`. Type annotations are required on non-test functions.
- **Run directory layout (verified):** each run dir has `results/ figures/ reports/ objects/ provenance/ logs/ scratch/`. Artifact manifest at `provenance/artifact_manifest.csv`. Per-stage status at `provenance/stage_execution_records.json` and `.csv`. Run metadata at `provenance/run_metadata.json`. There is no `run_summary.json` file.

---

## File Structure

**Create:**
- `envs/celloracle_env.yml`, `envs/pyscenic_env.yml`, `envs/hdwgcna_env.yml`, `envs/scclr.yml`, `envs/sccoda_env.yml` — source recipes for the isolated backend envs (Task 1).
- `envs/README.md` — how the envs relate + how to regenerate locks (Task 10).
- `src/cellquorum/workflow/__init__.py` — new subpackage.
- `src/cellquorum/workflow/scaffold.py` — the Table 0 scaffold constants + method→stage map (Task 2).
- `src/cellquorum/workflow/gen_configs.py` — pure `gen_configs()` + completeness check (Task 3).
- `src/cellquorum/workflow/gen_configs_cli.py` — thin CLI wrapper that writes configs + accounting to disk (Task 4).
- `src/cellquorum/workflow/bundle.py` — per-hypothesis bundle assembler (Task 5).
- `src/cellquorum/workflow/status.py` — status-matrix aggregator (Task 6).
- `workflow/Snakefile`, `workflow/rules/matrix.smk` — the DAG (Task 7).
- `workflow/hypotheses.yaml` — the real manifest, seeded from the track sheet (Task 7).
- `tests/workflow/__init__.py`, `tests/workflow/conftest.py`, `tests/workflow/fixtures/hypotheses_fixture.yaml` — test fixtures (Tasks 3,5,6,7).
- `tests/workflow/test_scaffold.py`, `test_gen_configs.py`, `test_gen_configs_cli.py`, `test_bundle.py`, `test_status.py`, `test_snakefile_dag.py`, `test_env_recipes.py` — tests.
- `docker/Dockerfile`, `docker/.dockerignore`, `docker/smoke/smoke.yaml` — image + smoke config (Tasks 8,9).
- `Makefile` — image / image-gpu / lock / smoke / matrix targets (Task 10).
- `docs/docker.md`, `docs/snakemake.md` — usage docs (Task 10).

**Modify:**
- `pyproject.toml` — register the `gen-configs` console script and confirm `workflow` package is discovered (Tasks 4).

---

### Task 1: Backend environment source recipes (C1a)

Create the five `envs/*.yml` files the image will build from. Recipes are transcribed verbatim from each backend's `install_hint` string (the documented working recipe), augmented with the script-level imports the hints omit but the code requires. A test guards that every backend `env_name` has a matching source yml whose `name:` field is exact.

**Files:**
- Create: `envs/celloracle_env.yml`, `envs/pyscenic_env.yml`, `envs/hdwgcna_env.yml`, `envs/scclr.yml`, `envs/sccoda_env.yml`
- Test: `tests/workflow/test_env_recipes.py`

**Interfaces:**
- Consumes: nothing (leaf task).
- Produces: five env yml files whose `name:` fields are exactly `celloracle_env`, `pyscenic_env`, `hdwgcna_env`, `scclr`, `sccoda_env`. The Dockerfile (Task 8) builds envs from these.

- [ ] **Step 1: Write the failing test**

Create `tests/workflow/__init__.py` (empty) and `tests/workflow/test_env_recipes.py`:

```python
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVS_DIR = REPO_ROOT / "envs"
BACKENDS_DIR = REPO_ROOT / "src" / "cellquorum" / "backends"

# The five isolated backend envs the image must bake, keyed to their yml file.
BACKEND_ENV_FILES = {
    "celloracle_env": "celloracle_env.yml",
    "pyscenic_env": "pyscenic_env.yml",
    "hdwgcna_env": "hdwgcna_env.yml",
    "scclr": "scclr.yml",
    "sccoda_env": "sccoda_env.yml",
}


def _declared_env_names() -> set[str]:
    """Every env_name string hardcoded in the backend modules."""
    names: set[str] = set()
    pattern = re.compile(r'env_name:\s*str\s*=\s*"([^"]+)"')
    for path in BACKENDS_DIR.glob("*_backend.py"):
        names.update(pattern.findall(path.read_text()))
    return names


def test_every_declared_backend_env_has_a_source_yml() -> None:
    declared = _declared_env_names()
    # Every env name the backends hardcode must be one we ship a recipe for.
    assert declared == set(BACKEND_ENV_FILES), (
        f"declared backend envs {declared} != recipe set {set(BACKEND_ENV_FILES)}"
    )


def test_each_recipe_file_exists_and_name_matches() -> None:
    for env_name, filename in BACKEND_ENV_FILES.items():
        path = ENVS_DIR / filename
        assert path.exists(), f"missing env recipe {path}"
        doc = yaml.safe_load(path.read_text())
        assert doc["name"] == env_name, f"{filename} name={doc['name']!r} != {env_name!r}"
        assert doc["dependencies"], f"{filename} has no dependencies"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_env_recipes.py -v`
Expected: FAIL — recipe files do not exist yet (and/or declared-name mismatch).

- [ ] **Step 3: Create the five recipe files**

`envs/celloracle_env.yml` (recipe: backend install_hint + spec C1 fuller recipe + script imports `anndata, scanpy, numpy, pandas`):

```yaml
name: celloracle_env
channels:
  - conda-forge
  - bioconda
dependencies:
  - python=3.10
  - numpy
  - cython
  - pandas
  - scipy
  - scikit-learn
  - numba
  - matplotlib
  - h5py
  - louvain
  - python-igraph
  - setuptools<81
  - anndata
  - scanpy
  - pip
  - pip:
      - velocyto
      - celloracle
```

`envs/pyscenic_env.yml` (recipe: install_hint pins `python=3.10 numpy=1.23.5 pandas=1.5.3 setuptools<81 pyscenic loompy`; scripts also import `h5py, scipy, pyarrow`):

```yaml
name: pyscenic_env
channels:
  - conda-forge
  - bioconda
dependencies:
  - python=3.10
  - numpy=1.23.5
  - pandas=1.5.3
  - setuptools<81
  - scipy
  - h5py
  - pyarrow
  - pyscenic
  - loompy
```

`envs/hdwgcna_env.yml` (recipe: install_hint `r-seurat r-hdwgcna r-wgcna bioconductor-zellkonverter`; script also uses SingleCellExperiment + SummarizedExperiment — add them explicitly):

```yaml
name: hdwgcna_env
channels:
  - conda-forge
  - bioconda
dependencies:
  - r-seurat
  - r-hdwgcna
  - r-wgcna
  - bioconductor-zellkonverter
  - bioconductor-singlecellexperiment
  - bioconductor-summarizedexperiment
```

`envs/scclr.yml` (recipe: install_hint `python=3.13 rust maturin pip` then `pip install -e /path/to/scclr`; scclr pins `anndata<0.10.9`. Source location is unknown — the toolchain is baked here; the `scclr` package itself is installed at image-build time via a Docker build ARG, see Task 8. The `pip:` entry is intentionally omitted so this env solves without the private source):

```yaml
name: scclr
channels:
  - conda-forge
dependencies:
  - python=3.13
  - rust
  - maturin
  - pip
  - numpy
  - scipy
  - anndata<0.10.9
```

`envs/sccoda_env.yml` (recipe: install_hint `python=3.10 pip` then `pip install sccoda tensorflow`; script imports `numpy, pandas` too):

```yaml
name: sccoda_env
channels:
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - pandas
  - pip
  - pip:
      - sccoda
      - tensorflow
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_env_recipes.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add tests/workflow/__init__.py tests/workflow/test_env_recipes.py envs/celloracle_env.yml envs/pyscenic_env.yml envs/hdwgcna_env.yml envs/scclr.yml envs/sccoda_env.yml
git commit -m "feat: add source env recipes for the five isolated backends"
```

---

### Task 2: Scaffold constants + method→stage map (C4/C5 core)

Define the Table 0 core scaffold, the always-on mandatory stages, and the map from each scaffold method to the engine `stages` boolean flags it enables. This module is the single source of truth for "which stage flags does a method turn on," consumed by `gen_configs`.

**Files:**
- Create: `src/cellquorum/workflow/__init__.py` (empty), `src/cellquorum/workflow/scaffold.py`
- Test: `tests/workflow/test_scaffold.py`

**Interfaces:**
- Consumes: the engine's `StageSelectionConfig` field names (verified list below).
- Produces:
  - `SCAFFOLD: list[str]` — the 7 Table 0 method keys.
  - `MANDATORY_STAGES: list[str]` — stage flags always `True` (upstream prerequisites).
  - `SCAFFOLD_METHOD_STAGES: dict[str, list[str]]` — method key → engine stage flag names.
  - `ALL_OPTIONAL_STAGES: frozenset[str]` — every stage flag that `gen_configs` may toggle off.

**Context — verified `StageSelectionConfig` boolean fields (the legal stage-flag names):** `ambient_correction, qc, preprocessing, feature_selection, dimensionality, clustering, integration, annotation, annotation_diagnostics, annotation_consensus, reference_mapping, integration_benchmark, integration_gate, population_identity, state_scoring, discovery, subclustering, adjudication, composition, differential_expression, coexpression, grn, perturbation, differential_abundance, enrichment, enrichment_viz, de_viz, ccc_viz, embeddings, trajectory, trajectory_viz, molecular_inference, cell_cell_communication, network_analysis`.

- [ ] **Step 1: Write the failing test**

Create `src/cellquorum/workflow/__init__.py` (empty) and `tests/workflow/test_scaffold.py`:

```python
from __future__ import annotations

from cellquorum.config.models import StageSelectionConfig
from cellquorum.workflow import scaffold


def test_scaffold_has_seven_table0_methods() -> None:
    assert scaffold.SCAFFOLD == [
        "pseudobulk",
        "subclustering",
        "pathway_enrichment",
        "rna_velocity",
        "phate_pseudotime",
        "cell_cell_communication",
        "progeny",
    ]


def test_every_mapped_stage_is_a_real_stage_flag() -> None:
    legal = set(StageSelectionConfig.model_fields)
    for method, stages in scaffold.SCAFFOLD_METHOD_STAGES.items():
        assert stages, f"{method} maps to no stages"
        for stage in stages:
            assert stage in legal, f"{method} -> unknown stage flag {stage!r}"
    for stage in scaffold.MANDATORY_STAGES:
        assert stage in legal, f"mandatory stage {stage!r} is not a real flag"


def test_every_scaffold_method_is_mapped() -> None:
    assert set(scaffold.SCAFFOLD_METHOD_STAGES) == set(scaffold.SCAFFOLD)


def test_optional_stages_exclude_mandatory() -> None:
    assert scaffold.MANDATORY_STAGES
    assert scaffold.ALL_OPTIONAL_STAGES.isdisjoint(scaffold.MANDATORY_STAGES)
    # Optional set is exactly the legal flags minus the mandatory ones.
    legal = set(StageSelectionConfig.model_fields)
    assert scaffold.ALL_OPTIONAL_STAGES == frozenset(legal) - set(scaffold.MANDATORY_STAGES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_scaffold.py -v`
Expected: FAIL with `ModuleNotFoundError: cellquorum.workflow.scaffold`.

- [ ] **Step 3: Write the module**

Create `src/cellquorum/workflow/scaffold.py`:

```python
"""Table 0 core scaffold: the seven methods every hypothesis runs by default,
and the mapping from each method to the engine stage flags it enables.

This is the single source of truth for translating a hypothesis manifest's
method selection into ``CellQuorumConfig.stages`` booleans. Stage-flag names
here MUST match fields of ``cellquorum.config.models.StageSelectionConfig``;
``tests/workflow/test_scaffold.py`` enforces that.
"""

from __future__ import annotations

from cellquorum.config.models import StageSelectionConfig

# The seven Table 0 methods, in track-sheet order.
SCAFFOLD: list[str] = [
    "pseudobulk",
    "subclustering",
    "pathway_enrichment",
    "rna_velocity",
    "phate_pseudotime",
    "cell_cell_communication",
    "progeny",
]

# Upstream stages every run needs regardless of method selection: load ->
# QC -> preprocessing -> dimensionality -> clustering -> integration ->
# annotation and its consensus/diagnostics. These are prerequisites for
# every downstream method, so they are always enabled.
MANDATORY_STAGES: list[str] = [
    "qc",
    "preprocessing",
    "feature_selection",
    "dimensionality",
    "clustering",
    "integration",
    "annotation",
    "annotation_consensus",
    "annotation_diagnostics",
    "population_identity",
]

# Each scaffold method -> the optional stage flags it turns on. Values are
# verified members of StageSelectionConfig. Rationale per method:
#   pseudobulk            -> pseudobulk differential expression + its figures
#   subclustering         -> cell-state subclustering + adjudication
#   pathway_enrichment    -> enrichment + enrichment figures
#   rna_velocity          -> trajectory (scVelo/CellRank) + trajectory figures
#   phate_pseudotime      -> embeddings (PHATE) driving pseudotime ordering
#   cell_cell_communication -> CCC + CCC figures + network analysis
#   progeny               -> pathway-activity inference (molecular_inference)
SCAFFOLD_METHOD_STAGES: dict[str, list[str]] = {
    "pseudobulk": ["differential_expression", "de_viz"],
    "subclustering": ["subclustering", "adjudication"],
    "pathway_enrichment": ["enrichment", "enrichment_viz"],
    "rna_velocity": ["trajectory", "trajectory_viz"],
    "phate_pseudotime": ["embeddings"],
    "cell_cell_communication": [
        "cell_cell_communication",
        "ccc_viz",
        "network_analysis",
    ],
    "progeny": ["molecular_inference"],
}

# Every stage flag gen_configs may toggle OFF (all legal flags minus mandatory).
ALL_OPTIONAL_STAGES: frozenset[str] = frozenset(StageSelectionConfig.model_fields) - set(
    MANDATORY_STAGES
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_scaffold.py -v`
Expected: PASS (all four tests). If `test_every_mapped_stage_is_a_real_stage_flag` fails, a stage name is wrong — fix it against the verified field list above; do not add a flag that isn't in `StageSelectionConfig`.

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/workflow/__init__.py src/cellquorum/workflow/scaffold.py tests/workflow/test_scaffold.py
git commit -m "feat: add Table 0 scaffold constants and method->stage map"
```

---

### Task 3: Pure config generator with completeness check (C5)

The heart of the "can't forget a step" guarantee. A pure function expands a hypothesis manifest into `{config_key: config_dict}`, one entry per `(hypothesis, cell_type)`, with method selection resolved to stage flags. It raises loudly on any incomplete or inconsistent manifest, and every emitted dict validates against the engine config schema.

**Files:**
- Create: `src/cellquorum/workflow/gen_configs.py`, `tests/workflow/fixtures/hypotheses_fixture.yaml`, `tests/workflow/conftest.py`
- Test: `tests/workflow/test_gen_configs.py`

**Interfaces:**
- Consumes: `scaffold.SCAFFOLD`, `scaffold.SCAFFOLD_METHOD_STAGES`, `scaffold.MANDATORY_STAGES`, `scaffold.ALL_OPTIONAL_STAGES` (Task 2); `cellquorum.config.loader.validate_config_dict(mapping) -> CellQuorumConfig` (raises `ConfigLoadError`).
- Produces:
  - `class ManifestError(ValueError)` — raised on any manifest inconsistency.
  - `def resolve_methods(entry: dict, scaffold: list[str]) -> dict[str, list[str]]` — returns `{"run": [...], "skip": [...], "blocked": [...]}`; raises `ManifestError` on incompleteness/unknown/double-listing.
  - `def gen_configs(manifest: dict, template: dict, *, scaffold=SCAFFOLD, method_stages=SCAFFOLD_METHOD_STAGES, mandatory_stages=MANDATORY_STAGES) -> dict[str, dict]` — keys are `f"{hypothesis_id}__{cell_type}"`, values are full config dicts. Used by Task 4 (CLI) and Task 7 (Snakefile).
  - `def accounting(manifest: dict, *, scaffold=SCAFFOLD) -> dict[str, dict[str, list[str]]]` — `{hypothesis_id: {"run","skip","blocked"}}`; used by Task 4 and Task 6.

- [ ] **Step 1: Write the fixture and the failing test**

Create `tests/workflow/fixtures/hypotheses_fixture.yaml`:

```yaml
il33_axis:
  title: "IL33/ST2 alarmin KC->ILC2 axis"
  cell_types: [KC, ILC]
  inputs:
    KC: /data/kc.h5ad
    ILC: /data/ilc.h5ad
  skip:
    pseudobulk: "single-condition subset; pseudobulk N/A"
  blocked:
    rna_velocity: "no spliced/unspliced layers for this subset"
  gene_programs:
    alarmin: [Il33, Il1rl1, Il13]
  config_overrides:
    run:
      random_seed: 7
emt_krt:
  title: "EMT / KRT-high keratinocyte program"
  cell_types: [KC]
  inputs:
    KC: /data/kc.h5ad
  # no skip / blocked -> full scaffold runs
```

Create `tests/workflow/conftest.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def manifest() -> dict:
    return yaml.safe_load((FIXTURES / "hypotheses_fixture.yaml").read_text())


@pytest.fixture
def template() -> dict:
    """Minimal valid-ish base config the generator fills in per run."""
    return {
        "project": {"name": "placeholder"},
        "input": {"h5ad": "/placeholder.h5ad"},
        "compute": {"backend": "cpu"},
    }
```

Create `tests/workflow/test_gen_configs.py`:

```python
from __future__ import annotations

import copy

import pytest

from cellquorum.config.loader import validate_config_dict
from cellquorum.workflow import scaffold
from cellquorum.workflow.gen_configs import (
    ManifestError,
    accounting,
    gen_configs,
    resolve_methods,
)


def test_resolve_full_scaffold_when_no_skip_or_blocked(manifest) -> None:
    resolved = resolve_methods(manifest["emt_krt"], scaffold.SCAFFOLD)
    assert set(resolved["run"]) == set(scaffold.SCAFFOLD)
    assert resolved["skip"] == []
    assert resolved["blocked"] == []


def test_resolve_subtracts_skip_and_blocked(manifest) -> None:
    resolved = resolve_methods(manifest["il33_axis"], scaffold.SCAFFOLD)
    assert "pseudobulk" not in resolved["run"]
    assert "rna_velocity" not in resolved["run"]
    assert resolved["skip"] == ["pseudobulk"]
    assert resolved["blocked"] == ["rna_velocity"]
    # run + skip + blocked exactly reconstitutes the scaffold.
    assert set(resolved["run"]) | set(resolved["skip"]) | set(resolved["blocked"]) == set(
        scaffold.SCAFFOLD
    )


def test_gen_configs_emits_one_config_per_cell_type(manifest, template) -> None:
    out = gen_configs(manifest, template)
    assert set(out) == {"il33_axis__KC", "il33_axis__ILC", "emt_krt__KC"}


def test_generated_configs_validate(manifest, template) -> None:
    out = gen_configs(manifest, template)
    for key, cfg in out.items():
        validate_config_dict(cfg)  # raises if invalid


def test_stage_flags_reflect_resolved_methods(manifest, template) -> None:
    out = gen_configs(manifest, template)
    kc = out["il33_axis__KC"]
    stages = kc["stages"]
    # skipped pseudobulk -> its stages off
    assert stages["differential_expression"] is False
    # blocked rna_velocity -> its stages off
    assert stages["trajectory"] is False
    # a run method (pathway_enrichment) -> its stages on
    assert stages["enrichment"] is True
    # mandatory stage always on
    assert stages["qc"] is True
    # an unrelated optional stage off
    assert stages["grn"] is False


def test_gene_programs_and_overrides_merged(manifest, template) -> None:
    out = gen_configs(manifest, template)
    kc = out["il33_axis__KC"]
    assert kc["run"]["random_seed"] == 7  # from config_overrides
    assert kc["input"]["h5ad"] == "/data/kc.h5ad"  # per-cell-type input
    assert kc["project"]["name"] == "il33_axis__KC"


def test_unknown_method_raises(manifest) -> None:
    bad = copy.deepcopy(manifest)
    bad["emt_krt"]["skip"] = {"not_a_method": "typo"}
    with pytest.raises(ManifestError, match="unknown method"):
        resolve_methods(bad["emt_krt"], scaffold.SCAFFOLD)


def test_method_in_two_categories_raises(manifest) -> None:
    bad = copy.deepcopy(manifest)
    bad["il33_axis"]["skip"] = {"rna_velocity": "also blocked"}  # already in blocked
    with pytest.raises(ManifestError, match="two categories|both"):
        resolve_methods(bad["il33_axis"], scaffold.SCAFFOLD)


def test_accounting_shape(manifest) -> None:
    acct = accounting(manifest)
    assert acct["il33_axis"]["skip"] == ["pseudobulk"]
    assert acct["il33_axis"]["blocked"] == ["rna_velocity"]
    assert set(acct["emt_krt"]["run"]) == set(scaffold.SCAFFOLD)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_gen_configs.py -v`
Expected: FAIL with `ModuleNotFoundError: cellquorum.workflow.gen_configs`.

- [ ] **Step 3: Write the module**

Create `src/cellquorum/workflow/gen_configs.py`:

```python
"""Pure expansion of a hypothesis manifest into per-(hypothesis, cell_type)
cellquorum config dicts, with a completeness check that makes a forgotten
scaffold method a hard error rather than a silent omission.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from cellquorum.workflow.scaffold import (
    ALL_OPTIONAL_STAGES,
    MANDATORY_STAGES,
    SCAFFOLD,
    SCAFFOLD_METHOD_STAGES,
)


class ManifestError(ValueError):
    """Raised when a hypothesis manifest is incomplete or inconsistent."""


def _method_keys(section: Any) -> list[str]:
    """A skip/blocked section may be a list or a {method: reason} mapping."""
    if section is None:
        return []
    if isinstance(section, Mapping):
        return list(section.keys())
    if isinstance(section, list):
        return list(section)
    raise ManifestError(f"skip/blocked must be a list or mapping, got {type(section).__name__}")


def resolve_methods(entry: Mapping[str, Any], scaffold: list[str]) -> dict[str, list[str]]:
    skip = _method_keys(entry.get("skip"))
    blocked = _method_keys(entry.get("blocked"))

    scaffold_set = set(scaffold)
    for name in [*skip, *blocked]:
        if name not in scaffold_set:
            raise ManifestError(f"unknown method {name!r}; not in scaffold {scaffold}")

    overlap = set(skip) & set(blocked)
    if overlap:
        raise ManifestError(f"method(s) {sorted(overlap)} listed in two categories (skip and blocked)")

    run = [m for m in scaffold if m not in set(skip) and m not in set(blocked)]

    # Completeness: run + skip + blocked must reconstitute the whole scaffold.
    accounted = set(run) | set(skip) | set(blocked)
    if accounted != scaffold_set:
        missing = scaffold_set - accounted
        raise ManifestError(f"scaffold methods unaccounted for: {sorted(missing)}")

    return {"run": run, "skip": skip, "blocked": blocked}


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _stage_flags(run_methods: list[str], method_stages: dict[str, list[str]],
                 mandatory: list[str]) -> dict[str, bool]:
    enabled: set[str] = set(mandatory)
    for method in run_methods:
        enabled.update(method_stages[method])
    flags = {stage: False for stage in ALL_OPTIONAL_STAGES}
    flags.update({stage: True for stage in mandatory})
    for stage in enabled:
        flags[stage] = True
    return flags


def gen_configs(
    manifest: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    scaffold: list[str] = SCAFFOLD,
    method_stages: dict[str, list[str]] = SCAFFOLD_METHOD_STAGES,
    mandatory_stages: list[str] = MANDATORY_STAGES,
) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for hyp_id, entry in manifest.items():
        resolved = resolve_methods(entry, scaffold)
        stages = _stage_flags(resolved["run"], method_stages, mandatory_stages)
        cell_types = entry["cell_types"]
        inputs = entry["inputs"]
        overrides = entry.get("config_overrides", {})
        programs = entry.get("gene_programs", {})
        for cell_type in cell_types:
            key = f"{hyp_id}__{cell_type}"
            cfg = _deep_merge(dict(template), overrides)
            cfg = _deep_merge(cfg, {
                "project": {"name": key},
                "input": {"h5ad": inputs[cell_type]},
                "stages": stages,
            })
            if programs:
                cfg = _deep_merge(cfg, {"state_scoring": {"gene_programs": programs}})
            configs[key] = cfg
    return configs


def accounting(
    manifest: Mapping[str, Any], *, scaffold: list[str] = SCAFFOLD
) -> dict[str, dict[str, list[str]]]:
    return {hyp_id: resolve_methods(entry, scaffold) for hyp_id, entry in manifest.items()}
```

- [ ] **Step 4: Run tests and iterate**

Run: `python -m pytest tests/workflow/test_gen_configs.py -v`
Expected: PASS. If `test_generated_configs_validate` fails, the template is missing a schema-required field or `state_scoring.gene_programs` is not the real field path — inspect the `ValidationError`, adjust the template fixture and/or the merge target to the correct schema field, and re-run. Do not weaken the validation assertion. If `state_scoring` has no `gene_programs` field, place programs under the correct config location revealed by the schema error and update the assertion in `test_stage_flags_reflect_resolved_methods` accordingly.

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/workflow/gen_configs.py tests/workflow/test_gen_configs.py tests/workflow/conftest.py tests/workflow/fixtures/hypotheses_fixture.yaml
git commit -m "feat: pure gen_configs with scaffold completeness check"
```

---

### Task 4: gen_configs CLI wrapper + console script (C5)

A thin CLI writes the generated configs and the accounting JSON to disk. Registered as a console script so the Snakefile can invoke it.

**Files:**
- Create: `src/cellquorum/workflow/gen_configs_cli.py`
- Modify: `pyproject.toml` (add `gen-configs` script)
- Test: `tests/workflow/test_gen_configs_cli.py`

**Interfaces:**
- Consumes: `gen_configs`, `accounting` (Task 3).
- Produces: `def main(manifest_path: Path, template_path: Path, out_dir: Path) -> None` writing `out_dir/configs/<key>.yaml` and `out_dir/accounting.json`. Console entry `gen-configs = "cellquorum.workflow.gen_configs_cli:app"`.

- [ ] **Step 1: Write the failing test**

Create `tests/workflow/test_gen_configs_cli.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import yaml

from cellquorum.workflow.gen_configs_cli import main


def test_cli_writes_configs_and_accounting(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({
        "project": {"name": "placeholder"},
        "input": {"h5ad": "/placeholder.h5ad"},
        "compute": {"backend": "cpu"},
    }))
    out = tmp_path / "out"
    main(fixtures / "hypotheses_fixture.yaml", template, out)

    cfg_dir = out / "configs"
    assert (cfg_dir / "il33_axis__KC.yaml").exists()
    assert (cfg_dir / "il33_axis__ILC.yaml").exists()
    assert (cfg_dir / "emt_krt__KC.yaml").exists()

    acct = json.loads((out / "accounting.json").read_text())
    assert acct["il33_axis"]["blocked"] == ["rna_velocity"]

    written = yaml.safe_load((cfg_dir / "il33_axis__KC.yaml").read_text())
    assert written["stages"]["qc"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_gen_configs_cli.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the CLI**

Create `src/cellquorum/workflow/gen_configs_cli.py`:

```python
"""CLI wrapper: expand a hypothesis manifest to config files on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from cellquorum.workflow.gen_configs import accounting, gen_configs

app = typer.Typer(name="gen-configs", add_completion=False)


def main(manifest_path: Path, template_path: Path, out_dir: Path) -> None:
    manifest = yaml.safe_load(Path(manifest_path).read_text())
    template = yaml.safe_load(Path(template_path).read_text())
    configs = gen_configs(manifest, template)
    acct = accounting(manifest)

    cfg_dir = Path(out_dir) / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for key, cfg in configs.items():
        (cfg_dir / f"{key}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (Path(out_dir) / "accounting.json").write_text(json.dumps(acct, indent=2))


@app.command()
def run(
    manifest: Annotated[Path, typer.Option("--manifest", "-m")],
    template: Annotated[Path, typer.Option("--template", "-t")],
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o")],
) -> None:
    main(manifest, template, out_dir)


if __name__ == "__main__":  # pragma: no cover
    app()
```

- [ ] **Step 4: Register the console script**

In `pyproject.toml`, under `[project.scripts]`, add the `gen-configs` line (keep existing lines):

```toml
[project.scripts]
cellquorum = "cellquorum.cli.app:main"
cq = "cellquorum.cli.app:main"
gen-configs = "cellquorum.workflow.gen_configs_cli:app"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_gen_configs_cli.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cellquorum/workflow/gen_configs_cli.py tests/workflow/test_gen_configs_cli.py pyproject.toml
git commit -m "feat: gen-configs CLI writes per-run configs and accounting"
```

---

### Task 5: Per-hypothesis bundle assembler (C6)

Collect a hypothesis's per-cell-type run outputs (figures, tables, reports) into `bundles/<hypothesis>/` with an index report titled from the manifest — the publication-ready deliverable.

**Files:**
- Create: `src/cellquorum/workflow/bundle.py`
- Test: `tests/workflow/test_bundle.py`

**Interfaces:**
- Consumes: run dirs following the verified layout (`figures/`, `results/`, `reports/`, `provenance/`).
- Produces: `def assemble_bundle(hypothesis_id: str, title: str, run_dirs: dict[str, Path], bundle_dir: Path) -> Path` — copies each cell type's `figures/` and `results/` under `bundle_dir/<cell_type>/`, writes `bundle_dir/report.html` (title + per-cell-type artifact index), returns the report path.

- [ ] **Step 1: Write the failing test**

Create `tests/workflow/test_bundle.py`:

```python
from __future__ import annotations

from pathlib import Path

from cellquorum.workflow.bundle import assemble_bundle


def _fake_run(dir_: Path) -> Path:
    for sub in ("figures", "results", "reports", "provenance"):
        (dir_ / sub).mkdir(parents=True)
    (dir_ / "figures" / "umap.png").write_bytes(b"PNG")
    (dir_ / "results" / "de_table.csv").write_text("gene,lfc\nIl33,1.2\n")
    return dir_


def test_assemble_bundle_collects_and_reports(tmp_path: Path) -> None:
    kc = _fake_run(tmp_path / "runs" / "il33_axis" / "KC")
    ilc = _fake_run(tmp_path / "runs" / "il33_axis" / "ILC")
    bundle_dir = tmp_path / "bundles" / "il33_axis"

    report = assemble_bundle(
        "il33_axis", "IL33/ST2 alarmin KC->ILC2 axis",
        {"KC": kc, "ILC": ilc}, bundle_dir,
    )

    assert report == bundle_dir / "report.html"
    assert report.exists()
    assert (bundle_dir / "KC" / "figures" / "umap.png").exists()
    assert (bundle_dir / "ILC" / "results" / "de_table.csv").exists()
    html = report.read_text()
    assert "IL33/ST2 alarmin KC-&gt;ILC2 axis" in html or "IL33/ST2 alarmin KC->ILC2 axis" in html
    assert "KC" in html and "ILC" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_bundle.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the module**

Create `src/cellquorum/workflow/bundle.py`:

```python
"""Assemble one hypothesis's per-cell-type run outputs into a publication bundle."""

from __future__ import annotations

import html
import shutil
from pathlib import Path

_COPY_SUBDIRS = ("figures", "results")


def assemble_bundle(
    hypothesis_id: str,
    title: str,
    run_dirs: dict[str, Path],
    bundle_dir: Path,
) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    for cell_type, run_dir in sorted(run_dirs.items()):
        dest = bundle_dir / cell_type
        items: list[str] = []
        for sub in _COPY_SUBDIRS:
            src = Path(run_dir) / sub
            if src.is_dir():
                shutil.copytree(src, dest / sub, dirs_exist_ok=True)
                for artifact in sorted(src.rglob("*")):
                    if artifact.is_file():
                        rel = artifact.relative_to(run_dir)
                        items.append(f"<li>{html.escape(str(rel))}</li>")
        listing = "\n".join(items) or "<li><em>no artifacts</em></li>"
        sections.append(f"<h2>{html.escape(cell_type)}</h2>\n<ul>\n{listing}\n</ul>")

    body = "\n".join(sections)
    doc = (
        "<!doctype html>\n<html>\n<head>\n<meta charset='utf-8'>\n"
        f"<title>{html.escape(title)}</title>\n</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n"
        f"<p>Hypothesis: <code>{html.escape(hypothesis_id)}</code></p>\n"
        f"{body}\n</body>\n</html>\n"
    )
    report = bundle_dir / "report.html"
    report.write_text(doc)
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_bundle.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/workflow/bundle.py tests/workflow/test_bundle.py
git commit -m "feat: per-hypothesis bundle assembler"
```

---

### Task 6: Status-matrix aggregator (C6)

Join each run's `stage_execution_records.json` with the C5 accounting to produce a per-hypothesis×method matrix: run / succeeded / skipped / failed / blocked.

**Files:**
- Create: `src/cellquorum/workflow/status.py`
- Test: `tests/workflow/test_status.py`

**Interfaces:**
- Consumes: `stage_execution_records.json` per run; `accounting.json` from Task 4; `scaffold.SCAFFOLD_METHOD_STAGES` (to map stage status back to method status).
- Produces:
  - `def method_status(stage_records: dict, run_methods: list[str], method_stages=SCAFFOLD_METHOD_STAGES) -> dict[str, str]` — method → one of `succeeded`/`failed`/`skipped`.
  - `def build_matrix(accounting: dict, run_records: dict[str, dict], method_stages=SCAFFOLD_METHOD_STAGES) -> list[dict]` — rows `{hypothesis, cell_type, method, status}` where status ∈ {succeeded, failed, skipped, blocked}. `run_records` keyed by `"<hyp>__<cell_type>"`.
  - `def matrix_to_csv(rows: list[dict]) -> str` and `def matrix_to_markdown(rows: list[dict]) -> str`.

**Context — `stage_execution_records.json` shape:** a list of records, each with at least `stage` (name) and a status field. Verify the exact key names by reading one real file or `src/cellquorum/core/pipeline.py`'s `stage_execution_records` writer during Step 3; the test below uses a `status` string field with values `succeeded`/`skipped`/`failed` — adjust the reader to the real field names if they differ, keeping the test's semantics.

- [ ] **Step 1: Write the failing test**

Create `tests/workflow/test_status.py`:

```python
from __future__ import annotations

from cellquorum.workflow.status import build_matrix, matrix_to_csv, method_status


STAGE_RECORDS = {
    "records": [
        {"stage": "qc", "status": "succeeded"},
        {"stage": "enrichment", "status": "succeeded"},
        {"stage": "enrichment_viz", "status": "succeeded"},
        {"stage": "subclustering", "status": "failed"},
        {"stage": "adjudication", "status": "skipped"},
    ]
}


def test_method_status_rolls_up_stage_status() -> None:
    status = method_status(
        STAGE_RECORDS, run_methods=["pathway_enrichment", "subclustering"]
    )
    assert status["pathway_enrichment"] == "succeeded"
    # any failed stage in the method -> failed
    assert status["subclustering"] == "failed"


def test_build_matrix_includes_skip_and_blocked() -> None:
    acct = {
        "il33_axis": {
            "run": ["pathway_enrichment", "subclustering"],
            "skip": ["pseudobulk"],
            "blocked": ["rna_velocity"],
        }
    }
    rows = build_matrix(acct, {"il33_axis__KC": STAGE_RECORDS})
    by_method = {(r["method"], r["status"]) for r in rows}
    assert ("pseudobulk", "skipped") in by_method
    assert ("rna_velocity", "blocked") in by_method
    assert ("pathway_enrichment", "succeeded") in by_method
    assert ("subclustering", "failed") in by_method


def test_csv_has_header() -> None:
    rows = [{"hypothesis": "h", "cell_type": "KC", "method": "m", "status": "succeeded"}]
    csv = matrix_to_csv(rows)
    assert csv.splitlines()[0] == "hypothesis,cell_type,method,status"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_status.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the module**

Create `src/cellquorum/workflow/status.py`:

```python
"""Aggregate per-run stage status + manifest accounting into a status matrix."""

from __future__ import annotations

import csv
import io

from cellquorum.workflow.scaffold import SCAFFOLD_METHOD_STAGES

_FAIL = "failed"
_OK = "succeeded"
_SKIP = "skipped"


def _records(stage_records: dict) -> list[dict]:
    # Tolerate either {"records": [...]} or a bare list.
    if isinstance(stage_records, dict):
        return stage_records.get("records", [])
    return list(stage_records)


def method_status(
    stage_records: dict,
    run_methods: list[str],
    method_stages: dict[str, list[str]] = SCAFFOLD_METHOD_STAGES,
) -> dict[str, str]:
    by_stage = {rec["stage"]: rec.get("status", _SKIP) for rec in _records(stage_records)}
    result: dict[str, str] = {}
    for method in run_methods:
        stages = method_stages[method]
        statuses = [by_stage.get(s) for s in stages]
        present = [s for s in statuses if s is not None]
        if not present:
            result[method] = _SKIP
        elif _FAIL in present:
            result[method] = _FAIL
        elif all(s == _OK for s in present):
            result[method] = _OK
        else:
            result[method] = _SKIP
    return result


def build_matrix(
    accounting: dict,
    run_records: dict[str, dict],
    method_stages: dict[str, list[str]] = SCAFFOLD_METHOD_STAGES,
) -> list[dict]:
    rows: list[dict] = []
    for hyp_id, acct in accounting.items():
        cell_runs = {
            key.split("__", 1)[1]: recs
            for key, recs in run_records.items()
            if key.startswith(f"{hyp_id}__")
        }
        for cell_type, recs in sorted(cell_runs.items()):
            statuses = method_status(recs, acct["run"], method_stages)
            for method in acct["run"]:
                rows.append({"hypothesis": hyp_id, "cell_type": cell_type,
                             "method": method, "status": statuses[method]})
            for method in acct.get("skip", []):
                rows.append({"hypothesis": hyp_id, "cell_type": cell_type,
                             "method": method, "status": _SKIP})
            for method in acct.get("blocked", []):
                rows.append({"hypothesis": hyp_id, "cell_type": cell_type,
                             "method": method, "status": "blocked"})
    return rows


_FIELDS = ["hypothesis", "cell_type", "method", "status"]


def matrix_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def matrix_to_markdown(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(_FIELDS) + " |", "| " + " | ".join(["---"] * len(_FIELDS)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[f]) for f in _FIELDS) + " |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_status.py -v`
Expected: PASS. During this step, read the real `stage_execution_records.json` writer in `src/cellquorum/core/pipeline.py`; if the status field is not named `status` or values differ (e.g. `succeeded_stage_names`), adjust `_records`/`method_status` to the real names and update `STAGE_RECORDS` in the test to match the real shape — keep the roll-up semantics (any failed → failed).

- [ ] **Step 5: Commit**

```bash
git add src/cellquorum/workflow/status.py tests/workflow/test_status.py
git commit -m "feat: status-matrix aggregator over stage records + accounting"
```

---

### Task 7: Snakemake workflow + real manifest (C6)

The DAG: one `run_analysis` per `(hypothesis, cell_type)`, `bundle_hypothesis` per hypothesis, `aggregate_status` at the top. A dry-run test asserts the target set exactly. Also seed the real `workflow/hypotheses.yaml` from the track sheet.

**Files:**
- Create: `workflow/Snakefile`, `workflow/rules/matrix.smk`, `workflow/hypotheses.yaml`, `workflow/template.yaml`
- Test: `tests/workflow/test_snakefile_dag.py`

**Interfaces:**
- Consumes: `gen-configs` CLI (Task 4), `cellquorum run` CLI, `assemble_bundle` (Task 5), `build_matrix`/`matrix_to_csv`/`matrix_to_markdown` (Task 6).
- Produces: `snakemake -n` expands to exactly `{runs/<hyp>/<ct>/provenance/artifact_manifest.csv}` ∪ `{bundles/<hyp>/report.html}` ∪ `{runs/matrix_status.csv, runs/matrix_status.md}`.

- [ ] **Step 1: Write the failing dry-run test**

Create `tests/workflow/test_snakefile_dag.py`:

```python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="snakemake not installed")
def test_dry_run_expands_expected_targets(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "hypotheses_fixture.yaml"
    result = subprocess.run(
        ["snakemake", "-n", "--snakefile", str(REPO_ROOT / "workflow" / "Snakefile"),
         "--config", f"manifest={fixture}", f"workdir={tmp_path}", "--cores", "1"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for target in ("il33_axis", "emt_krt", "matrix_status"):
        assert target in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_snakefile_dag.py -v`
Expected: FAIL (Snakefile missing) or SKIP if snakemake absent. If it skips, install snakemake in the dev env (`micromamba run -n cellquorum-dev pip install snakemake`) so the test runs, then re-run — it must FAIL, not skip.

- [ ] **Step 3: Write the Snakefile, rules, template, and real manifest**

Create `workflow/template.yaml` (base config the generator fills per run; `compute.backend: auto` lets the engine pick CPU/GPU):

```yaml
project:
  name: placeholder
input:
  h5ad: /placeholder.h5ad
  counts_layer: counts
compute:
  backend: auto
  fallback_to_cpu: true
run:
  random_seed: 1337
report:
  enabled: true
  html: true
  markdown: true
```

Create `workflow/Snakefile`:

```python
from pathlib import Path

import yaml

# `config` is a Snakemake global populated from --config key=value pairs.
MANIFEST = Path(config.get("manifest", "workflow/hypotheses.yaml"))
WORKDIR = Path(config.get("workdir", "."))
TEMPLATE = Path(config.get("template", "workflow/template.yaml"))

manifest = yaml.safe_load(MANIFEST.read_text())

RUNS = WORKDIR / "runs"
BUNDLES = WORKDIR / "bundles"
GEN = WORKDIR / "generated"

# (hypothesis, cell_type) pairs
PAIRS = [(h, ct) for h, entry in manifest.items() for ct in entry["cell_types"]]

include: "rules/matrix.smk"

rule all:
    input:
        [str(BUNDLES / h / "report.html") for h in manifest],
        str(RUNS / "matrix_status.csv"),
        str(RUNS / "matrix_status.md"),
```

Create `workflow/rules/matrix.smk`:

```python
rule gen_configs:
    input:
        manifest=str(MANIFEST),
        template=str(TEMPLATE),
    output:
        accounting=str(GEN / "accounting.json"),
    params:
        out=lambda w: str(GEN),
    shell:
        "gen-configs run --manifest {input.manifest} --template {input.template} --out-dir {params.out}"

rule run_analysis:
    input:
        config=str(GEN / "configs" / "{hyp}__{ct}.yaml"),
        accounting=str(GEN / "accounting.json"),
    output:
        manifest=str(RUNS / "{hyp}" / "{ct}" / "provenance" / "artifact_manifest.csv"),
    params:
        out=lambda w: str(RUNS / w.hyp / w.ct),
    shell:
        "cellquorum run --config {input.config} --output-dir {params.out}"

def _hyp_run_targets(wildcards):
    cts = manifest[wildcards.hyp]["cell_types"]
    return [str(RUNS / wildcards.hyp / ct / "provenance" / "artifact_manifest.csv") for ct in cts]

rule bundle_hypothesis:
    input:
        _hyp_run_targets,
    output:
        report=str(BUNDLES / "{hyp}" / "report.html"),
    run:
        from cellquorum.workflow.bundle import assemble_bundle
        entry = manifest[wildcards.hyp]
        run_dirs = {ct: RUNS / wildcards.hyp / ct for ct in entry["cell_types"]}
        assemble_bundle(wildcards.hyp, entry.get("title", wildcards.hyp),
                        run_dirs, BUNDLES / wildcards.hyp)

rule aggregate_status:
    input:
        [str(RUNS / h / ct / "provenance" / "artifact_manifest.csv") for h, ct in PAIRS],
        accounting=str(GEN / "accounting.json"),
    output:
        csv=str(RUNS / "matrix_status.csv"),
        md=str(RUNS / "matrix_status.md"),
    run:
        import json
        from cellquorum.workflow.status import build_matrix, matrix_to_csv, matrix_to_markdown
        accounting = json.loads(Path(input.accounting).read_text())
        run_records = {}
        for h, ct in PAIRS:
            rec = RUNS / h / ct / "provenance" / "stage_execution_records.json"
            run_records[f"{h}__{ct}"] = json.loads(rec.read_text()) if rec.exists() else {"records": []}
        rows = build_matrix(accounting, run_records)
        Path(output.csv).write_text(matrix_to_csv(rows))
        Path(output.md).write_text(matrix_to_markdown(rows))
```

Create the real `workflow/hypotheses.yaml` seeded from the track sheet Table 1. Populate every hypothesis with `title`, `cell_types`, `inputs` (use `TODO_SET_PATH` placeholders for data paths — these are real deployment inputs, documented as such in `docs/snakemake.md`), and `skip`/`blocked` where the track sheet marks a method N/A or unbuilt. Seed at least these entries with correct `cell_types`:

```yaml
emt_krt_high:
  title: "EMT / KRT-high keratinocyte program"
  cell_types: [KC]
  inputs: {KC: TODO_SET_PATH}
il33_st2_alarmin:
  title: "IL33/ST2 alarmin axis"
  cell_types: [KC, ILC]
  inputs: {KC: TODO_SET_PATH, ILC: TODO_SET_PATH}
  blocked:
    rna_velocity: "spliced/unspliced not yet generated for these subsets"
par2_f2rl1:
  title: "PAR2 / F2RL1 protease-activated signaling"
  cell_types: [KC]
  inputs: {KC: TODO_SET_PATH}
piezo_yap_tead:
  title: "PIEZO / YAP-TEAD mechanosensing"
  cell_types: [Fibroblast]
  inputs: {Fibroblast: TODO_SET_PATH}
lec_endomt:
  title: "LEC EndoMT"
  cell_types: [LEC]
  inputs: {LEC: TODO_SET_PATH}
  blocked:
    rna_velocity: "EndoMT velocity not wired for LEC"
il13_th2_mast:
  title: "IL13 / Th2 mast-cell state"
  cell_types: [Mast]
  inputs: {Mast: TODO_SET_PATH}
kc_ilc_il33_axis:
  title: "KC<->ILC IL33 crosstalk"
  cell_types: [KC, ILC]
  inputs: {KC: TODO_SET_PATH, ILC: TODO_SET_PATH}
fib_kc_periostin:
  title: "Fibroblast<->KC periostin axis"
  cell_types: [Fibroblast, KC]
  inputs: {Fibroblast: TODO_SET_PATH, KC: TODO_SET_PATH}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_snakefile_dag.py -v`
Expected: PASS. Verify `snakemake -n` prints a job for each `(hypothesis, cell_type)`, each bundle, and `aggregate_status`. If `run:`-directive rules cannot import `cellquorum.workflow` inside the Snakemake process, prepend `sys.path`/`PYTHONPATH` handling in the Snakefile (the package is pip-installed in the dev/image env, so this should already resolve).

- [ ] **Step 5: Commit**

```bash
git add workflow/Snakefile workflow/rules/matrix.smk workflow/template.yaml workflow/hypotheses.yaml tests/workflow/test_snakefile_dag.py
git commit -m "feat: Snakemake DAG (run per hypothesis-celltype, bundle, status) + seed manifest"
```

---

### Task 8: Multi-stage Dockerfile (C2)

Multi-stage build creating the primary env + all five backend envs (CPU-complete), and a GPU target extending it. Env names must be exact. scclr's package source is unknown, so its env bakes the toolchain and installs `scclr` via a build ARG (default empty → toolchain-only, documented).

**Files:**
- Create: `docker/Dockerfile`, `docker/.dockerignore`
- Test: none automated (image build is a `make` verification, Task 10). This task's deliverable is verified by a lint/parse check.

**Interfaces:**
- Consumes: `envs/*.yml` (Tasks 1), the package source.
- Produces: build targets `cpu` and `gpu`; entrypoint `micromamba run -n cellquorum-core cellquorum`.

- [ ] **Step 1: Write the Dockerfile**

Create `docker/.dockerignore`:

```
.git
.claude
.superpowers
runs
bundles
generated
**/__pycache__
*.pyc
docs/superpowers
```

Create `docker/Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1

########## base: primary env + package ##########
FROM mambaorg/micromamba:1.5.8 AS base
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /opt/cellquorum
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/cellquorum-core.yml /tmp/core.yml
RUN micromamba create -y -n cellquorum-core -f /tmp/core.yml && micromamba clean -a -y
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/cellquorum
RUN micromamba run -n cellquorum-core pip install -e .

########## backends: five isolated envs ##########
FROM base AS backends
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/celloracle_env.yml /tmp/celloracle_env.yml
RUN micromamba create -y -n celloracle_env -f /tmp/celloracle_env.yml && micromamba clean -a -y
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/pyscenic_env.yml /tmp/pyscenic_env.yml
RUN micromamba create -y -n pyscenic_env -f /tmp/pyscenic_env.yml && micromamba clean -a -y
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/hdwgcna_env.yml /tmp/hdwgcna_env.yml
RUN micromamba create -y -n hdwgcna_env -f /tmp/hdwgcna_env.yml && micromamba clean -a -y
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/sccoda_env.yml /tmp/sccoda_env.yml
RUN micromamba create -y -n sccoda_env -f /tmp/sccoda_env.yml && micromamba clean -a -y
# scclr: bake toolchain from recipe; install the package only if SCCLR_SRC is provided.
ARG SCCLR_SRC=""
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/scclr.yml /tmp/scclr.yml
RUN micromamba create -y -n scclr -f /tmp/scclr.yml && micromamba clean -a -y
RUN if [ -n "$SCCLR_SRC" ]; then micromamba run -n scclr pip install "$SCCLR_SRC"; \
    else echo "SCCLR_SRC not set: scclr toolchain baked, package NOT installed"; fi

########## cpu: base + backends + R env ##########
FROM backends AS cpu
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/cellquorum-r.yml /tmp/r.yml
RUN micromamba create -y -n cellquorum-r -f /tmp/r.yml && micromamba clean -a -y
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "micromamba", "run", "-n", "cellquorum-core", "cellquorum"]
CMD ["--help"]

########## gpu: cpu + gpu env ##########
FROM cpu AS gpu
COPY --chown=$MAMBA_USER:$MAMBA_USER envs/cellquorum-gpu.yml /tmp/gpu.yml
RUN micromamba create -y -n cellquorum-gpu -f /tmp/gpu.yml && micromamba clean -a -y
```

- [ ] **Step 2: Parse-check the Dockerfile**

Run: `docker build --check -f docker/Dockerfile . 2>&1 | head -40` (if `docker` present). Expected: no syntax errors reported. If `docker` is unavailable in this environment, run a grep sanity check instead: confirm all five env names and both `--target` stage names appear:

Run: `grep -E "micromamba create -y -n (celloracle_env|pyscenic_env|hdwgcna_env|scclr|sccoda_env)" docker/Dockerfile | wc -l`
Expected: `5`.

- [ ] **Step 3: Commit**

```bash
git add docker/Dockerfile docker/.dockerignore
git commit -m "feat: multi-stage Dockerfile baking all backend envs (cpu + gpu)"
```

---

### Task 9: Self-contained smoke config (C2)

A portable config for `make smoke` that never opens external data and never fails-fast on a CPU image.

**Files:**
- Create: `docker/smoke/smoke.yaml`
- Test: `tests/workflow/test_smoke_config.py`

**Interfaces:**
- Consumes: the engine config schema.
- Produces: `docker/smoke/smoke.yaml` that passes `cellquorum plan --config docker/smoke/smoke.yaml --json` (planning does not open input data).

- [ ] **Step 1: Write the failing test**

Create `tests/workflow/test_smoke_config.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml

from cellquorum.config.loader import validate_config_dict

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_smoke_config_is_cpu_and_valid() -> None:
    doc = yaml.safe_load((REPO_ROOT / "docker" / "smoke" / "smoke.yaml").read_text())
    cfg = validate_config_dict(doc)
    assert cfg.compute.backend == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_smoke_config.py -v`
Expected: FAIL — file missing.

- [ ] **Step 3: Write the smoke config**

Create `docker/smoke/smoke.yaml`:

```yaml
project:
  name: docker_smoke
input:
  h5ad: /nonexistent/smoke.h5ad
  counts_layer: counts
compute:
  backend: cpu
run:
  random_seed: 1337
report:
  enabled: false
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_smoke_config.py -v`
Expected: PASS. If validation requires additional fields, add only the minimum the schema error names, keeping `compute.backend: cpu`.

- [ ] **Step 5: Commit**

```bash
git add docker/smoke/smoke.yaml tests/workflow/test_smoke_config.py
git commit -m "feat: self-contained CPU smoke config for image verification"
```

---

### Task 10: Makefile, env README, and docs (C7 + C1b)

Operational entry points and documentation. `make lock` is the lock-generation step (heavyweight, network-bound — a documented operational target, not a unit test). `make smoke` runs the three-part image check.

**Files:**
- Create: `Makefile`, `envs/README.md`, `docs/docker.md`, `docs/snakemake.md`
- Test: `tests/workflow/test_makefile_targets.py`

**Interfaces:**
- Consumes: version from `cellquorum.version.__version__`; Dockerfile targets `cpu`/`gpu`; smoke config; Snakefile.
- Produces: `make image`, `make image-gpu`, `make lock`, `make smoke`, `make matrix` targets.

- [ ] **Step 1: Write the failing test**

Create `tests/workflow/test_makefile_targets.py`:

```python
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_makefile_declares_required_targets() -> None:
    text = (REPO_ROOT / "Makefile").read_text()
    targets = set(re.findall(r"^([a-zA-Z0-9_-]+):", text, re.MULTILINE))
    assert {"image", "image-gpu", "lock", "smoke", "matrix"} <= targets


def test_smoke_target_runs_three_checks() -> None:
    text = (REPO_ROOT / "Makefile").read_text()
    # version, plan, and env-list assertions must all appear in the smoke recipe.
    assert "--version" in text
    assert "plan --config docker/smoke/smoke.yaml" in text
    for env in ("celloracle_env", "pyscenic_env", "hdwgcna_env", "scclr", "sccoda_env"):
        assert env in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_makefile_targets.py -v`
Expected: FAIL — Makefile missing.

- [ ] **Step 3: Write the Makefile and docs**

Create `Makefile`:

```makefile
VERSION := $(shell python -c "import cellquorum; print(cellquorum.__version__)")
IMAGE := cellquorum:$(VERSION)
IMAGE_GPU := cellquorum:$(VERSION)-gpu
REQUIRED_ENVS := cellquorum-core celloracle_env pyscenic_env hdwgcna_env scclr sccoda_env cellquorum-r

.PHONY: image image-gpu lock smoke matrix

image:
	docker build --target cpu -t $(IMAGE) -f docker/Dockerfile .

image-gpu:
	docker build --target gpu -t $(IMAGE_GPU) -f docker/Dockerfile .

lock:
	@command -v conda-lock >/dev/null || { echo "install conda-lock first"; exit 1; }
	for f in envs/*.yml; do \
	  [ -s "$$f" ] || continue; \
	  conda-lock lock -f "$$f" --lockfile "$${f%.yml}.conda-lock.yml" || exit 1; \
	done

smoke:
	docker run --rm $(IMAGE) --version
	docker run --rm --entrypoint micromamba $(IMAGE) run -n cellquorum-core cellquorum plan --config docker/smoke/smoke.yaml --json
	@for env in $(REQUIRED_ENVS); do \
	  docker run --rm --entrypoint micromamba $(IMAGE) env list | grep -qw $$env \
	    || { echo "MISSING ENV: $$env"; exit 1; }; \
	done
	@echo "smoke OK"

matrix:
	docker run --rm -v $(PWD):/work -w /work --entrypoint micromamba $(IMAGE) \
	  run -n cellquorum-core snakemake --snakefile workflow/Snakefile --cores $(or $(CORES),4) --keep-going
```

Create `envs/README.md` documenting: the primary envs (core/dev/gpu/r), the five isolated backend envs and why they are isolated (verbatim rationale from each backend docstring — pyscenic pins numpy 1.23.5/pandas 1.5.3, scclr pins anndata<0.10.9 + Py3.13 cap, sccoda needs old scipy.signal.gaussian), the exact-env-name constraint, and `make lock` usage. Note that `scclr` bakes only the toolchain unless `SCCLR_SRC` is passed to `docker build`.

Create `docs/docker.md`: how to `make image` / `make image-gpu`, the `SCCLR_SRC` build arg, `make smoke`, and running `cellquorum run` inside the image.

Create `docs/snakemake.md`: the manifest format (scaffold-by-default, `skip`/`blocked` with reasons, `gene_programs`, `config_overrides`, `inputs` with `TODO_SET_PATH`), how to add a hypothesis, `make matrix`, and interpreting `runs/matrix_status.md`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_makefile_targets.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Makefile envs/README.md docs/docker.md docs/snakemake.md tests/workflow/test_makefile_targets.py
git commit -m "feat: Makefile (image/lock/smoke/matrix) + env and workflow docs"
```

---

### Task 11: Full workflow suite green + final integration check

Confirm the whole new subpackage passes and nothing else regressed.

**Files:** none (verification task).

- [ ] **Step 1: Run the workflow test suite**

Run: `python -m pytest tests/workflow -v`
Expected: all PASS (snakemake DAG test PASS if snakemake installed, else SKIP with a clear reason — install it if skipped so it runs).

- [ ] **Step 2: Run the broader suite for regressions**

Run: `python -m pytest tests/ -q`
Expected: no NEW failures attributable to this work. Pre-existing failures unrelated to `workflow/`, Docker, or env recipes are out of scope — note them but do not fix here.

- [ ] **Step 3: Lint the new code**

Run: `micromamba run -n cellquorum-dev ruff check src/cellquorum/workflow tests/workflow`
Expected: clean. Fix any findings (line-length 100, annotations required on non-test funcs).

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -A
git commit -m "chore: lint workflow subpackage" || echo "nothing to commit"
```
