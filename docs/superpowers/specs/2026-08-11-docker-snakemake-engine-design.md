# CellQuorum Reproducible Engine: Docker + Snakemake — Design Spec

**Date:** 2026-08-11
**Status:** Approved design (pending user review of this document)
**Sub-project:** A+B of the "cellquorum-as-engine" track (A = Package + Docker, B = Snakemake orchestration). Sub-project C (per-analysis project-repo template + KC migration) is a separate follow-on spec.

## Problem & Goal

The Mehrara Lab scRNA analysis portfolio is a sprawl of one-off analysis repos, each re-implementing pipeline logic. The `cellquorum` engine already runs a *single* analysis reproducibly (`cellquorum run --config X`: planner, backend registry, provenance, artifact manifest, standardized run dirs). What is missing is the layer that makes the engine **distributable and multi-analysis**:

1. A **canonical reproducible artifact** (Docker image) that recreates the whole multi-backend environment with zero host setup.
2. An **orchestration layer** (Snakemake) that expands the track-sheet matrix (cell types × 7 methods) into a DAG of `cellquorum run` jobs.

**End state (vision, beyond this spec):** each analysis (KC, Fibroblast, LEC, …) becomes a thin project repo containing only a config + data manifest + a pinned image tag + a Snakefile — no analysis code. This spec builds the engine-side foundation those repos will pin.

**This spec's goal:** produce (a) a versioned, lock-pinned package + Docker images (CPU-complete and GPU) that bake every backend env, and (b) a Snakemake workflow that runs the cell-type × method matrix by invoking `cellquorum run`, inside the image.

## Global Constraints

- **No publishing.** Build + tag locally (GHCR-ready naming), do NOT push to any registry or index. Distribution is a later decision.
- **Keep-local.** No pushing the branch to remote; local merge only. `docs/superpowers/` stays gitignored (local commit only, via `-f`).
- **Zero new analysis methods.** This is orchestration + packaging over the engine that already exists. Track-sheet methods that need a not-yet-built engine method are declared `blocked:` in the matrix — honest gaps, never faked.
- **Exact env names.** The Dockerfile MUST create micromamba envs with the exact names the backends hardcode, or subprocess backends break: `celloracle_env`, `pyscenic_env`, `hdwgcna_env`, `scclr`, `sccoda_env` (plus the primary env for core/R/GPU). Source of truth: `src/cellquorum/backends/*_backend.py` `env_name` fields.
- **Reproducibility.** Images build from pinned lockfiles, not floating `>=` specs. Determinism seeds already handled by the engine.
- **Determinism of the DAG.** `snakemake -n` must expand to a stable, inspectable target set derived from the matrix manifest — no hidden targets.

## Architecture

Two artifacts, layered:

```
┌─────────────────────────────────────────────────────────────┐
│ workflow/Snakefile  (runs INSIDE the image)                  │
│   matrix.yaml ──► config generator ──► N cellquorum configs  │
│                          │                                    │
│                          ▼                                    │
│   DAG: one rule instance per (cell_type, method) target      │
│        each runs `cellquorum run --config <generated>`        │
│                          │                                    │
│                          ▼                                    │
│   per-target run dir (provenance + artifact_manifest already) │
│                          │                                    │
│                          ▼                                    │
│   aggregate: top-level matrix status report                  │
└─────────────────────────────────────────────────────────────┘
        runs within
┌─────────────────────────────────────────────────────────────┐
│ Docker image  cellquorum:<ver>  (CPU-complete)               │
│   micromamba base                                            │
│   ├─ primary env: core + package (pip -e .)                  │
│   ├─ celloracle_env   ├─ pyscenic_env   ├─ hdwgcna_env        │
│   ├─ scclr            └─ sccoda_env      (+ R/Bioconductor)   │
│                                                              │
│ Docker image  cellquorum:<ver>-gpu  (extends CPU)            │
│   CUDA base + rapids-singlecell + scvi/scArches +            │
│   tensor-cell2cell on top of the same package/backend layers │
└─────────────────────────────────────────────────────────────┘
```

**Division of responsibility:** Snakemake owns the *DAG* (dependencies, resume, parallelism, per-target logs). CellQuorum owns *each single run* (already solved). The image owns *the environment*. No overlap.

## Components & Boundaries

### C1. Environment lockfiles — `envs/*.lock.yml` (or conda-lock `*.lock`)
- **Responsibility:** freeze the existing `envs/*.yml` (core, r, gpu, and the isolated backend envs) to pinned, reproducible lockfiles. This is *the packaging deliverable* — the "actual package" gets a frozen dependency set.
- **Interface:** `conda-lock` (or `micromamba env export --explicit`) per source yml. Lockfiles are committed; images build from them.
- **Depends on:** existing `envs/*.yml`. No new dependency choices — pinning only.
- **Note:** the isolated backend envs (`celloracle_env`, `pyscenic_env`, `hdwgcna_env`, `scclr`, `sccoda_env`) do not all have `envs/*.yml` files today. Where a backend env is created ad-hoc (e.g. celloracle_env was hand-built), this sub-project adds a real `envs/<name>.yml` capturing the working recipe (celloracle recipe is known: micromamba install numpy cython pandas scipy scikit-learn numba matplotlib h5py louvain python-igraph "setuptools<81" → pip --no-build-isolation velocyto → pip celloracle), then locks it.

### C2. Dockerfile — `docker/Dockerfile`
- **Responsibility:** multi-stage build producing the CPU-complete and GPU images.
- **Stages:**
  - `base`: `mambaorg/micromamba` + primary env (core + `pip install -e .`).
  - `backends` (extends base): create each isolated env by name from its lockfile.
  - `cpu` (final CPU target = base + backends + R env).
  - `gpu` (extends cpu): CUDA base layer + gpu env (rapids-singlecell, scvi/scArches, tensor-cell2cell). Because GPU extends the CPU target, package + backend layers are shared, not duplicated.
- **Interface:** `docker build --target cpu -t cellquorum:<ver> .` and `--target gpu -t cellquorum:<ver>-gpu .`.
- **Entrypoint:** `micromamba run -n <primary> cellquorum` so `docker run cellquorum:<ver> run --config ...` works directly.
- **Depends on:** C1 lockfiles.

### C3. Version wiring — `src/cellquorum/version.py` + `pyproject.toml`
- **Responsibility:** single source of version truth; the image tag derives from it (`cellquorum:$(python -c 'import cellquorum; print(cellquorum.__version__)')` or a `make` var). Bump `0.1.0` → a tagged engine version for pinning.
- **Interface:** `cellquorum --version` already exists; ensure it matches `pyproject` and drives the tag.

### C4. Matrix manifest — `workflow/matrix.yaml`
- **Responsibility:** declare the analysis surface — `{cell_type: {methods: [...], input: <h5ad>, config_overrides: {...}}}` mirroring the track sheet's 7-method core matrix. Methods map to cellquorum stages/config toggles.
- **Interface:** consumed by C5. A method a cell type doesn't run is simply absent; a method whose engine support isn't built yet is listed under `blocked: [...]` with a reason string (surfaced in the status report, never silently run).
- **Depends on:** the track sheet (cell types: KC, Fibroblast, DC, T cell, ILC, LEC, Mast; methods: pseudobulk, subclustering, pathway enrichment, RNA velocity, PHATE/pseudotime, cell–cell communication, PROGENy).

### C5. Config generator — `workflow/gen_configs.py`
- **Responsibility:** expand `matrix.yaml` × a base config template into N validated cellquorum configs under `workflow/configs/<cell_type>__<method>.yaml`. Avoids hand-maintaining 40+ YAMLs.
- **Interface:** `gen_configs(matrix: dict, template: dict) -> dict[str, dict]` (pure, unit-tested: matrix → expected config dicts). CLI wrapper writes them to disk.
- **Depends on:** existing config schema (`cellquorum.config.models.CellQuorumConfig`) — generated configs must validate against it.

### C6. Snakemake workflow — `workflow/Snakefile` + `workflow/rules/matrix.smk`
- **Responsibility:** read matrix → define one `run_analysis` rule instance per `(cell_type, method)` target → each shells `cellquorum run --config workflow/configs/<ct>__<method>.yaml -o runs/<ct>/<method>`. `rule all` collects the per-target artifact manifests. A final `aggregate_status` rule writes `runs/matrix_status.{csv,md}`.
- **Interface:** `snakemake --cores N` (real run), `snakemake -n` (dry-run DAG preview). Runs inside the image.
- **Target/dependency:** target = the per-run `provenance/artifact_manifest.csv`. Methods with intra-cell-type ordering (e.g. subclustering before velocity) express it as Snakemake input dependencies; independent methods run in parallel.
- **Depends on:** C5 outputs, the image (C2).

### C7. Make targets + docs — `Makefile`, `docs/docker.md`, `docs/snakemake.md`
- **Responsibility:** `make image`, `make image-gpu`, `make lock`, `make smoke`, `make matrix`. Docs: build the image, run the matrix, add a cell type/method.

## Data Flow

`matrix.yaml` → `gen_configs.py` → N validated cellquorum configs → Snakemake DAG → `cellquorum run` per target (inside image) → each writes its standard run dir (provenance + `artifact_manifest.csv`) → `aggregate_status` → `runs/matrix_status.{csv,md}` (per target: succeeded / skipped / failed / blocked, with the artifact count).

## Error Handling & Skip Semantics

- **Blocked methods** (engine support not built): declared in `matrix.yaml`, emitted into the status report as `blocked` with the reason, never dispatched as jobs. This is how the ⬜ track-sheet cells (KC/DC RNA velocity, LEC in-silico KO, EndoMT) stay visible without being faked.
- **Skipped stages** (engine's existing MethodSkip): a target can succeed with skipped internal stages; the status report reflects the engine's own summary (already in the run JSON).
- **Failed target:** Snakemake marks the job failed; `--keep-going` lets the rest of the matrix finish; the status report lists failures. A failed target does not corrupt sibling targets (separate run dirs).
- **GPU-required target on CPU image:** the engine already fails-fast when `compute.backend: gpu` and CUDA is absent; the matrix routes GPU targets to the `-gpu` image (a matrix field `image: gpu|cpu` per method, defaulting cpu).

## Testing

- **C5 config generator:** unit tests — a small matrix + template produces the expected config dicts; every generated config validates via `validate_config_dict`. Edge cases: empty method list, blocked method excluded from output, override merge precedence.
- **C6 Snakefile:** a `snakemake -n` dry-run test asserts the DAG expands to exactly the expected target set for a fixture matrix (no missing/extra targets). Uses a tiny fixture matrix, not the real one.
- **C2 image:** `make smoke` runs the existing `configs/le_smoke.yaml` inside the freshly built CPU image and asserts exit 0 + expected artifact manifest — reuses the engine's existing smoke path. (Image build itself is not unit-tested in CI; it's a `make` verification.)
- **C1 locks:** a test asserting each `envs/*.yml` has a corresponding committed lockfile and that env names referenced by backends exist among the env set (guards the "exact env names" constraint).

## Explicitly Out of Scope (YAGNI)

- PyPI / Bioconda / Docker registry publishing.
- The cookiecutter per-analysis project-repo template and the KC pilot migration (Sub-project C).
- Any new analysis method implementation (velocity for KC/DC, LEC in-silico KO wiring, EndoMT program) — those are engine-method specs; here they are matrix entries, `blocked` until built.
- Kubernetes / cloud executors for Snakemake (local `--cores` only for now).
- Re-homing the ✅ analyses' *results*; this builds the machine that will regenerate them (migration is Sub-project C+).

## Open Questions (resolved during brainstorming)

- Reproducibility unit → Docker image canonical + conda lock escape hatch. ✅
- Distribution → set up now, do not publish. ✅
- First spec scope → Docker + Snakemake together (A+B). ✅
- GPU → first-class in this sub-project (rapids-singlecell / scArches / tensor-cell2cell); both CPU and GPU images are deliverables. ✅
