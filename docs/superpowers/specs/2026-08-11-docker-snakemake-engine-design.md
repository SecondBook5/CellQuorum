# CellQuorum Reproducible Engine: Docker + Snakemake — Design Spec

**Date:** 2026-08-11
**Status:** Approved design (pending user review of this document)
**Sub-project:** A+B of the "cellquorum-as-engine" track (A = Package + Docker, B = hypothesis-keyed Snakemake orchestration). Sub-project C (per-publication project-repo template + first pilot migration) is a separate follow-on spec.

**Core principle:** one implementing repo = one high-impact publication = one hypothesis (or tight hypothesis group). The repo is the paper's reproducible backend: its hypothesis manifest defines the figures/tables, cellquorum generates them, and the per-hypothesis bundle is the publication's regenerable methods+results.

## Problem & Goal

The Mehrara Lab scRNA analysis portfolio is a sprawl of one-off analysis repos, each re-implementing pipeline logic. The `cellquorum` engine already runs a *single* analysis reproducibly (`cellquorum run --config X`: planner, backend registry, provenance, artifact manifest, standardized run dirs). What is missing is the layer that makes the engine **distributable and multi-analysis**:

1. A **canonical reproducible artifact** (Docker image) that recreates the whole multi-backend environment with zero host setup.
2. An **orchestration layer** (Snakemake) that expands a **hypothesis-keyed** matrix into a DAG of `cellquorum run` jobs, emitting **one comprehensive, organized, publication-ready bundle per hypothesis**.

**Organizing unit = the hypothesis, not the cell type.** The track sheet proves this: Table 1 is a list of hypotheses/programs (EMT/KRT-high, IL33/ST2 alarmin, PAR2/F2RL1, PIEZO/YAP–TEAD mechanosensing, patient Th2 response score, LEC EndoMT, IL13/Th2 mast state, KC↔ILC IL33 axis, Fib↔KC periostin…). Table 0 (cell type × 7 methods) is the *core scaffold* each hypothesis draws from. Keying the matrix on hypotheses makes cross-cell-type hypotheses (KC↔ILC, Fib↔KC) first-class rather than orphaned between cell-type folders, and yields exactly the goal: **one single comprehensive organized deliverable per hypothesis, ready for publication.**

**End state (vision, beyond this spec):** each hypothesis (or a group of related hypotheses for one cell type) becomes a thin project repo containing only a hypothesis manifest + data manifest + a pinned image tag + a Snakefile — no analysis code. This spec builds the engine-side foundation those repos will pin.

**This spec's goal:** produce (a) a versioned, lock-pinned package + Docker images (CPU-complete and GPU) that bake **every** backend env, and (b) a Snakemake workflow that runs the hypothesis matrix by invoking `cellquorum run` inside the image and aggregates a per-hypothesis publication bundle.

## Global Constraints

- **No publishing.** Build + tag locally (GHCR-ready naming), do NOT push to any registry or index. Distribution is a later decision.
- **Keep-local.** No pushing the branch to remote; local merge only. `docs/superpowers/` stays gitignored (local commit only, via `-f`).
- **Zero new analysis methods.** This is orchestration + packaging over the engine that already exists. Track-sheet methods that need a not-yet-built engine method are declared `blocked:` in the matrix — honest gaps, never faked.
- **Bake ALL backends — do not trim.** The image bakes core + R + GPU + all five isolated backend envs so **no hypothesis is ever blocked by a missing environment**. A ~15GB always-works image beats a lean one that skips scCODA composition or hdWGCNA co-expression when a hypothesis needs it. Size is not optimized (not published yet); completeness is the point of a reusable lab tool.
- **Exact env names.** The Dockerfile MUST create micromamba envs with the exact names the backends hardcode, or subprocess backends break: `celloracle_env`, `pyscenic_env`, `hdwgcna_env`, `scclr`, `sccoda_env` (plus the primary env for core/R/GPU). Source of truth: `src/cellquorum/backends/*_backend.py` `env_name` fields.
- **Reproducibility.** Images build from pinned lockfiles, not floating `>=` specs. Determinism seeds already handled by the engine.
- **Determinism of the DAG.** `snakemake -n` must expand to a stable, inspectable target set derived from the matrix manifest — no hidden targets.

## Architecture

Two artifacts, layered:

```
┌─────────────────────────────────────────────────────────────┐
│ workflow/Snakefile  (runs INSIDE the image)                  │
│   hypotheses.yaml ──► config generator ──► N cellquorum cfgs │
│      (per hypothesis: cell_type(s), methods, gene programs)   │
│                          │                                    │
│                          ▼                                    │
│   DAG: one rule instance per (hypothesis, method) target      │
│        each runs `cellquorum run --config <generated>`        │
│                          │                                    │
│                          ▼                                    │
│   per-target run dir (provenance + artifact_manifest already) │
│                          │                                    │
│                          ▼                                    │
│   per-hypothesis bundle: figures + tables + rendered report   │
│   under bundles/<hypothesis>/  (the publication-ready unit)   │
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
- **Responsibility:** freeze the existing `envs/*.yml` (core, r, gpu, and ALL isolated backend envs) to pinned, reproducible lockfiles. This is *the packaging deliverable* — the "actual package" gets a frozen dependency set.
- **Interface:** `conda-lock` (or `micromamba env export --explicit`) per source yml. Lockfiles are committed; images build from them.
- **Depends on:** existing `envs/*.yml`. No new dependency choices — pinning only.
- **Scope:** lock every env the image bakes — core, r, gpu, `celloracle_env`, `pyscenic_env`, `hdwgcna_env`, `scclr`, `sccoda_env` (bake-all constraint). The isolated backend envs do not all have `envs/*.yml` files today. Where a backend env is created ad-hoc (e.g. celloracle_env was hand-built), this sub-project adds a real `envs/<name>.yml` capturing the working recipe (celloracle recipe is known: micromamba install numpy cython pandas scipy scikit-learn numba matplotlib h5py louvain python-igraph "setuptools<81" → pip --no-build-isolation velocyto → pip celloracle), then locks it.

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

### C4. Hypothesis manifest — `workflow/hypotheses.yaml`
- **Responsibility:** declare the analysis surface **keyed by hypothesis** — the publication-scoped unit. Each entry:
  ```yaml
  <hypothesis_id>:
    title: "IL33/ST2 alarmin KC→ILC2 axis"      # human-readable, drives report title
    cell_types: [KC, ILC]                         # one or more (cross-type is first-class)
    inputs: {KC: <h5ad>, ILC: <h5ad>}             # per cell type
    # NO `methods:` key → the hypothesis inherits the FULL Table 0 scaffold.
    # You subtract, you never add:
    skip: [pseudobulk]                             # N/A for this hypothesis; reason recorded
    blocked: [rna_velocity]                        # engine support not built; reason in status
    gene_programs:                                # the hypothesis-specific biology
      alarmin: [IL33, IL1RL1, IL13, ...]
    config_overrides: {...}                        # per-hypothesis knobs
  ```
- **Scaffold-by-default (the "can't forget a step" guarantee):** the manifest is **opt-out, not opt-in**. Every hypothesis runs the complete Table 0 core scaffold (`pseudobulk, subclustering, pathway_enrichment, rna_velocity, phate_pseudotime, cell_cell_communication, progeny`) *unless* a method appears in `skip` (deliberately N/A — reason recorded) or `blocked` (engine support not built — reason in status). There is no `methods:` allow-list to forget to update; the only way a scaffold method does not run is an explicit, recorded decision. An optional `extra_methods:` key adds Table-1-specific programs beyond the scaffold. C5 must account for **every** scaffold method for **every** hypothesis as exactly one of {run, skip, blocked} — see the completeness check in C5.
- **Interface:** consumed by C5. `skip`/`blocked` methods are surfaced in the status report with their reason, never silently omitted. `gene_programs` feed the scoring/enrichment/target-figure stages so the biology is declared, not hardcoded — preserving the engine's zero-study-specific-biology invariant (biology lives in the manifest, engine stays generic).
- **Depends on:** the track sheet — Table 1 hypotheses (EMT/KRT-high, IL33/ST2 alarmin, PAR2/F2RL1, PIEZO/YAP–TEAD mechanosensing, patient Th2 response score, LEC EndoMT, IL13/Th2 mast state, KC↔ILC IL33 axis, Fib↔KC periostin, …) and the Table 0 core scaffold (methods: pseudobulk, subclustering, pathway enrichment, RNA velocity, PHATE/pseudotime, cell–cell communication, PROGENy).
- **Note on repo mapping:** one implementing repo = one high-impact publication = one hypothesis (or a tightly-related hypothesis group). A repo's `hypotheses.yaml` may hold a single entry or a small related set; the workflow treats each entry independently and bundles each.

### C5. Config generator — `workflow/gen_configs.py`
- **Responsibility:** expand `hypotheses.yaml` × a base config template into N validated cellquorum configs under `workflow/configs/<hypothesis>__<method>.yaml`. For each hypothesis, resolve the scaffold: `run = SCAFFOLD - skip - blocked` (plus `extra_methods`), emitting one config per `run` method. Gene programs + overrides are merged into each config so the run is fully specified. Avoids hand-maintaining dozens of YAMLs.
- **Completeness check (the guarantee's enforcement):** `gen_configs` fails loudly if, for any hypothesis, the union of `run ∪ skip ∪ blocked` does not equal the full scaffold — i.e. a scaffold method that is neither run nor explicitly accounted for is an error, not a silent omission. It also errors on an unknown method name (typo guard) or a method in two categories at once. This is what makes "forgot a step" structurally impossible.
- **Interface:** `gen_configs(hypotheses: dict, template: dict, scaffold: list[str]) -> dict[str, dict]` (pure, unit-tested: manifest → expected config dicts; raises on an incomplete or inconsistent manifest). CLI wrapper writes the configs to disk and also emits a per-hypothesis `{run, skip, blocked}` accounting consumed by the status report.
- **Depends on:** existing config schema (`cellquorum.config.models.CellQuorumConfig`) — generated configs must validate against it.

### C6. Snakemake workflow — `workflow/Snakefile` + `workflow/rules/matrix.smk`
- **Responsibility:** read manifest → define one `run_analysis` rule instance per `(hypothesis, method)` target → each shells `cellquorum run --config workflow/configs/<hyp>__<method>.yaml -o runs/<hyp>/<method>`. A `bundle_hypothesis` rule then collects every method's figures + tables + run summaries for one hypothesis into `bundles/<hypothesis>/` with a rendered report (title from the manifest) — **the publication-ready deliverable**. `rule all` collects all per-hypothesis bundles. A final `aggregate_status` rule writes `runs/matrix_status.{csv,md}`.
- **Interface:** `snakemake --cores N` (real run), `snakemake -n` (dry-run DAG preview). Runs inside the image.
- **Target/dependency:** per-method target = the run's `provenance/artifact_manifest.csv`; per-hypothesis target = `bundles/<hypothesis>/report.html` (+ its figure/table tree). Methods with intra-hypothesis ordering (e.g. subclustering before velocity/CCC) express it as Snakemake input dependencies; independent methods run in parallel. Bundling depends on all of a hypothesis's method targets.
- **Depends on:** C5 outputs, the image (C2), the engine's existing report renderer (`cellquorum.reports`) where available.

### C7. Make targets + docs — `Makefile`, `docs/docker.md`, `docs/snakemake.md`
- **Responsibility:** `make image`, `make image-gpu`, `make lock`, `make smoke`, `make matrix`. Docs: build the image, run a hypothesis manifest, add a hypothesis/method, interpret the bundle.

## Data Flow

`hypotheses.yaml` → `gen_configs.py` → N validated cellquorum configs → Snakemake DAG → `cellquorum run` per `(hypothesis, method)` target (inside image) → each writes its standard run dir (provenance + `artifact_manifest.csv`) → `bundle_hypothesis` collects a hypothesis's method outputs into `bundles/<hypothesis>/` (figures + tables + `report.html`) → `aggregate_status` → `runs/matrix_status.{csv,md}` (per target: succeeded / skipped / failed / blocked, with artifact count) + a top-level index of the per-hypothesis bundles.

## Error Handling & Skip Semantics

- **Scaffold accounting** (the completeness guarantee): because the manifest is opt-out, every hypothesis is checked against the full Table 0 scaffold at config-generation time. A scaffold method must be exactly one of run / `skip` / `blocked`; anything unaccounted for is a hard error before any job runs. The status report shows this `{run, skip, blocked}` breakdown per hypothesis so a reader sees the whole scaffold was considered, not just what happened to be listed.
- **Skipped methods** (`skip:` — deliberately N/A for this hypothesis): recorded in the status report as `skip` with the reason; not dispatched. Distinct from an engine-internal MethodSkip below.
- **Blocked methods** (engine support not built): declared per hypothesis in `hypotheses.yaml`, emitted into the status report as `blocked` with the reason, never dispatched as jobs. This is how the ⬜ track-sheet items (KC/DC RNA velocity, LEC in-silico KO, EndoMT) stay visible without being faked.
- **Skipped stages** (engine's existing MethodSkip): a target can succeed with skipped internal stages; the status report reflects the engine's own summary (already in the run JSON).
- **Failed target:** Snakemake marks the job failed; `--keep-going` lets the rest of the matrix finish; the status report lists failures. A failed target does not corrupt sibling targets (separate run dirs).
- **GPU-required target on CPU image:** the engine already fails-fast when `compute.backend: gpu` and CUDA is absent; the manifest routes GPU targets to the `-gpu` image (a per-method `image: gpu|cpu` field, defaulting cpu).

## Testing

- **C5 config generator:** unit tests — a small hypothesis manifest + template produces the expected config dicts; every generated config validates via `validate_config_dict`. Edge cases: scaffold-by-default (a hypothesis with no `skip`/`blocked` yields one config per scaffold method), `skip`/`blocked` methods excluded from output but present in the accounting, gene-program + override merge precedence, multi-cell-type hypothesis (per-input configs). **Completeness check must fail loudly:** a manifest that leaves a scaffold method unaccounted, names an unknown method, or lists a method in two categories raises — assert each raises.
- **C6 Snakefile:** a `snakemake -n` dry-run test asserts the DAG expands to exactly the expected `(hypothesis, method)` target set plus the per-hypothesis bundle target for a fixture manifest (no missing/extra targets). Uses a tiny fixture manifest, not the real one.
- **C6 bundling:** unit test of the bundle assembler — given fixture run dirs for a hypothesis's methods, it collects the expected figure/table set and emits a report with the manifest title.
- **C2 image:** `make smoke` runs the existing `configs/le_smoke.yaml` inside the freshly built CPU image and asserts exit 0 + expected artifact manifest — reuses the engine's existing smoke path. (Image build itself is not unit-tested in CI; it's a `make` verification.)
- **C1 locks:** a test asserting each `envs/*.yml` has a corresponding committed lockfile and that every backend `env_name` exists among the baked env set (guards the "exact env names" + "bake all" constraints).

## Explicitly Out of Scope (YAGNI)

- PyPI / Bioconda / Docker registry publishing.
- The cookiecutter per-publication project-repo template and the first pilot migration (Sub-project C). This spec builds + validates the machinery on a fixture/representative manifest inside the engine repo; standing up a real per-publication repo is C.
- Any new analysis method implementation (velocity for KC/DC, LEC in-silico KO wiring, EndoMT program) — those are engine-method specs; here they are manifest entries, `blocked` until built.
- Kubernetes / cloud executors for Snakemake (local `--cores` only for now).
- Re-homing the ✅ analyses' *results*; this builds the machine that will regenerate them (migration is Sub-project C+).

## Open Questions (resolved during brainstorming)

- Reproducibility unit → Docker image canonical + conda lock escape hatch. ✅
- Distribution → set up now, do not publish. ✅
- First spec scope → Docker + Snakemake together (A+B). ✅
- GPU → first-class in this sub-project (rapids-singlecell / scArches / tensor-cell2cell); both CPU and GPU images are deliverables. ✅
- Organizing unit → **hypothesis-keyed**: one implementing repo = one high-impact publication = one hypothesis (or tight hypothesis group); workflow emits one publication-ready bundle per hypothesis. ✅
- Backend baking → **bake all** (core + R + GPU + all five isolated envs); completeness over image size. ✅
