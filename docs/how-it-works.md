# How CellQuorum Works: A File-Level Walkthrough

This document walks you through one CellQuorum run from entry point to artifact, file by file. It's designed for collaborators who want to understand the control flow, locate implementation logic, or modify a stage — without needing the author present.

> **Paths below are relative to `src/cellquorum/`** — e.g. `cli/app.py` is
> `src/cellquorum/cli/app.py`, and `stages/qc/stage.py` is
> `src/cellquorum/stages/qc/stage.py`.

---

## The Two Front Doors

CellQuorum offers two entry points, both leading into the same validated execution pipeline:

### 1. Command-Line Interface (CLI)

**Entry point:** `cli/app.py:main`

The `cellquorum` or `cq` command is backed by a Typer application in `cli/app.py`. The `main()` function launches the app, which exposes two primary commands:

- **`cellquorum plan`** — validates configuration, checks backend availability, and prints a preview of which stages will run (without executing them)
- **`cellquorum run`** — validates configuration, bootstraps the run directory, and executes the full pipeline

Both commands parse arguments (config path, output directory, verbosity flags), load and validate the YAML configuration via `config/loader.py`, then delegate to `api/pipeline.py`.

### 2. Programmatic API

**Entry point:** `api/pipeline.py:run_pipeline`

The `run_pipeline` function is the main Python API for notebooks, scripts, and programmatic runs. It accepts:
- A YAML config path, a validated `CellQuorumConfig` Pydantic model, or a plain dictionary
- Optional output directory override
- Execution control flags (`execute=True` to run stages, `execute=False` for bootstrap-only)

The function normalizes the input into a validated `CellQuorumConfig` object, optionally overrides verbosity when `quiet=True` is passed, then routes to the core bootstrap and execution utilities in `core/pipeline.py`.

Both entry points converge on the same validation → planning → bootstrap → execution flow. The CLI adds Rich-formatted terminal output; the API returns structured `PipelineRunResult` objects.

---

## One Run, File by File

Here's how a typical `cellquorum run --config configs/config.yaml` flows through the codebase:

### Step 1: Entry and Config Validation

**`cli/app.py`** receives the command, parses CLI options, and calls `load_config` from `config/loader.py`. The loader reads the YAML file, resolves any Hydra composition, and validates it against the strict Pydantic schema in **`config/models.py`**.

`config/models.py` defines the entire configuration contract: `CellQuorumConfig` is the top-level model, with nested blocks for every stage (e.g., `qc: QCConfig`, `differential_expression: DifferentialExpressionConfig`). Each field is validated at load time — types are checked, enums are constrained, cross-field consistency rules run via `@model_validator`. If validation fails, the error is surfaced before any heavy computation begins.

Once validated, the config is a frozen Pydantic model that threads through the entire run.

### Step 2: Backend Registry and Planning

**`cli/app.py`** (or **`api/pipeline.py`**) next builds the backend registry via `backends/registry.py:build_default_backend_registry`. The registry probes the environment for available backends: can we import `cupy`? is `Rscript` on the PATH? The result is a `BackendRegistry` object that stages query to decide whether to dispatch GPU methods, R scripts, or CPU fallbacks.

The validated config and backend registry are passed to **`core/planner.py:build_pipeline_plan`**, which:
1. Reads the ordered list of all stages from **`core/stage_catalog.py`** (the `@register_stage` decorator catalog)
2. For each stage, checks its `config_flag` (e.g., `stages.qc`) to determine if it's enabled
3. Produces a `PipelinePlan` containing an ordered list of `PlannedStage` objects (each marked `enabled` or `disabled`) plus backend availability warnings

The planner **does not yet instantiate stages** — it only decides *which* stages will run. The full pipeline order lives in the catalog, populated at import time by `@register_stage` decorators on each stage class.

### Step 3: Bootstrap the Run Directory

**`core/pipeline.py:bootstrap_pipeline_run`** creates the standardized directory layout under `output_dir` (or `paths.run_root/<run_id>`):

```
<run_root>/<run_id>/
  results/          # stage CSV/JSON tables
  figures/          # visualizations
  reports/          # final HTML/Markdown reports
  objects/          # AnnData checkpoints
  provenance/       # config snapshots, manifest, plan, artifact list
  logs/             # stage execution logs
  scratch/          # temporary R inputs/outputs
```

Bootstrap writes the pre-execution provenance to `provenance/`, including:
- `resolved_config.json` — the resolved, validated configuration
- `pipeline_plan.json` / `stage_plan.csv` — the ordered stage plan with enablement reasons
- `backend_status.json` / `.csv` — backend availability at run start
- `planner_warnings.json` and `run_metadata.json` — planner warnings and the run identity / environment-version stamp

It also initializes the artifact manifest CSV, which stages append to as they produce outputs.

### Step 4: Load Input Data

**`io/manifest.py`** handles manifest loading if `paths.manifest` is set (for multi-sample runs), validating required columns (`sample_id`, `path`, optional `donor_id`/`condition`/`batch`) and resolving paths relative to `paths.data_root`.

For single-object runs, `input.h5ad` is loaded directly into `context.adata` via `anndata.read_h5ad` (in backed mode if `input.subset` is specified, so large global objects are sliced before full materialization).

The loaded data, validated config, pipeline plan, run paths, and backend registry are bundled into a `PipelineContext` object, which threads through every stage.

### Step 5: Execute Stages

**`core/executor.py`** owns the stage execution loop. The `PipelineExecutor` class:
1. Iterates over the ordered `plan.stages` list
2. For each `PlannedStage`, checks if it's enabled — if disabled, records a skip and continues
3. If enabled, looks up the stage implementation in the `StageRegistry` (built from **`core/stage_catalog.py`**'s implemented specs)
4. If no implementation exists, records a "planned but not yet implemented" skip
5. Otherwise, calls `stage.run(context)`, passing the current pipeline context

Each stage's `run` method:
- Reads its stage-specific config from `context.config.<stage_name>` (e.g., `context.config.qc`)
- Validates required inputs (layers, obs columns, sample count) — if missing, the stage returns `StageResult(status="skipped", skip_reason="...")`
- Computes its analysis (calling methods in `methods/`, `backends/`, or stage-internal modules)
- Writes artifacts via **`core/stage_artifact_writer.py`** (CSV tables, JSON summaries, figures) to `context.paths.results/<stage_subdir>/`
- Returns a `StageResult` containing the updated `adata`, a list of `StageArtifact` records, notes, warnings, and structured metrics

The executor:
- Threads the updated `adata` forward: `context = context.with_adata(stage_result.adata)`
- Records a `StageExecutionRecord` (success, skipped, or failed) with start/end timestamps and input/output fingerprints
- Writes artifacts to the manifest via **`core/stage_artifact_writer.py`**
- Stops on failure if `stop_on_failure=True` (the default), or continues if `continue_on_stage_failure=True`

At the end of execution, the executor returns a `PipelineExecutionResult` containing:
- The final `context` (with the fully processed `adata`)
- All successful `StageResult` objects (keyed by stage name)
- All `StageExecutionRecord` objects (for provenance)

### Step 6: Write Execution Provenance

**`core/pipeline.py`** (via `write_pipeline_provenance`) appends post-execution provenance to `provenance/`:
- `stage_execution_records.json` / `.csv` — per-stage success/skip/fail records, timestamps, fingerprints
- `artifact_manifest.csv` — the complete list of every CSV, figure, and h5ad produced

If `run.write_final_object=True` (the default), the final `adata` is written to `objects/final_annotated.h5ad`.

### Step 7: The Shape Every Stage Follows

**`stages/qc/stage.py`** is the canonical example of a stage's structure. Every stage:
1. Opens with a docstring header: `# Pipeline step (order=20): qc — ...`
2. Is decorated with `@register_stage(name="qc", order=20, config_flag="qc", config_field="qc")`
3. Defines a `@dataclass(frozen=True)` class with an optional `config` override and an `output_subdir` default
4. Implements `run(self, context) -> StageResult`, which:
   - Retrieves `adata = context.adata`
   - Resolves its config from `context.config.<stage_name>` (with fallback to the override)
   - Checks enablement and prerequisites — if not met, returns a skipped `StageResult`
   - Computes its analysis (calling domain modules like `stages/qc/metrics.py`, `stages/qc/thresholds.py`, `stages/qc/decisions.py`)
   - Writes artifacts to `context.paths.results/<stage_subdir>/`
   - Returns `StageResult(adata=updated_adata, artifacts=[...], notes=[...], warnings=[...], metrics={...})`

The executor never directly instantiates stages — it calls the `factory` function registered in the catalog, which is just the class itself (since `@register_stage` decorates the class).

---

## If You're Looking for X, Open Y

This table maps common developer questions to the files that answer them:

| **What you're looking for** | **Where to look** |
|------------------------------|-------------------|
| A stage's main logic | `stages/<stage_name>/stage.py` (e.g., `stages/qc/stage.py`, `stages/comparative/differential_expression/stage.py`) |
| A stage's configuration schema | `stages/<stage_name>/config.py` |
| A stage's visualization code | `visualization/<category>/` (shared) or `stages/<stage_name>/viz/` (stage-local) |
| Reusable analytics utilities | `cellquorum.utils` — a flat re-export surface (`utils/__init__.py`) exposing `aggregate_pseudobulk`, `de_table_to_ranking`, `get_net` (canonical code in `cellquorum.stages.comparative`) |
| Notebook-facing helpers | `api/tl.py`, `api/pp.py`, `api/diag.py`, `api/evidence.py` — the `cq.tl` / `cq.pp` / `cq.diag` / `cq.evidence` namespaces, re-exported via `api/__init__.py` |
| The ordered list of all stages | `stages/README.md` (the catalog table) |
| Config validation | `config/models.py` (Pydantic schemas) and `config/loader.py` (YAML loading) |
| Backend detection | `backends/registry.py` |
| Stage ordering and planning | `core/planner.py` (reads from `core/stage_catalog.py`) |
| Stage execution loop | `core/executor.py` |
| Artifact writing | `core/stage_artifact_writer.py` |
| Input data loading | `io/manifest.py` (manifests), `io/anndata.py` (AnnData h5ad) |
| Run directory layout | `core/context.py:PipelinePaths` |
| Provenance writing | `core/provenance.py` |
| Method registries | `methods/registry.py` (method contracts and dispatch) |
| GPU/RAPIDS support | `backends/gpu.py` |
| R/Rscript dispatch | `backends/rscript.py` |

### Key Structural Facts

- **One canonical import path per public thing.** The old top-level re-export shims (`cellquorum.differential_expression`, `cellquorum.tl`) were removed. Every stage, config, and utility has exactly one import path:
  - Stages: `cellquorum.stages.<stage_name>` (e.g., `cellquorum.stages.qc`, `cellquorum.stages.comparative.differential_expression`)
  - Config: `cellquorum.config.models` (top-level) or `cellquorum.stages.<stage_name>.config` (stage-local)
  - Notebook API: `cellquorum.api.tl` / `cellquorum.api.pp` / `cellquorum.api.diag` / `cellquorum.api.evidence` (separate modules, re-exported from `cellquorum.api`)
  - Utilities: `cellquorum.utils.<module>`

- **The stage catalog is the single source of truth.** The ordered list of stages, their enablement flags, and their config blocks are all declared once via `@register_stage`. The planner reads this catalog to order stages; the executor reads it to instantiate them. There is no hand-maintained "stage list" file.

- **Stages are self-contained packages.** Each stage lives in `cellquorum/stages/<stage_name>/`, with its own `stage.py` (the `@register_stage` class), `config.py` (the Pydantic config model), and any domain modules (e.g., `stages/qc/metrics.py`, `stages/qc/thresholds.py`). Some stages also have a `viz/` subpackage for stage-local figures, but shared visualization lives in `cellquorum/visualization/`.

- **Notebook and programmatic namespaces are in `cellquorum.api`.** The `cq.tl`, `cq.pp`, `cq.diag`, and `cq.evidence` namespaces (for users who want to call individual stages or methods interactively) are defined in `api/tl.py`, `api/pp.py`, `api/diag.py`, `api/evidence.py` (re-exported via `api/__init__.py`) and backed by the same stage classes and methods the pipeline uses.

- **Fail-loud data contracts.** Every stage validates its inputs (required layers, obs columns, sample support, etc.) at the top of its `run` method. If a prerequisite is missing, the stage returns a skipped `StageResult` with a recorded reason — it never crashes with a cryptic KeyError or silently produces wrong results.
