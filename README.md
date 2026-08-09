<div align="center">

# CellQuorum

### A reproducible single-cell RNA-seq workflow engine for publication-oriented analysis

CellQuorum provides a Python API, command-line interface, validated configuration system, backend-aware execution planning, standardized run outputs, and provenance tracking for advanced scRNA-seq workflows.

<br>

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20development-orange)
![Tests](https://img.shields.io/badge/tests-~1100-brightgreen)
![Stages](https://img.shields.io/badge/stages-20%20implemented-blue)
![GPU](https://img.shields.io/badge/GPU-rapids--singlecell-76b900)
![Interface](https://img.shields.io/badge/interface-CLI%20%7C%20Python-informational)
![Workflow](https://img.shields.io/badge/workflow-single--cell%20RNA--seq-purple)

</div>

---

## Overview

CellQuorum is designed to make advanced single-cell RNA-seq analysis easier to run without losing reproducibility, auditability, or scientific discipline.

The engine provides both the execution spine and a working config-driven analysis backbone:

**Execution spine**

- strict YAML/Pydantic configuration validation
- backend registry for Python, R/Rscript, GPU, and RAPIDS availability checks
- execution planner for enabled stages and backend status
- standardized run directory layout
- provenance artifact writing
- command-line interface
- public Python API
- pytest and pre-commit support

**Fail-loud data contracts** — every stage boundary validates the AnnData it
receives (required layers/obs/embeddings, layer-provenance tags, and statistical
sanity such as rejecting raw counts mislabeled as log-normalized). A method whose
required inputs are absent *skips with a recorded reason* rather than crashing or
silently producing wrong output.

**Config-driven analysis backbone** — twenty registered stages run in
best-practices order, each dispatching to a config-selected method:

```
ambient_correction → qc → preprocessing → dimensionality → integration
    → clustering → annotation → embeddings → differential_expression
    → differential_abundance → enrichment → enrichment_viz
```

Every method is chosen by config (e.g. `integration.method: harmony | scvi`),
so a dataset is expressed as a YAML config, not code. Downstream discovery
stages (gene-regulatory network construction, cell-cell communication,
trajectory/potency) are planned slots not yet implemented.

**GPU acceleration, by default when available** — normalization (PFlog1pPF via
cupy), PCA, and neighbors + Leiden (via rapids-singlecell) run on the GPU when
`rapids-singlecell` + `cupy` are installed and a CUDA device is present, and fall
back to scanpy (CPU) otherwise. A shared compute router gates on real capability
(not merely a visible device); set `compute.backend: cpu` to force CPU. Routed
methods produce identical output keys, so results and data contracts are
path-independent.

---

## Current capabilities

### Engine

| Capability | Status |
|---|---:|
| Installable Python package | Implemented |
| CLI entry points: `cellquorum`, `cq` | Implemented |
| Strict config validation | Implemented |
| Backend registry (Python / R / Rscript / GPU / RAPIDS) | Implemented |
| Execution planner + registry-driven executor | Implemented |
| Run bootstrapper | Implemented |
| Provenance artifacts | Implemented |
| Public Python API | Implemented |
| Pre-commit hooks | Implemented |
| Fail-loud data contracts (structural + semantic-tag + statistical) | Implemented |
| Strategy-based method registry (`AnalysisMethod` / `MethodDispatchStage`) | Implemented |
| R/Bioconductor method execution (`run_script` adapter) | Implemented |
| GPU-gated method execution (self-gating stages) | Implemented |
| GPU compute routing (rapids-singlecell / cupy, GPU-by-default) | Implemented |

### Analysis stages

| Stage | Methods | Status |
|---|---|---:|
| `ambient_correction` | SoupX (R) | Implemented |
| `qc` | MAD/fixed thresholds; Scrublet + scDblFinder doublet consensus; Tirosh cell-cycle | Implemented |
| `preprocessing` | PFlog1pPF / log1p-CP10k normalization (layer-tagged) — GPU via cupy | Implemented |
| `dimensionality` | PCA + scree plot + `n_pcs: auto` (knee) — GPU via rapids-singlecell | Implemented |
| `integration` | Harmony (CPU); scVI (GPU-gated) | Implemented |
| `clustering` | Leiden + neighbors (auto-routes onto the integration embedding) — GPU via rapids-singlecell | Implemented |
| `annotation` | marker-vote; consensus labeling; annotation diagnostics | Implemented |
| `feature_selection` | HVG / deviance | Implemented |
| `subclustering` | recursive Leiden with sc-SHC significance | Implemented |
| `population_identity` | population-level identity scoring | Implemented |
| `reference_mapping` | scArches (optional, atlas-agnostic) | Implemented |
| `integration_benchmark` | scIB-style integration metrics | Implemented |
| `embeddings` | UMAP + PHATE + PAGA (incl. PAGA-on-UMAP); generic feature-overlay (genes / module scores / cell-cycle / any obs column) with opt-in MAGIC | Implemented |
| `differential_expression` | donor-aware pseudobulk DE | Implemented |
| `differential_abundance` | paired arcsin-sqrt t-test; Milo; scCODA; sccomp | Implemented |
| `enrichment` | GSEA; ORA; GSVA; decoupler activity (CollecTRI / PROGENy) | Implemented |
| `enrichment_viz` | 8 figure types over GSEA / ORA / GSVA / activity | Implemented |
| `adjudication` | cross-method result adjudication | Implemented |
| `network_analysis`, `cell_cell_communication`, `trajectory` | — | Planned |

### Verification

- Test suite: **~1100 tests** (`python -m pytest`). GPU tests skip in a CPU-only
  environment and run in the `cellquorum-gpu` env.
- **End-to-end chain tests** run the full backbone
  (`qc → preprocessing → dimensionality → integration → clustering → annotation
  → embeddings → differential_expression → differential_abundance → enrichment
  → enrichment_viz`) and assert every stage's output threads to the final
  object — on both the CPU and GPU paths — so the pipeline is verified as a
  chain, not just stage-by-stage.
- A skippable real-data SoupX integration test confirms ambient correction runs
  end-to-end on Cell Ranger matrices when R + SoupX are present.

---

## Installation

Clone the repository:

```bash
git clone git@github.com:SecondBook5/CellQuorum.git
cd CellQuorum
```

Create a development environment:

```bash
mamba create -n cellquorum-dev python=3.12 -y
mamba activate cellquorum-dev
```

Install CellQuorum in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Install pre-commit hooks:

```bash
pre-commit install
```

Run tests:

```bash
pytest
```

Run repository checks:

```bash
pre-commit run --all-files
```

---

## Quick start

Show CLI help:

```bash
cellquorum
```

Show the installed version:

```bash
cellquorum --version
```

Build an execution plan:

```bash
cellquorum plan --config configs/config.yaml
```

Build an execution plan as JSON:

```bash
cellquorum plan --config configs/config.yaml --json
```

Initialize a CellQuorum run:

```bash
cellquorum run \
  --config configs/config.yaml \
  --output-dir runs/example_run
```

Initialize a run and print a JSON summary:

```bash
cellquorum run \
  --config configs/config.yaml \
  --output-dir runs/example_run \
  --json
```

`cellquorum run` initializes the execution frame, writes provenance artifacts, and executes every enabled stage in the registry-driven executor, threading the AnnData object from stage to stage.

---

## Python API

```python
from cellquorum import run_pipeline

result = run_pipeline(
    config="configs/config.yaml",
    output_dir="runs/example_run",
)

print(result.context.paths.root)
print(result.context.paths.provenance)
print(result.plan.enabled_stage_names())
```

`run_pipeline` accepts:

| Input type | Example |
|---|---|
| YAML config path | `"configs/config.yaml"` |
| validated config object | `CellQuorumConfig(...)` |
| dictionary config | `{"project": {"name": "my_project"}}` |

---

## Configuration

The default configuration is:

```text
configs/config.yaml
```

A minimal example:

```yaml
project:
  name: cellquorum_project
  organism: human
  species_id: 9606

paths:
  data_root: /mnt/e/CellQuorumData
  run_root: /mnt/e/CellQuorumRuns
  scratch_root: /mnt/e/CellQuorumScratch
  manifest: null
  output_dir: null

run:
  profile: standard
  run_id: null
  random_seed: 1337
  overwrite: false

compute:
  backend: auto
  prefer_gpu: true
  fallback_to_cpu: true
  n_jobs: 1

r:
  enabled: true
  preferred_backend: auto
  fallback_to_rscript: true
  rscript_path: Rscript
  timeout_seconds: 30

report:
  enabled: true
  html: true
  markdown: true
  pdf: false
  fail_on_report_error: false

stages:
  qc: true
  preprocessing: true
  integration: true
  annotation: true
  state_scoring: true
  discovery: true
  subclustering: true
  composition: true
  differential_expression: true
  molecular_inference: true
  cell_cell_communication: true
  network_analysis: true
```

Stage flags define whether a stage is allowed to run. Individual methods are still expected to pass data, metadata, backend, and statistical validity checks before execution.

---

## Output layout

A run initialized with:

```bash
cellquorum run --config configs/config.yaml --output-dir runs/example_run
```

creates:

```text
runs/example_run/
├── figures/
├── logs/
├── objects/
├── provenance/
│   ├── artifact_manifest.csv
│   ├── backend_status.csv
│   ├── backend_status.json
│   ├── pipeline_plan.json
│   ├── planner_warnings.json
│   ├── resolved_config.json
│   ├── run_metadata.json
│   └── stage_plan.csv
├── reports/
├── results/
└── scratch/
```

---

## Provenance

CellQuorum writes machine-readable provenance before analysis begins.

| File | Purpose |
|---|---|
| `resolved_config.json` | validated runtime configuration |
| `pipeline_plan.json` | enabled/disabled stage plan and backend summary |
| `stage_plan.csv` | tabular stage-level execution plan |
| `backend_status.json` | structured backend availability report |
| `backend_status.csv` | tabular backend availability report |
| `planner_warnings.json` | planner warnings |
| `run_metadata.json` | run identity, paths, profile, seed, and metadata |
| `artifact_manifest.csv` | index of generated artifacts |

---

## Architecture

```text
YAML config
   │
   ▼
Pydantic validation
   │
   ▼
Backend registry ───────► backend availability report
   │
   ▼
Pipeline planner ───────► stage plan
   │
   ▼
Run bootstrapper
   │
   ├── standardized directories
   ├── resolved config
   ├── pipeline plan
   ├── backend status
   ├── run metadata
   └── artifact manifest
   │
   ▼
Registry-driven executor
   │  for each planned + enabled stage:
   ├── resolve config sub-block
   ├── validate input data contract (fail-loud)
   ├── dispatch to config-selected method (skip-with-reason if inputs absent)
   ├── thread updated AnnData to the next stage
   └── record structured success / skip / failure
```

---

## Workflow spine

Stages marked ✅ are implemented and run today; ⏳ are planned slots the planner
already reserves.

```text
✅ ambient correction (SoupX)
✅ quality control (MAD/doublet-consensus/cell-cycle)
✅ preprocessing (normalization, layer-tagged)
✅ feature selection (HVG / deviance)
✅ dimensionality reduction (PCA + scree + auto n_pcs)
✅ integration (Harmony / scVI) + integration benchmark (scIB-style)
✅ clustering (Leiden) + recursive subclustering (sc-SHC)
✅ annotation (marker-vote / consensus / diagnostics)
✅ reference mapping (scArches, optional/atlas-agnostic)
✅ embeddings (UMAP + PHATE + PAGA, feature-overlay + opt-in MAGIC)
✅ differential expression (donor-aware pseudobulk)
✅ differential abundance (paired t-test / Milo / scCODA / sccomp)
✅ enrichment (GSEA / ORA / GSVA / decoupler activity)
✅ enrichment visualization (8 figure types)
⏳ gene-regulatory networks
⏳ cell-cell communication
⏳ trajectory / potency
⏳ report generation (auto methods text)
```

---

## Planned analysis modules

The full scientific and engineering plan is preserved in
[`docs/SCIENTIFIC_ENGINEERING_PLAN.md`](docs/SCIENTIFIC_ENGINEERING_PLAN.md).
The active implementation roadmap is tracked in
[`docs/ROADMAP.md`](docs/ROADMAP.md), which records current working
capabilities, the production QC output contract, highest-priority engineering
work, and the scientific module backlog.

CellQuorum is being built toward advanced scRNA-seq analysis layers, including:

| Layer | Examples |
|---|---|
| QC and preprocessing | MAD-based QC, mitochondrial/ribosomal/hemoglobin checks, doublet detection, ambient RNA audit |
| Annotation | marker-based annotation, reference-assisted annotation, consensus labeling |
| State scoring | cell cycle, senescence, stress, hypoxia, EMT, fibrosis, inflammation, immune polarization |
| Differential analysis | donor-aware pseudobulk DE, differential abundance, subcluster DE |
| Molecular inference | GSEA, pathway activity, TF activity, master regulators |
| Regulatory networks | GRN inference, VIPER/DoRothEA-style activity, regulator-target networks |
| Communication | ligand-receptor and ligand-target analysis |
| Protein/network analysis | STRINGdb, PPI networks, centrality, robustness, curvature |
| Optional advanced modeling | trajectory, transport, perturbation, lineage, generative models |

---

## Development workflow

Run tests:

```bash
pytest
```

Run pre-commit:

```bash
pre-commit run --all-files
```

Run CLI smoke checks:

```bash
cellquorum --version
cellquorum plan --config configs/config.yaml
cellquorum run --config configs/config.yaml --output-dir /tmp/cellquorum_test_run
```

Check repository status:

```bash
git status -sb
```

---

## Repository status

CellQuorum is in early development. The execution spine is implemented and tested.

Implemented:

- package scaffold
- strict config validation
- backend registry
- planner
- CLI `plan`
- CLI `run`
- public Python API
- standardized run directory layout
- provenance artifact writing
- smoke tests
- pre-commit hooks

Next milestones:

1. manifest schema and validation
2. stage lifecycle records
3. QC-safe single-cell core
4. preprocessing stage
5. report skeleton
6. method-gated advanced analysis modules

---

## License

MIT License.
