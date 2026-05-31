<div align="center">

# CellQuorum

### A reproducible single-cell RNA-seq workflow engine for publication-oriented analysis

CellQuorum provides a Python API, command-line interface, validated configuration system, backend-aware execution planning, standardized run outputs, and provenance tracking for advanced scRNA-seq workflows.

<br>

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20development-orange)
![Interface](https://img.shields.io/badge/interface-CLI%20%7C%20Python-informational)
![Workflow](https://img.shields.io/badge/workflow-single--cell%20RNA--seq-purple)

</div>

---

## Overview

CellQuorum is designed to make advanced single-cell RNA-seq analysis easier to run without losing reproducibility, auditability, or scientific discipline.

The current implementation provides the execution spine of the project:

- strict YAML/Pydantic configuration validation
- backend registry for Python, R/Rscript, GPU, and RAPIDS availability checks
- execution planner for enabled stages and backend status
- standardized run directory layout
- provenance artifact writing
- command-line interface
- public Python API
- pytest and pre-commit support

Full biological analysis stages are being added module by module, starting with manifest validation, stage lifecycle records, and QC-safe single-cell preprocessing.

---

## Current capabilities

| Capability | Status |
|---|---:|
| Installable Python package | Implemented |
| CLI entry points: `cellquorum`, `cq` | Implemented |
| Strict config validation | Implemented |
| Backend registry | Implemented |
| Execution planner | Implemented |
| Run bootstrapper | Implemented |
| Provenance artifacts | Implemented |
| Public Python API | Implemented |
| Pre-commit hooks | Implemented |
| Full scRNA-seq QC stage | In progress |
| Manifest validation | Planned next |
| Report generation | Planned |
| R/Bioconductor method execution | Planned |
| GPU/RAPIDS execution | Planned |

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

At the current stage, `cellquorum run` initializes the execution frame and writes provenance artifacts. It does not yet execute full scRNA-seq analysis stages.

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
```

---

## Planned workflow spine

```text
ingest
→ quality control
→ preprocessing
→ integration
→ annotation
→ state scoring
→ discovery
→ subclustering
→ composition
→ differential expression
→ molecular inference
→ communication analysis
→ network analysis
→ report generation
```

---

## Planned analysis modules

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
