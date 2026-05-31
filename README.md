# CellQuorum

CellQuorum is a publication-grade single-cell RNA-seq workflow engine with Python, R/Bioconductor, and GPU backends.

The goal is to make advanced scRNA-seq analysis easy to run without sacrificing rigor. CellQuorum is designed around staged execution, validated configuration, backend isolation, structured artifacts, provenance tracking, and final analysis reports.

## Core Idea

CellQuorum is not intended to be a loose collection of single-cell methods. It is a reusable analysis engine organized around biological and statistical questions:

- What cells and states are present?
- Which populations are altered across conditions?
- Which transcriptional programs, pathways, and regulators are active?
- Which subclusters or rare populations deserve deeper analysis?
- Which ligand-receptor, regulatory, protein, and network structures explain the observed biology?
- Which analyses are statistically defensible given the available samples and metadata?

## Planned Interface

```bash
cellquorum run \
  --project examples/minimal_scrna/project.yaml \
  --manifest examples/minimal_scrna/manifest.csv \
  --output-dir runs/minimal_scrna
```

## Planned Python API

```python
from cellquorum import run_pipeline

run_pipeline(
    project="examples/minimal_scrna/project.yaml",
    manifest="examples/minimal_scrna/manifest.csv",
    output_dir="runs/minimal_scrna",
)
```

## Design Principles

1. Python is the primary orchestration layer.
2. R/Bioconductor methods are first-class backends.
3. GPU/RAPIDS methods are first-class optional backends.
4. Every stage emits machine-readable tables, figures, reports, warnings, and provenance.
5. Advanced methods are gated by data availability, metadata, and biological question.
6. The default workflow should be easy to run and scientifically conservative.
7. Project-specific biology should come from configuration, not hard-coded package logic.
8. The final output should include a structured analysis report, not only raw files.

## Planned Workflow Spine

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

## Advanced Analysis Layers

CellQuorum is planned to support advanced optional modules including:

* state scoring for cell cycle, senescence, stress, hypoxia, EMT, fibrosis, inflammation, and immune polarization
* automatic discovery of subclustering candidates and rare signature-enriched populations
* donor-aware pseudobulk differential expression
* differential abundance and compositional analysis
* GSEA, pathway activity, and transcription factor activity
* GRN inference and master regulator analysis
* STRINGdb and protein association networks
* ligand-receptor and ligand-target communication analysis
* network centrality, robustness, and curvature analysis
* optional trajectory, transport, perturbation, lineage, and generative modeling backends

## Repository Status

CellQuorum is currently in early development. The first milestone is the execution spine:

* package scaffold
* configuration system
* stage contract
* pipeline context
* artifact manager
* backend registry
* planner
* provenance tracking
* CLI
* smoke tests

