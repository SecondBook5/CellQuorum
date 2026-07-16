# CellQuorum roadmap

This file records the working plan for CellQuorum so the project direction is
not dependent on chat history.

The full finalized scientific and engineering plan is preserved in
[`docs/SCIENTIFIC_ENGINEERING_PLAN.md`](SCIENTIFIC_ENGINEERING_PLAN.md). This
roadmap is the shorter implementation tracker derived from that plan.

CellQuorum is being built as a Python-native, GPU-capable scientific framework
for single-cell RNA-seq systems biology. The execution engine is one feature of
that framework: it should make analyses reproducible, resumable, inspectable,
and easy to run, but the scientific identity is the evidence model, diagnostic
gates, adjudication logic, and reusable analytical APIs.

## Current working capability

As of the KC production run, the repository can run a real CellQuorum workflow
through:

- strict config validation and execution planning;
- backend probing and provenance output;
- QC metrics, thresholds, decisions, h5ad output, audit figures, and
  publication-style QC panels;
- preprocessing / normalization with CellQuorum-tagged layers;
- PCA dimensionality reduction with scree output and automatic PC selection;
- Harmony integration;
- Leiden clustering;
- annotation stage infrastructure, with marker-vote available when configured;
- adjudication stage infrastructure and basic evidence output;
- scArches reference mapping with multi-seed checkpoint/resume support;
- annotation-diagnostic entropy fallback from transferred probabilities;
- generic population/state identity evidence output that uses atlas/reference
  labels when available and otherwise falls back to annotation labels or native
  clusters;
- integration benchmarking with scIB-style metrics, including iLISI/cLISI-style
  outputs where configured.

The KC production configuration is:

- `configs/le_kc.yaml`

The most recent KC production outputs are under:

- `runs/kc_production/`

## QC output contract

Every QC run with figures enabled should emit both:

1. standard audit plots in `results/qc/`;
2. publication-style QC panels in `results/qc/publication/`.

The publication QC panels are wired into the normal QC artifact writer, not run
as a separate manual method. They include:

- per-sample mito/ribo/hemoglobin box panels;
- doublet-score ECDF;
- QC UMAP overlay;
- PCA scree;
- detected genes and UMI count panels;
- MAD outlier threshold panel;
- UMI-vs-detected-gene mitochondrial-gradient panels;
- condition-level QC violin/statistics panel;
- cells-per-sample bar chart.

Explicit `qc.outputs.write_figures: false` remains the escape hatch for tests or
no-figure runs.

## Highest-priority engineering work

These are the next structural fixes because they support the whole project
rather than only one analysis.

1. Stage result lifecycle

   `StageResult` and `StageExecutionRecord` now carry explicit lifecycle fields
   for status, skip reason, method version, backend, device, fingerprints, and
   checkpoints. The remaining work is making every production stage populate
   those fields consistently instead of leaving them optional.

   Immediate target:

   - stable input fingerprints for every stage;
   - stable output fingerprints for every stage that writes reusable artifacts;
   - method/version labels for every registered method;
   - backend/device labels for GPU-, R-, and external-tool-backed methods;
   - consistent skip reasons for disabled or ineligible stages.

2. Resume/checkpoint model

   Move toward durable stage outputs and fingerprints so completed stages can be
   skipped safely on rerun. The immediate goal is not a full workflow engine; it
   is reliable reruns for long analyses.

   Requirements:

   - stable cache keys from config + input fingerprints;
   - stage-level completion markers in `provenance/stages/<stage>/completion.json`;
   - clear invalidation when config/input changes;
   - no silent reuse when required artifacts are missing;
   - per-seed checkpointing for long stochastic methods.

3. Fail-closed statistical layer guards

   Statistical methods should require explicit non-imputed layer tags. Untagged
   matrices should not be treated as safe for statistics.

4. Design/confounding validation

   `DesignConfig` should validate whether the requested design is estimable:

   - condition-by-batch contingency checks;
   - donor/sample replication checks;
   - rank deficiency checks for model matrices;
   - warnings or failures before DE/DA/modeling stages run.

5. Generic configuration examples

   Add user-facing examples that show the same engine applied to a generic
   dataset, not only the KC dataset.

   Needed examples:

   - minimal report-only QC + preprocessing;
   - production QC + integration + clustering;
   - reference mapping against an atlas;
   - benchmarked integration;
   - donor-aware downstream analysis once implemented.

## Highest-priority scientific work

The project should not grow sideways by adding every possible method before the
core scientific spine exists. Prioritize modules that make CellQuorum more than
a wrapper.

1. Evidence adjudication

   Build a real population/state adjudicator that separates technical clusters
   from biological states.

   Evidence categories:

   - technical validity;
   - donor replication;
   - sample balance;
   - cluster stability;
   - molecular marker coherence;
   - reference support;
   - annotation confidence;
   - doublet / stress / cell-cycle contamination;
   - discreteness versus continuum evidence.

   First implementation can be threshold-based and imperfect. It should emit
   explicit PASS/WARN/FAIL/SKIP decisions and a table explaining each verdict.

2. Donor replication and held-out generalization

   Add tests that ask whether a cluster/state survives donor/sample resampling
   instead of only appearing in one library.

3. Annotation QC and taxonomy diagnostics

   Strengthen annotation diagnostics around:

   - entropy/uncertainty;
   - reference agreement;
   - cluster-level label purity;
   - marker support;
   - donor balance;
   - suspicious technical labels.

   Population identity must remain general. Atlas labels are treated as external
   evidence when present; when no atlas exists, CellQuorum should build
   dataset-native population candidates from clusters, annotations, donor/sample
   replication, QC, and marker evidence.

4. Differential analysis

   Implement donor-aware pseudobulk DE and differential abundance before adding
   more exotic downstream methods.

5. State scoring and program analysis

   Add robust scoring for common biological programs:

   - cell cycle;
   - stress;
   - hypoxia;
   - interferon/inflammation;
   - senescence;
   - fibrosis/ECM;
   - lineage- or tissue-specific programs.

6. Molecular inference

   Add pathway and TF activity modules after DE/state scoring are stable.

7. Communication and network layers

   Ligand-receptor, ligand-target, GRN, and multicellular program modules are
   planned, but they should follow the evidence/adjudication spine rather than
   become unvalidated package wrappers.

## Optional / later modules

These remain in scope but should not block the core framework:

- PHATE and other visualization manifolds;
- MAGIC/imputation, only with strict layer tags and statistics guards;
- trajectory;
- RNA velocity;
- optimal transport;
- GRN rewiring;
- network topology/curvature;
- perturbation and generative models.

These should be gated by explicit eligibility checks and should skip cleanly when
the dataset does not support them.

## Current immediate next steps

1. Populate lifecycle fingerprints/version/backend/device fields across the
   production stages.
2. Add stage-level resume that reads completion sidecars and skips completed
   stages only when fingerprints and required artifacts match.
3. Start the LE-KC-style biological result modules as first-class output
   directories with `plots/`, `tables/`, evidence reports, and audit metadata.
4. Make layer-tag statistical guards fail closed.
5. Add design/confounding validation.
6. Start the real adjudicator with simple explicit evidence rules.

## Principle for future additions

New methods should be added only when they satisfy this contract:

```text
validate input semantics
→ check scientific eligibility
→ run method
→ emit quantitative diagnostics
→ apply evidence gate
→ write durable artifacts
→ record provenance
```

If a method cannot yet meet that contract, it belongs in an exploratory notebook
or adapter sandbox, not the core CellQuorum workflow.
