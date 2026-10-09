# scTenifoldKnk gene perturbations

Use `perturbation.method: sctenifoldknk` for cell-intrinsic virtual gene knockouts,
including non-TF genes. This is a separate method from CellOracle; its default
and existing CellOracle configuration remain unchanged.

The backend calls **scTenifoldpy 0.5.1**, the Python implementation of
[scTenifoldKnk](https://doi.org/10.1016/j.patter.2022.100434), using its actual
network inference, tensor decomposition, knockout and manifold-alignment workflow.
It does not approximate this method by setting an expression column to zero.

## Runtime

Install outside the main CellQuorum environment:

```sh
python3.12 -m venv /path/to/sctenifoldknk
/path/to/sctenifoldknk/bin/python -m pip install scTenifoldpy==0.5.1
```

Set `sctenifoldknk.python_executable` to that interpreter. If unset, the current
Python is used and missing dependencies produce an explicit skip. No runtime
installation or data download occurs during the stage. Planner and run probes
check the configured isolated interpreter (or current Python when unset).

## Configuration

```yaml
perturbation:
  enabled: true
  method: sctenifoldknk
  layer: counts_raw
  cluster_key: KC_publication_state
  condition_key: condition
  seed: 1337
  timeout_seconds: 10800  # per cohort
  n_top_targets: 15
  sctenifoldknk:
    python_executable: /path/to/sctenifoldknk/bin/python
    ko_genes: [IL33]       # multiple genes specify a JOINT gene knockout
    condition_label: Lymphedema
    donor_key: donor_id
    groups: [Basal, Spinous, Transitional differentiating]
    by_cluster: true
    pooled: true
    min_cells: 100
    max_cells: 2000        # 0 disables cell sampling
    max_genes: 2000
    n_nets: 10
    n_samp_cells: 500
    n_comp: 3
    network_quantile: 0.95
    tensor_rank: 3
    tensor_max_iter: 1000
    manifold_dim: 30
    ko_method: default     # zero KO-gene outgoing rows in the inferred WT network
    pathway_gmt: /path/to/curated_human_pathways.gmt
```

Omit both condition fields to analyze all input cells. Subtypes require an explicit
`cluster_key`; unknown requested groups fail rather than silently disappear.
For a pooled-only run set `by_cluster: false`. Input should already be restricted
to the relevant cell compartment. This method does not remove contaminating cells.

## Input preparation and audit

Requires unique gene symbols and cell IDs, finite nonnegative **integer counts**,
and nonzero cell libraries. Selects expressed variable genes by log-library-normalized
variance, retaining all requested KO genes within the gene cap. Counts, not those
normalized selection scores, are passed to scTenifoldKnk. A KO gene must meet
expression and variation thresholds in each cohort; unsupported cohorts are
explicitly skipped. The gene panel is selected independently within each cohort.
The input must already be quality-controlled: the native package's library-size
outlier filtering is disabled so selecting a gene panel does not silently redo
whole-transcriptome QC. Any remaining native filtering is recorded in fitted-cell
metadata and in the membership table's `knk_in_native_fit` column.

Sampling is seeded and proportional, preserving at least one cell per configured
donor stratum (and subtype for pooled runs). The exported membership tables record
exact cells, donors and cohort sizes. Each network is fitted separately. Multiple
donors contribute observations, but native gene p-values are **not donor-level
condition tests**. Network distances are not calibrated effect sizes across cohorts.

The runner checks that KO genes survive package processing and exports their fitted
outgoing edges. A cohort with no outgoing KO edges is marked uninformative, not
presented as evidence of no biological effect. A native execution error or timeout
fails the stage, rather than appearing as successful analysis. Old output files
cannot mask a failed fresh execution.

## Outputs and interpretation

Under `results/sctenifoldknk/`:

- `differential_regulation.csv`: native statistics and standardized `gene`,
  `distance`, `p`, `FDR`, group and KO labels. Native BH is per cohort.
- `cohort_audit.csv`: eligibility, used cells/genes/donors, fitted KO-edge count,
  package version and reasons for missing cohorts.
- Per-cohort native tables, outgoing KO edges, membership, configuration, metadata
  and execution logs, all registered as stage artifacts.
- Optional `pathway_enrichment.csv`: hypergeometric ORA of affected genes using
  **all tested non-KO genes** as background. The forced KO genes are excluded from
  both foreground and background. BH is reported per cohort and across all tested
  pathway/cohort combinations. Terms with zero overlap remain in the test family;
  uncovered terms do not appear as negative evidence.

Figures use CellQuorum's house style: affected-gene distance rankings and an optional
pathway FDR heatmap. No custom gene set is substituted automatically when no GMT
is supplied. Missing requested pathway coverage is recorded as a warning.

**Scores are unsigned differential network regulation**, not predicted expression
fold changes or proof of reduced inflammation/fibrosis. Subtype KO and pooled-KC KO
estimate intrinsic network consequences in those KCs; pooled is not a multicellular
simulation with receiver mast/T/fibroblast responses. For source-specific effects
in other cells, use a separately validated intercellular perturbation model.
