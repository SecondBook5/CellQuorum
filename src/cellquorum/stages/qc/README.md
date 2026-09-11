# QC implementation

Start with `QCStage.run` in [stage.py](stage.py). The pipeline measures the input,
fits provisional lineages and the mitochondrial mixture, applies detection floors,
scores doublets, assigns graded eligibility, and writes its audit and artifacts.
Projection and finalization run later against the fitted biological reference.

## Read the workflow in this order

1. **Ambient correction**, when configured and supported by the inputs, runs in
   [`ambient_correction/stage.py`](../ambient_correction/stage.py) before QC.
   Its SoupX path uses raw and filtered droplet matrices per library. QC records
   upstream correction provenance; `qc.ambient` does not run a correction method.
   Method reference: [Young and Behjati, SoupX (2020)](https://doi.org/10.1093/gigascience/giaa151).
2. **Metrics** measure counts, detected genes, and feature-family fractions from
   one explicit expression source.
3. **Provisional lineages and mixture fitting** establish reference groups for
   judging unusual QC profiles. Provisional groups are not final biological clusters.
4. **Detection floors** physically remove barcodes and genes below the configured
   minimum counts or detection limits.
5. **Doublet detection** produces per-detector scores and consensus calls.
   Detectors receive the run seed and record it in metrics. An explicit
   `doublets.score_threshold` overrides native calls; without an override, native
   calls take precedence and a 0.5 score cutoff is used only when calls are absent.
   `doublets.expected_doublet_rate` is passed to Scrublet's expected rate and
   scDblFinder's `dbr` parameter. It is an expectation, not a quota of cells to
   flag. Earlier versions omitted it from the R call; reruns can therefore
   change scDblFinder calibration even when the configuration is unchanged.
   Detector outputs must have one probability score and at most one boolean call
   per cell. NaN scores remain unmeasured; infinite/out-of-range scores and
   malformed shapes fail validation. `doublet_measured_<method>` and the
   `measured_cells` metrics expose coverage, with warnings for partial scoring.
   A false `predicted_doublet` flag alone does not establish a measured singlet.
   The R bridge exports input cell IDs; Python checks complete, unique coverage
   and aligns results by ID. Unknown or missing class labels fail validation
   instead of becoming singlet calls.
   Per-sample detection requires complete, nonblank library IDs that remain
   distinct when serialized to R. An explicitly requested missing column fails;
   passing no sample key retains the standalone pooled-detection behavior.
   Consensus-called doublets lose fitting permission even when their robust-tail
   score is low. Optional physical removal is controlled by `doublets.remove`.
6. **Grading and eligibility** assign core/borderline/quarantine state and explicit
   masks for each analysis. Quarantined cells and probable doublets cannot fit the
   manifold or clustering. An empty fitting population stops computation; it never
   falls back to fitting excluded cells.
7. **Auditing and reporting** expose what ran, what could not be measured, which
   cells were excluded, and whether exclusions differ across populations or samples.

A barcode being isolated from the main population is not sufficient evidence to
remove it: it may be a rare cell type. Exclusion requires technical evidence; the
population audits identify suspicious or disproportionately excluded groups for review.

## Where changes belong

| Responsibility | Location |
| --- | --- |
| All QC, projection, and rescue configuration and field validation | [config.py](config.py) |
| Input validation and explicit `X` / layer / `raw.X` selection | [validation.py](validation.py) |
| Gene-family definitions | [features.py](features.py) |
| Canonical cell/gene metrics and matrix reductions | [metrics.py](metrics.py) |
| Absolute cell and gene detection floors | [floors.py](floors.py) |
| Measurement-to-severity conversion, evidence families, initial adjudication | [evidence.py](evidence.py) |
| Provisional cell identity and reference-group fallback | [lineage.py](lineage.py) |
| Mitochondrial mixture estimation and diagnostics | [mixture.py](mixture.py) |
| Upstream SampleQC joint assessment | [sampleqc.py](sampleqc.py) |
| Doublet detector adapters and consensus | [doublets.py](doublets.py) |
| Analysis-specific permissions | [eligibility.py](eligibility.py) |
| Reference projection primitive and pipeline stage | [projection.py](projection.py) |
| Final rescue decisions and stage | [finalization.py](finalization.py) |
| AnnData annotation, figure inputs, and stage summaries | [reporting.py](reporting.py) |
| Artifact serialization and output orchestration | [artifacts.py](artifacts.py) |
| Population/design audits and consistency checks | [attrition.py](attrition.py), [archetypes.py](archetypes.py), [selfcheck.py](selfcheck.py) |

`_context.py`, `_types.py`, and `_errors.py` contain shared plumbing, not alternate
QC algorithms. Figure rendering lives in `cellquorum.visualization.qc` and consumes
decisions; it must not fit another QC model.

## Contracts

- Resolve the count source explicitly. A missing requested layer is an error.
  Metrics, provisional lineages, gene evidence, and doublet adapters use that source.
  `raw.X` has its own gene axis, which can differ from `adata.var_names`.
- Compute metrics once. `floors_from_metrics` consumes those fresh tables; the
  standalone `apply_floors` entry point delegates to the same decision function.
- Gene-fraction numerators use the pre-filter expression source that produced their
  denominators. Sum every matching feature, including repeated gene names.
- Count correlated metrics once per evidence family. Keep missing evidence distinct
  from zero severity. Doublet evidence is separate from damage evidence.
- Preserve explicit all-false fitting masks. Missing or malformed masks on an object
  with graded QC state are errors, not permission to include every cell.
- Project queries onto a fixed reference. Compare the same number of neighbors for
  query and leave-one-out reference distances. Automatic representation selection
  uses scVI or PCA; an explicitly requested input must exist. Re-running projection
  replaces its columns.
- Audit against the full input population, including cells physically removed by
  configured doublet removal. Gene-floor decisions are based on detection in the
  original input population, not an iterative cell/gene pruning procedure.

## Scientific interpretation and validation

Robust-tail severity is `z / (z + half_severity_z)`, a heuristic transformation;
0.5 does **not** establish a 50% probability of damage. The mitochondrial model
retains its own posterior, inspired by the joint mitochondrial/complexity model in
[Hippen et al., miQC (2021)](https://doi.org/10.1371/journal.pcbi.1009290).
The shared numeric range does not make all evidence axes probabilistically equivalent.

The internal name `severity` means a **QC concern score**. With the default robust
scaling, deviations of 0, 3, and 6 map to scores of 0, 0.5, and approximately 0.67.
The mitochondrial mixture posterior is a separate type of evidence. These values
are not interchangeable measurements of the probability that a cell is damaged.

This implementation targets single-cell and single-nucleus RNA count matrices.
It does not automatically establish assay identity, species-specific feature names,
or whether a supplied expression matrix is corrected for ambient RNA. A filtered
matrix alone cannot supply the raw-droplet inputs required by this SoupX workflow.
Single-nucleus data require the nuclear-integrity axis to be configured appropriately.

The shipped severity and rescue thresholds remain uncalibrated defaults. Synthetic
damage and rare-population tests establish specific regression guarantees, not
general sensitivity or specificity on independent tissues. A publication-level
accuracy claim requires held-out data with independent quality evidence, explicit
false-removal and damaged-cell retention measurements, and sensitivity analyses by
sample, population, and assay. Do not tune thresholds and report validation on the
same cohort.

Run the QC regression suite from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_qc*.py \
  -m 'not gpu and not r and not slow and not integration'
```

The conceptual evidence and eligibility design is described in
[`docs/design/qc-graded-adjudication.md`](../../../../docs/design/qc-graded-adjudication.md).
Historical migration sections there are not a map of current Python modules.

## Consolidated imports

Use `cellquorum.stages.qc` for its existing public metric/floor/stage API. Internal
imports have these single locations; the former files were removed:

| Former module | Current owner |
| --- | --- |
| `config_validators`, `finalization_config`, `query_projection_config` | `config` |
| `producers` | `evidence` |
| `query_projection_stage` | `projection` |
| `_annotate`, `_report` | `reporting` |

## SampleQC joint assessment

The existing `mixture.py` implements a miQC-style two-component regression in
Python. Optional [sampleqc.py](sampleqc.py) runs the actual upstream R SampleQC
package through the shared Rscript backend. It exchanges only the QC metric table,
not the expression matrix.

```yaml
qc:
  sampleqc:
    enabled: true
    sample_key: sample_id
    n_components: 2
    alpha: 0.01
```

`n_components` is required when enabled; 2 above is an example, not an automatically
selected value. Inspect the distributions and assess sensitivity to component count.
This adapter fits one joint group of compatible samples with sample-specific shifts.
It does not automatically group incompatible tissues or assays. Each sample needs
25 usable cells by default. Smaller samples and invalid metric rows remain explicitly
unscored. For the declared joint group, the adapter constructs SampleQC's input
object directly and calls its upstream fitter. Sample-distance clustering and
sample embeddings are not computed; this avoids upstream plotting failures for
one or three-to-five samples and unnecessary pairwise sample computations.
The input-object contract is tested against the pinned upstream version. Non-degenerate counts, detected genes, and mitochondrial measurements
are required. Assays without informative mitochondrial measurements need another
feature specification, which this initial adapter does not implement.

Install SampleQC in the R library used by the configured Rscript backend. The
upstream source inspected for this integration is version 0.6.6, commit
`adf0d97fa6e06a617e85fc4d95a8441175b0788e`:

```r
remotes::install_github("wmacnair/SampleQC@adf0d97fa6e06a617e85fc4d95a8441175b0788e")
```

Outputs are consolidated into the existing cell metrics CSV and AnnData `obs`:
`sampleqc_distance` (minimum squared Mahalanobis distance), `sampleqc_pvalue`
(approximate chi-squared upper-tail value), `sampleqc_component` (nearest QC
component, not a biological annotation), `sampleqc_outlier`, and `sampleqc_status`.
Unscored rows have NaN distances/p-values and component -1; their false outlier flag
must be interpreted together with status. Package version, parameters, scoring
counts, and backend diagnostics are recorded in the QC summary and
`uns['cellquorum']['sampleqc']`.

SampleQC is assessment-only: its flags do not change grading, eligibility, or
physical retention. Its p-value is not a probability of damage. Counts, complexity,
and mitochondrial evidence overlap with existing axes, so treating this as another
independent vote would double-count evidence. Benchmark rare-population retention
and damage detection before introducing a decision policy.

Method: [Macnair and Robinson, SampleQC (2023)](https://doi.org/10.1186/s13059-023-02859-3).
