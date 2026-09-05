# QC: technical evidence → graded adjudication → analysis-specific eligibility

**Status: architecture FROZEN. Calibration OPEN.**

Date: 2026-09-03

This document is the authoritative design for CellQuorum's QC system. The conceptual
design is settled and must not be revised during implementation. Every numeric
threshold in this system is deliberately **unspecified** and will be calibrated against
the lymphedema QC figures. Implementers must leave them as named, configurable
parameters with no defaults chosen by guesswork.

---

## 0. Why this replaces the current system

The current QC stage computes a careful verdict and writes it to
`cellquorum_qc_keep`. Across the whole codebase, three places read that column:
`visualization/qc/diagnostics.py`, `visualization/qc/publication.py`, and
`stages/annotation/population_identity/stage.py:450`. Two of them draw pictures.

Nothing else reads it — not preprocessing, feature selection, PCA, integration,
clustering, annotation, DE, DA, trajectory, or CCC. So in `flag_no_drop` (the
production default in `configs/le_global.yaml`), QC-failing cells are fully included in
normalization, HVG selection, the embedding, the clustering, the labels, and every
statistical test. The only cells removed anywhere are consensus doublets.

**That is QC reporting without QC control.**

The root cause was not a QC-design failure. It was an **engine-contract failure**:
nothing prevented a developer from calling `model.fit(adata)` on everything. Adding six
prettier QC columns without fixing the contract would recreate the same problem. §4 is
therefore as load-bearing as the QC science.

## 1. Principles

These four sentences govern every decision below.

1. **Never destroy information unnecessarily.** The master object retains all barcodes.
2. **Never let questionable cells define the biological reference.** Fitting happens on
   core cells only, through the entire fitted chain.
3. **Never pretend one QC statistic is biological truth.** No single model's posterior
   is equivalent to a verdict about a cell.
4. **Never use a cell for an inference its evidence doesn't support.** Eligibility is
   per-analysis, not one global boolean.

The goal is not "keep all" and not "delete all". It is: **use every cell for every
inference it can defensibly support.**

## 2. Stage layout

Final ordering. Orders between 20 and 135 operate on `qc_state_initial` and the scope
contracts — that is what the initial state is *for*, not a weakness.

```
10   ambient correction / raw-droplet diagnostics
15   optional lightweight splice-QC extraction        [if BAM/loom available]
20   qc_evidence
       ├─ metrics
       ├─ evidence-family states
       ├─ model diagnostics
       ├─ evidence availability
       └─ qc_state_initial = core | borderline | quarantine
30+  preprocessing            FIT: core        TRANSFORM: core + borderline
     integration              FIT: core        TRANSFORM: core + borderline
     clustering               FIT: core only
90   core annotation          marker_vote / CellTypist / …
100  core subclustering
105  query_projection         borderline → frozen core manifold
       ├─ neighbour support
       ├─ query labels
       ├─ OOD score
       └─ mapping uncertainty
120  reference_mapping        atlas evidence
130  annotation_consensus     core + query-aware evidence integration
135  qc_finalization
       ├─ per-cell rescue
       ├─ qc_state_final = core | rescued | unresolved_borderline | quarantine
       ├─ fit/transform/inference eligibility
       ├─ QC bias decomposition
       └─ sensitivity universes
140  annotation diagnostics
145  biological / state adjudication   (consumes final QC validity)
150  population identity
     downstream inference
```

### Changes to existing stage orders

- **New:** `qc_evidence` (20, replaces `qc`), `qc_splice_metrics` (15, optional),
  `query_projection` (105), `qc_finalization` (135).
- **Moved:** the existing `adjudication` stage (currently 110) moves to **145**. It is a
  cluster/state evidence adjudicator whose evidence categories include *technical
  validity*; it must not adjudicate biological claims before technical validity is
  final. The flow is technical adjudication → biological adjudication.
- `qc_finalization` sits at 135 — after `reference_mapping` (120) and
  `annotation_consensus` (130) — because rescue uses atlas support as evidence. Placing
  it at 95 and mutating the result later was rejected: `qc_state_final` must be
  genuinely final.

## 3. Evidence model

### 3.1 Evidence families

Concordance is required **across families**, never across correlated metrics within one
family. Low UMI plus low gene count is one hit, not two.

| Family | Metrics |
|---|---|
| **CAPTURE / COMPLEXITY** | UMI, detected genes, top-gene concentration |
| **NUCLEAR–CYTOPLASMIC INTEGRITY** | intronic fraction, MALAT1 fraction, splice ratios |
| **METABOLIC / STRESS** | mitochondrial fraction, dissociation-stress program |
| **AMBIENT / BACKGROUND** | SoupX / CellBender evidence |
| **CELL-CALLING** | Cell Ranger call, EmptyDrops / future cell probability |
| **MULTIPLET** | Scrublet, scDblFinder |

**Multiplet is not a damage axis.** A doublet can be an excellent library that is simply
not one cell. Damage, doublet, and ambient evidence are carried separately:

```
qc_damage_probability
qc_doublet_probability
qc_ambient_burden
qc_cell_probability
```

Doublets are **not** physically deleted. They are graded
(`doublet_high_confidence` / `doublet_discordant` / `doublet_low_confidence`) and
excluded from relevant analyses via masks, so we can inspect whether high-RNA-content
keratinocytes are being disproportionately called.

### 3.2 Per-axis availability

Absent evidence must never read as evidence of health. Every axis carries a state:

| State | Meaning |
|---|---|
| `available_valid` | measured and trustworthy |
| `unavailable_input` | required input absent (no loom → no splice ratios) |
| `not_applicable` | meaningless for this assay (intronic fraction in snRNA) |
| `model_unstable` | measured, but the fitted model is weakly identified |
| `computation_failed` | should have worked, did not |

`computation_failed` must not silently degrade into the same behaviour as
`not_applicable`. The adjudication rules must state what happens when a required axis is
missing.

Each cell additionally carries:

```
qc_evidence_coverage    how many trustworthy families contributed
qc_confidence           certainty of the verdict given that coverage
```

A cell adjudicated on three families is not presented with the same certainty as one
adjudicated on six.

### 3.3 Directional rules

MAD is retained as a **transparent evidence generator**, demoted from decision engine.
Two-sided bounds on every metric is conceptually wrong — the tails mean different things:

| Metric | Low tail | High tail |
|---|---|---|
| genes | possible damage | possible doublet / large cell |
| UMI | possible capture failure | possible doublet / large cell |
| mt% | not concerning | possible damage |
| top-20 fraction | not damage | low-complexity dominance |

Tails feed **different evidence channels**.

### 3.4 Mitochondrial and stress evidence are context-sensitive

`stages/qc/mixture.py` implements a native miQC-style two-component regression of
mitochondrial percentage on library complexity, fit per sample by EM. Its rationale
stands: mt% is a mixture, not unimodal noise, so a location-scale estimator reports the
spread of the healthy mode and *tightens as a sample gets cleaner* — 2.0% for the
cleanest sample and 11.2% for the dirtiest on the skin atlas, which is backwards.

**Its posterior is evidence, never a `hard_fail` on its own.** A posterior is a statement
about a fitted distribution, not about a membrane. A legitimate low-complexity
population can fall into the low-quality component, which is exactly the risk for
keratinocyte differentiation states, mast cells, LECs, and lymphocytes.

The same applies to dissociation stress, and more strongly: FOS/JUN/HSP/EGR1 programs
are genuine biology in inflamed and lesional tissue. A fixed low weight still encodes a
questionable assumption. Both are **context-sensitive supporting evidence**:

```
stress abnormality alone            → never quarantine
stress + structural/capture damage  → strengthens damage evidence
```

A `hard_fail` requires either an essentially uninformative barcode (near-zero genes
*and* near-zero counts) or **concordant severe evidence across multiple independent
families**.

### 3.5 Sample-aware models with shrinkage

Neither one global model nor 18 fully independent per-sample models is right. A small or
unusually clean sample cannot robustly estimate a two-component mixture; a very poor
sample's internal model can redefine *poor = normal*.

Conceptually `q_is = μ_k + b_s + ε_is` (legitimate QC phenotype + sample shift +
cell-level abnormality). Full hierarchical Bayes is V2. V1:

```
fit per sample when stable
  ↓ shrink unstable fits toward cohort-level estimates
  ↓ record fit quality
  ↓ fall back conservatively
```

Per-sample fit diagnostics are mandatory: `mixture_converged`,
`component_separation`, `component_size`, `posterior_entropy`, `fit_stability`. When
two components are not convincingly supported, **do not force two components** — return
`model_unstable` and fall back to evidence-only adjudication. Automation must be able to
decline to make a strong call.

## 4. The scope contract — a framework invariant

Stage registration gains an explicit cell-scope policy. Three concepts, not two:

```
FIT         cells allowed to determine parameters / statistics
TRANSFORM   cells allowed to receive a representation / output
INFERENCE   cells allowed to contribute to scientific inference
```

A rescued keratinocyte may legitimately *receive* an scVI coordinate and a cell label
while being prohibited from *influencing* the model that produced either.

```
fit_scvi            false      transform_scvi        true
fit_clustering      false      receive_query_label   true
use_composition     true       use_de                conditional / false
```

A quarantined cell: fit nothing, optionally transform for plots, no scientific
inference.

### Enforcement

Under QC schema v2 there is **no implicit "all cells"**. Every stage must declare its
scope, including declaring it wide:

```
fit_scope = core
fit_scope = none
fit_scope = all,  reason = "per-cell independent transform"
```

A test enumerates the stage registry and **fails** when a relevant stage has no scope
declaration. This is the mechanism that stops `cellquorum_qc_keep` from happening
again; without it the masks decay back into decoration.

### 4.1 The cohort-derived-quantity rule

The invariant is *not* "ML models fit on core cells". It is:

> **Any cohort-derived quantity used to transform biological data must be estimated from
> the permitted fit population.**

That includes things nobody mentally classifies as models: normalization targets, gene
prevalence filters, HVG means/variances/dispersions, scaling means and SDs, PCA
loadings, batch-correction parameters, latent models, neighbourhood graphs, cluster
centroids.

Concretely for the default PFlog1pPF recipe, the proportional-fitting target is

```
T_PF = mean over CORE cells of L_i        NOT mean over ALL cells
```

then applied frozen to core, borderline, and (for QC visualization) quarantine. This is
exactly the fit/transform distinction, and it is easy to miss because it does not look
like a fitted model.

Provide this as a utility-level API — `fit_obs_mask` / `transform_obs_mask` — rather
than making every normalization implementation reinvent masking.

### 4.2 The whole fitted chain is protected

Damaged keratinocytes carry stress genes, mitochondrial genes, and immediate-early
genes. If they participate in HVG selection they change the biological manifold **even
if later excluded from PCA**. No leakage upstream:

```
CORE → normalization fit statistics → gene eligibility → HVG → scaling
     → PCA/scVI → neighbours → clusters
then BORDERLINE → transform / project
```

## 5. The query projection primitive

Build **once**, as an engine primitive. Do not reimplement neighbourhood logic
separately in QC rescue, annotation, reference mapping, and diagnostics.

`marker_vote` cannot solve this: it assigns labels by `clusters.map(assignments)`
(`marker_vote.py:98`), so an unclustered query cell has no path through it. CellTypist
is per-cell but the production config routes through `majority_voting: true`. Nothing
resembling `sc.tl.ingest` exists in the repo.

Takes: reference cells, query cells, frozen representation, reference labels.
Returns per query cell:

```
nearest_reference_distance      neighbor_label_probabilities
local_density                   neighbor_label_entropy
neighborhood_purity             neighbor_cluster_probabilities
OOD_score                       mapping_confidence
top_label                       top_label_probability
second_label_probability        margin
effective_neighbor_count
```

**Borderline cells never receive an ordinary cluster ID.** Not `leiden = "7"`. Instead
`query_nearest_cluster` and `query_cluster_probabilities` — that cell did not
participate in Leiden and the provenance distinction will matter later.

Serves three purposes: QC rescue (does the questionable cell land convincingly inside a
legitimate core population?), annotation (what identities do core neighbours support?),
and OOD detection (does it fit nowhere?).

### Reference immutability

For scVI/scArches query mapping, borderline cells may adapt against the frozen
reference but there must be **no joint retraining** that alters the core latent space.
The contract is explicitly tested:

```
reference model parameters before query mapping == after query mapping
```

PCA is simpler: `Z_borderline = (X_borderline − μ_core) · W_core`.

**Harmony cannot be the authoritative reference manifold** — it jointly corrects an
existing embedding and has no out-of-sample transform. It remains a
diagnostic/comparison representation. The authoritative manifold is scVI or PCA.

## 6. Rescue

Per-cell, deliberately. A minimum rescued-cluster size is **explicitly rejected** —
rare cells are precisely where this must help, and a population of eight real
keratinocytes must be rescuable.

Because there is no cluster-size averaging to suppress noise, multi-family concordance
is not garnish; it is the only thing standing in for the statistical power a cluster
requirement would have provided.

Rescue requires **positive biological support AND absence of severe technical
contradiction**:

```
Rescue_i = BiologicalSupport_i ∧ ¬SevereTechnicalContradiction_i
```

not `Rescue_i = kNN_i > 0.9`. Damaged cells can still map near keratinocytes.

- *Biological support*: high core-neighbourhood support, marker coherence, reference
  support.
- *Severe technical contradiction*: severe nuclear/cytoplasmic failure, overwhelming
  damage evidence, high-confidence heterotypic doublet, severe OOD.

Cross-donor recurrence of a rescued phenotype **raises** confidence; its absence does
**not** veto a rare population.

### States

```
qc_state_initial :  core | borderline | quarantine
qc_state_final   :  core | rescued | unresolved_borderline | quarantine
```

`rescued` is a post-reference conclusion and therefore only exists after stage 135. The
first pass must not pretend to know which questionable cells are recoverable.

## 7. Condition-blind inference, condition-aware audit

The QC model may know `sample_id`, technical batch, assay, and library. It must **not**
use disease condition to decide what counts as quality — otherwise it learns
"lymphedema-looking cells are normal for lymphedema", or the reverse.

Auditing afterwards is condition-aware: does `P(QC state | condition)` differ?

Condition-associated quality is **not automatically a FAIL**. Lesional tissue may
genuinely dissociate worse, contain more stressed or fragile cells, or have different
RNA complexity. Forcing equal removal across arms would hide real biology. The output is
a WARNING plus a **decomposition**:

```
QC CONDITION ASSOCIATION: WARNING
  Normal      quarantine = 4.8%
  Lymphedema  quarantine = 17.2%
  OR / CI = …

  excess QC burden in lymphedema:
     52% stress family
     21% nuclear / cytoplasmic
     18% capture
      9% other

  Downstream composition estimates may be sensitive to QC.
  Run inclusion sensitivity analysis.
```

The per-family decomposition matters because stress is the axis most likely to *cause*
the divergence, and "is this stress-driven?" is the first question to ask when it fires.

Report `P(QC state | cell type)`, `P(QC state | condition)`, and
`P(QC state | donor/sample)`.

## 8. Sensitivity universes

Always computed for composition and differential abundance (cheap — a recount):

```
core only
core + rescued
all non-quarantine
```

For DE, do not triple the whole contrast universe. Robustness is mandatory for
**declared primary contrasts**, optional for all. Report `cor(LFC_core,
LFC_core+rescued)`, sign concordance, top-hit overlap, and significance stability.

If a biological conclusion appears only under one QC definition, flag it as
QC-sensitive.

## 9. Raw droplets and splice metrics

`stages/ambient_correction/stage.py:102` already opens `raw_feature_bc_matrix.h5`
alongside the filtered matrix at order 10, so raw-vs-filtered comparison needs no new
inputs.

For V1 this is a **diagnostic recovery audit only** — those barcodes do not enter the
cell universe. Skin is exactly where ambient keratin transcripts are abundant, so a
low-count empty droplet containing KRT transcripts is *not* evidence that Cell Ranger
discarded a keratinocyte. A real candidate eventually requires: not called by Cell
Ranger, **and** EmptyDrops/CellBender cell probability, **and** marker program above
ambient expectation, **and** sufficient complexity.

`stages/trajectory/_velocyto.py:64` can generate looms from `possorted_genome_bam.bam`.
Do **not** move the trajectory velocyto machinery to order 20 — that couples QC to
trajectory and makes it expensive. Build a lightweight `qc_splice_metrics` (order 15)
that reuses an existing loom or generates only what is needed, with the trajectory stage
free to consume the resulting layers later.

Splice/intronic metrics are **evidence channels, not prerequisites**. CellQuorum must
remain fully functional on a filtered matrix alone.

## 10. Assay type

```yaml
qc:
  assay_type: auto     # → scrna_whole_cell | snrna
```

The engine cannot encode "high intronic fraction = damage" without knowing the assay;
in intact nuclei it is expected. Extendable later for CITE-seq etc.

## 11. Migration

Schema versioning, not a hard error — silent reinterpretation is unacceptable, but
breaking existing reproducible runs is also unacceptable for a workflow engine.

```yaml
qc:
  schema_version: 2
  policy: graded_adjudication
```

`schema_version: 1` (or absent) retains legacy `flag_no_drop` / `filter` / `both`
behaviour and emits a deprecation warning. `QCMode`, `should_filter()`, and
`fail_any_qc`-as-fate exist only under v1. All new configs are v2. The resolved config
is already recorded in `provenance/resolved_config.json`, so which policy ran is
captured.

## 12. Scope

**V1:** graded states; `fail_any_qc` no longer fate; directional evidence-only MAD;
miQC-style model as evidence with fit diagnostics; MALAT1 + stress immediately;
intronic/splice where inputs allow; separate doublet evidence; load-bearing scope
contracts in preprocessing/integration/statistics; core-only manifold fitting; query
projection and per-cell rescue; sample/donor/condition audits with family
decomposition; sensitivity universes.

**V1.5 / V2:** hierarchical sample model; CellBender/EmptyDrops raw-droplet recovery;
learned integrity classifier; synthetic corruption benchmarking; formal
analysis-specific recoverability.

## 13. What is frozen, and what the figures decide

**Frozen — do not revisit during implementation:** stage ordering; existence of
core-only fitting; requirement of query mapping; evidence-family structure;
fit/transform/inference contracts; missing-evidence semantics; condition-blind
inference with condition-aware audit; per-cell rescue; finalization after
reference/consensus.

**Open — the lymphedema QC figures decide:** calibration of every threshold; where the
initial core boundary belongs; what constitutes severe contradiction; how strongly each
evidence family behaves empirically; whether the miQC-like mixture is well identified
per sample; whether stress disproportionately drives LE flags; how many keratinocytes
become borderline; whether they map coherently back into core populations; how many can
defensibly be rescued; which sensitivity universes are scientifically stable.

Implementers: leave thresholds as named configurable parameters. Do not choose defaults
by guesswork.
