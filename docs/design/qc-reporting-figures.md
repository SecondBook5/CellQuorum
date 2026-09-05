# QC V2 — reporting and calibration figures

Date: 2026-09-03

Companion to `2026-09-03-qc-graded-adjudication-design.md`, which is **FROZEN**. Do not
change the architecture and do not invent numeric thresholds. This document specifies
the diagnostic and publication outputs needed to calibrate and audit that architecture,
with particular attention to the lymphedema dataset and keratinocyte retention.

Motivating observation: under the legacy system, loss is not uniform across populations
— keratinocytes ~20%, SMC/pericytes ~47%, some tiny populations effectively 100%. A
table that only reports "removed" is no longer adequate. **Every figure here exists to
answer *why* a cell was lost, decomposed by evidence family.**

## Style and placement

- Reuse the shared style in `src/cellquorum/visualization/figstyle.py`. Do **not**
  duplicate style code. Available and directly relevant: `NORMAL_BLUE` (`#1B4F8A`),
  `LE_RED` (`#C41E3A`), `QC_FAIL_COLOR` (`#7E858B`, the neutral grey for quarantine),
  `DOUBLET_COLOR`, `condition_palette()`, `distinct_palette()`, `set_style()`,
  `save_figure()` (PNG + PDF companion), `atomic_savefig()`, `FIGSIZE`, `FONTSIZE`.
- New module: `src/cellquorum/visualization/qc/graded.py`. `publication.py` is built
  around legacy `cellquorum_qc_keep` / threshold semantics and must not be retrofitted.
- **Ordering is always paired, Normal first then Lymphedema, within donor.**
- Early evidence figures are callable from `qc_evidence` (order 20). Anything requiring
  cell identities, query support, rescue, or `qc_state_final` belongs to
  `qc_finalization` (order 135) — never order 20.

## Requirements

### 1. Two summary tables, not one

**1a. Sample/donor outcome table.** Rows grouped by donor, Normal first. Columns: input
cells; initial core n/%; initial borderline n/%; initial quarantine n/%; rescued n and %
*of initial borderline*; unresolved-borderline n/%; final quarantine n/%; final
eligible-for-composition n/%; median evidence coverage; median confidence; median UMI;
median genes; median mitochondrial fraction. Full CSV plus rendered PDF/PNG/SVG.

**1b. QC reason decomposition table.** Two versions: by sample, and by final cell type.
Columns per evidence family from the frozen spec — capture/complexity, nuclear–cytoplasmic
integrity, metabolic/stress, ambient/background, cell-calling, multiplet — each showing
`n flagged` and `% flagged`. Plus `unique-family-only`, `>=2 independent evidence
families`, final quarantine, rescued, unresolved borderline.

- If the finalizer explicitly emits a `primary_driver`, show mutually exclusive
  primary-driver counts.
- **The visualization layer must never infer a primary cause from overlapping flags.**
- Evidence-family columns overlap and **must be labelled non-additive**.
- Rendered cell-type table: major populations plus explicitly protected populations
  (keratinocytes, LECs, mast cells, endothelial, …). Machine-readable CSV: **every** cell
  type including tiny ones.
- **Always show the denominator.** `1/1 = 100%` is not `5,000/10,000 = 50%`.

### 2. Evidence-family UpSet plot

Most common evidence-family combinations and their cell counts. Cohort-wide,
condition-stratified, and optionally cell-type-filtered views. **Correlated metrics
inside one family count as a single family hit** — capture + complexity is one hit, not
several. This is what answers "what caused QC concern?" without pretending overlapping
rules are independent.

### 3. Paired raincloud / split-violin diagnostics per patient

Replaces the simple per-patient boxplots. Per patient: **Normal left, Lymphedema right**.
Each side carries a half-violin/density, compact box/IQR, median, whiskers, and
deterministically subsampled individual cells where useful.

Separate modular figures for: total UMI, detected genes, mitochondrial fraction,
ribosomal fraction, top-gene concentration, MALAT1 fraction, dissociation-stress score,
intronic/splice metrics when available, doublet scores. Log axes for UMI/genes where
appropriate.

- Never plot a fake value for an unavailable axis. Mark unavailable / not-applicable
  explicitly.
- These are **descriptive**. Do **not** compute cell-level p-values between Normal and
  LE — cells are not biological replicates. Any inferential comparison must operate at
  the paired donor/sample level.

### 4. The same distribution framework by cell type

One of the most important diagnostics for this dataset. Per major cell type, Normal left
and LE right for the principal QC metrics. **Overlay donor-level medians** so a shift
driven by one donor is distinguishable from a systematic one. Configurable
`cell_types=` so keratinocytes, LECs, mast cells can be rendered explicitly.

Must answer: are keratinocytes legitimately shifted in complexity? Is mitochondrial
burden higher *within* KCs rather than globally? Is one evidence family
disproportionately flagging KCs?

### 5. UMI × genes joint plot, upgraded to first-class

Keep the visual idea of `qc_joint_density` (currently `panels.py:2586`). X = total UMI,
Y = detected genes, log or log-spaced axes, all cells rasterized, marginal
distributions. **Shade** rather than merely dash regions for explicitly configured
diagnostic bounds.

At least two versions: coloured by mitochondrial fraction (shared sequential colormap,
or configurable magma/viridis), and coloured by QC damage/evidence score where that
quantity exists. Final-state overlays render quarantine cells in `QC_FAIL_COLOR` grey
rather than letting them dominate a continuous colormap. Robust colour limits, with the
limits **reported in metadata** rather than silently clipping.

### 6. Final-state-by-cell-type retention figure

Horizontal 100% stacked bars, one cell type per row: `core`, `rescued`,
`unresolved_borderline`, `quarantine`. Sort primarily by quarantine + unresolved
fraction; optionally preserve taxonomy order. Annotate input n.

This must make "20% of KCs were removed" impossible to miss, and must reveal whether the
new architecture converts formerly-removed KCs into `rescued`.

### 7. Sample × evidence-family heatmap

Rows = samples ordered `P1 Normal, P1 LE, P2 Normal, P2 LE, …`. Columns = evidence
families. Value = % of cells flagged by that family. Second heatmap for final quarantine
fraction attributable to each explicit primary driver, if primary drivers exist.

Reveals sample-specific technical problems and whether LE samples systematically show
more stress or capture loss.

### 8. Condition-bias audit figure

Paired donors, Normal visually left. Donor-connected points for initial borderline %,
initial quarantine %, final quarantine %, and rescue rate. Family-specific versions or
small multiples, so an LE excess can be identified as primarily stress-associated rather
than nuclear or capture-associated.

**This is an audit. A condition difference is a warning, not automatically a failure.**

### 9. Rescue diagnostic figure (`qc_finalization`)

Initial-borderline cells only. Technical evidence severity against query/reference
support. Useful axes: core-neighbourhood support, query mapping confidence / OOD score,
marker + reference support, technical contradiction. Colour by final state (`rescued`,
`unresolved_borderline`, `quarantine`), with optional highlighting of a requested cell
type. Makes the rescue decision inspectable rather than a hidden rule.

### 10. UMAP diagnostics adapted to V2 semantics

Do **not** colour cells as `keep/fail`. Produce views for initial QC state, final QC
state, damage evidence, doublet evidence, ambient evidence, and mapping/OOD support where
available. **Borderline cells are query-projected and must never be represented as
though they participated in clustering.**

### 11. Individual panels plus one compact QA sheet

Individual files are the canonical outputs. Additionally one compact multi-panel
`qc_v2_calibration_overview` for quick inspection. Do not cram every metric into one
giant figure.

### 12. Implementation requirements

- Visualizations **consume** QC evidence/state columns and never recompute or reinterpret
  QC decisions.
- No plotting function may silently substitute `0` for unavailable evidence. Support all
  five frozen availability states: `available_valid`, `unavailable_input`,
  `not_applicable`, `model_unstable`, `computation_failed`.
- Deterministic sampling for dense scatter/jitter.
- Rasterize dense point layers; keep text, axes, and vector elements editable in
  PDF/SVG.
- Export PNG/PDF/SVG through existing CellQuorum artifact infrastructure.
- Unit tests for: ordering, missing metrics, missing modalities, denominators,
  overlapping-family counts, output naming, deterministic rendering inputs.
- **Do not restore hard-coded visual defaults** such as fixed mitochondrial or
  detected-gene thresholds when V2 has no such configured bound. Legacy V1 figures may
  keep legacy behaviour; V2 plots display only thresholds and evidence bounds actually
  emitted by V2.

## Primary calibration target

Trace every keratinocyte: input → initial QC evidence → initial state → query support →
final state, and quantify which evidence family would otherwise have caused its loss.
The same machinery must work with **no keratinocyte-specific logic** on arbitrary
datasets.

## Sequencing — PROPOSED, needs a decision

The stated priority order is: reason-decomposition table → evidence UpSet → patient
rainclouds → cell-type rainclouds → UMI×genes → final-state retention bars →
condition-bias audit.

Items 1b, 2, 6, and 8 consume **flags and states**, which exist only once per-family
thresholds have been chosen. But the figures are the instrument for choosing them, and
the frozen spec §13 forbids guessing defaults. So those four cannot come first without
either inventing thresholds or rendering empty panels.

Proposed split:

**Phase 1 — calibration inputs (threshold-free).** Extend `qc_evidence` extraction with
MALAT1 fraction and a dissociation-stress score (neither exists today anywhere in
`stages/qc/`), plus **continuous** per-family evidence values and their availability
states. No thresholds, no flags, no state assignment. Then build: patient rainclouds
(§3), cell-type rainclouds (§4), UMI×genes coloured by continuous mito and continuous
damage evidence (§5), and a sample × family heatmap keyed on **median continuous
evidence** rather than % flagged (§7 variant). These are what you read to decide where
core ends.

Cell-type figures in Phase 1 can run against an already-annotated legacy run
(`runs/kc_production/`), so cell-type calibration is not blocked on the full V2 pipeline.

**Phase 2 — audit outputs (thresholds now chosen).** States exist, so: reason
decomposition (§1b), UpSet (§2), summary table (§1a), final-state retention (§6),
sample × family flagged heatmap (§7), condition-bias audit (§8), rescue diagnostic (§9),
UMAP views (§10), QA sheet (§11).

## Calibration evidence from `runs/le_global_clean` (measured 2026-09-03)

Computed from `results/qc/{cell_metrics,cell_decisions,cell_labels}.csv`, 195,347 cells.
**No new pipeline code was needed** — pre-filter metrics, per-rule decisions, and
cell-type labels are all already on disk. Phase 1 calibration is unblocked today.

### Only three rules were active

`thresholds.csv` contains exactly: `fixed_min_genes_per_cell ≥ 200`,
`fixed_max_mito_percent ≤ 20`, `fixed_min_cells_per_gene ≥ 3`. **No MAD rows.**

### The fixed mito ceiling accounts for essentially all removal

Of the cells removed, the share explained by each rule:

| Cell type | n removed | high-mito only | low-genes only | both |
|---|---|---|---|---|
| Macrophages | 918 | 76.6% | 0% | 23.4% |
| Keratinocytes | 691 | **100%** | 0% | 0% |
| Fibroblasts | 417 | 95.2% | 4.6% | 0.2% |
| Mast | 388 | **100%** | 0% | 0% |
| T/NK | 142 | 100% | 0% | 0% |
| Pericyte/SMC | 115 | 100% | 0% | 0% |
| Melanocytes | 84 | 100% | 0% | 0% |
| LEC | 22 | 100% | 0% | 0% |
| Plasma | 9 | 0% | **100%** | 0% |

The `min_genes ≥ 200` floor removes almost nothing. **This directly validates the
`mixture.py` effort: the single rule destroying these populations is exactly the fixed
mito ceiling that miQC replaces.**

### Removal rates in this run

Keratinocytes 7.35%, Macrophages 6.47%, Melanocytes 5.15%, Mast 2.03%, Plasma 1.70%,
Pericyte/SMC 1.11%, LEC 1.00%, all others <0.6%.

Keratinocytes are the most-removed population in absolute terms. Note that
higher attrition figures quoted elsewhere (~20% KC, ~47% SMC/pericyte) come from
**different runs** and are not comparable to these numbers; do not treat them as targets
or as evidence about this configuration.

### Two qualitatively different removal signatures, already separable

| | mito p50 | mito p90 | mito max | median genes (removed) | median genes (kept) |
|---|---|---|---|---|---|
| **Macrophages** | **39.0%** | 78.5% | 95.5% | **380** | 2,248 |
| **Keratinocytes** | 25.5% | 41.9% | 68.4% | 1,151 | 2,836 |
| **Mast** | 22.8% | 29.8% | 49.2% | 416 | **744** |
| LEC | 26.6% | 35.7% | 37.5% | 802 | 2,438 |

- Removed **macrophages** show concordant severe failure across capture *and* metabolic
  families — median 39% mito with 380 genes. These are debris and should still quarantine.
- Removed **keratinocytes** show moderate single-family elevation: 25.5% median mito,
  1,151 genes, and **0.00% failed the complexity rule**. Under the frozen hard-fail rule
  (concordant severe evidence across multiple families), **none of these 691 cells would
  be a hard fail** — all would enter `borderline` and become rescue-eligible.
- **Mast cells are natively the lowest-complexity population in the tissue** (retained
  median 744 genes vs 2,248–2,836 elsewhere). A global complexity floor would be
  actively dangerous for them; their mito elevation is marginal (p50 22.8%, barely over
  the line).

### The mast-cell condition bias is real and ceiling-driven

| | n | mito p50 | mito p90 | % above 20% |
|---|---|---|---|---|
| Mast, Lymphedema | 10,209 | 6.95% | 14.5% | **2.85%** |
| Mast, Normal | 8,918 | 5.45% | 11.6% | **1.09%** |
| KC, Lymphedema | 4,148 | 5.50% | 18.2% | 8.00% |
| KC, Normal | 5,252 | 5.35% | 16.4% | 6.84% |

A hard ceiling converts a modest distributional shift into 2.6× differential removal
(CMH OR 2.86, p_adj 7.2e-20 cell-level; donor-level paired Wilcoxon p_adj 0.051). For
keratinocytes the bulk medians are nearly identical (5.50 vs 5.35) and **only the tail
diverges** — precisely what a fixed ceiling amplifies. Mast cells are a target
population with their own config and manuscript.

### Baseline mito varies ~3× across cell types in the same tissue

Retained-cell median mito: LEC 2.3%, Macrophages 4.4%, Pericyte/SMC 4.9%,
Keratinocytes 5.1%, Mast 6.1%. One ceiling cannot be correct for all of them. This is
the per-cell-type analogue of the per-sample argument in `mixture.py`, and it is why
final adjudication belongs at order 135 after annotation rather than at order 20.

### The condition-bias audit already partly exists

`qc_attrition.csv` and `qc_attrition_by_cell_type.csv` already implement much of §7 of
the architecture spec: CMH stratified by donor, paired donor-level Wilcoxon, per cell
type, with FDR adjustment. The two tests **disagree in the informative direction** —
cell-level OR 1.39 p=4.6e-17 versus donor-level p=0.098 — confirming that cell-level
inference is anticonservative here. Build on this rather than replacing it.

### Figure-grammar defects to fix in V2

- Patient order is **string-sorted** (`P1, P10, P12, P2, P4…`) in every current figure.
  Use natural sort.
- Cell-type composition figure facets **Disease before Normal**; the split-violin QC
  figure puts **Lymphedema on the left**. Both violate Normal-first.
- Boxplots are drawn with `showfliers` off, so **the cells the thresholds act on are
  invisible**. This is what caused an incorrect reading of panel A — the mito ceiling
  looked non-binding when in fact it accounts for ~100% of removals. V2 density/raincloud
  plots must show the tail.
- Distributions labelled "After Filtering" cannot calibrate the thresholds that produced
  them. Calibration figures must use pre-filter metrics.

## Open question — module layout

There are currently three candidate homes for QC plots:

- `visualization/qc/publication.py` — tracked, legacy `cellquorum_qc_keep` semantics
- `visualization/qc/panels.py` — **untracked, in flight**, 2135 lines, already contains
  `qc_joint_density`
- `visualization/qc/graded.py` — proposed new V2 module

Creating `graded.py` alongside an in-flight `panels.py` risks three parallel QC plotting
modules. Decide before implementation whether `panels.py` is itself becoming the V2
module, whether `graded.py` absorbs the parts of `panels.py` worth keeping, or whether
`panels.py` is legacy-bound.
