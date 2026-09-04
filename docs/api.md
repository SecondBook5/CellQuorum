# Python API

CellQuorum has two front doors: the `cellquorum` / `cq` command-line interface
(see the [README](https://github.com/SecondBook5/cellquorum#cli-reference)) and the
Python API documented here. The Python surface is deliberately small — one
entry point plus four notebook namespaces — and re-exported from the top-level
`cellquorum` package, so `cq.run_pipeline`, `cq.tl`, `cq.pp`, `cq.diag`, and
`cq.evidence` are the canonical import paths.

```python
import cellquorum as cq

result = cq.run_pipeline(config="configs/config.yaml")
```

## Pipeline entry point

::: cellquorum.run_pipeline

::: cellquorum.api.pipeline.PipelineRunResult

## Notebook namespaces

These namespaces expose the engine's stages as ergonomic functions for
interactive (notebook) use. Each is a thin adapter over the same stage classes
the CLI runs, so a notebook exploration and a config-driven run stay consistent.

### Tools — `cq.tl`

::: cellquorum.api.tl

### Preprocessing — `cq.pp`

::: cellquorum.api.pp

### Diagnostics — `cq.diag`

::: cellquorum.api.diag

### Evidence — `cq.evidence`

::: cellquorum.api.evidence

## Reusable utilities — `cellquorum.utils`

A few of the engine's internal helpers are useful on their own in analysis
scripts, independent of a full pipeline run. They are exposed here as a stable,
versioned surface (`cq.utils.*`) — re-exports of the canonical implementations in
`cellquorum.stages.comparative`, so a fix to the engine is a fix here. Importing this
module pulls in no heavy optional dependency (`get_net` lazy-imports `decoupler`
only when called).

```python
import cellquorum as cq

ranked = cq.utils.de_table_to_ranking(de_table)        # DE table -> preranked GSEA input
net = cq.utils.get_net("hallmark", organism="human")   # long-format prior-knowledge net
pb = cq.utils.aggregate_pseudobulk(                     # cells -> donor x condition pseudobulk
    adata, layer="counts", donor_col="donor_id", condition_col="condition"
)
```

::: cellquorum.utils.de_table_to_ranking

::: cellquorum.utils.get_net

::: cellquorum.utils.aggregate_pseudobulk

::: cellquorum.utils.PseudobulkResult

::: cellquorum.utils.PriorFetchError

## Statistical primitives — `cellquorum.stats`

Study-agnostic statistics that operate on a **per-cell score matrix plus a design
frame**, not on an `AnnData`. A caller pulls the matrix a stage wrote (for example
`obsm["X_state_aucell"]` from `state_scoring`) and the design columns, then calls these
directly — from a stage, a notebook, or a hypothesis repo. Because they take plain
numpy/pandas they are unit-testable against synthetic fixtures, which is why the house
statistical bar is enforced *here* rather than left to each caller:

- pseudoreplication is absorbed **at the level the contrast actually varies at**
  (`lmm_effect_sizes`), never a raw per-cell test. A donor random intercept absorbs the
  donor *mean*, which is enough only when condition is a property of the donor. In a
  paired cohort condition varies *within* donor, at the sample, so the model carries a
  sample variance component nested in donor as well, and its fixed effect is referred to
  a **t** with between-within denominator df rather than to a normal. Both parts matter:
  on a null fixture the donor intercept alone reads p≈1e-46, adding the t reference makes
  it 2e-06, adding the sample component makes it 0.05, and both together give 0.09 —
  which is what the donor-level paired t on the same data says;
- every parametric p-value is placed against the smallest one the design's own
  randomization set can reach (`randomization_floor`), and the assumption-free
  donor-level randomization p (exact sign-flip, or label permutation for an unpaired
  stratum) is reported beside it. A parametric p *may* legitimately fall below the floor —
  that is what the distributional assumption buys — so the floor is reported as a scale,
  not a veto: one or two orders below it means the result leans on the assumption, forty
  orders below means the model is not describing the cohort;
- every test family is **BH-FDR** corrected (`bh_fdr`, which holds NaN p-values out of
  the correction instead of letting one un-fittable row move the others) — and where the
  family is wide enough that its own floor puts BH out of reach for any lone result,
  `fdr_floor_reachability` says how many tests would have to reach the floor *together*
  before any of them could be called. Eight donor pairs in a 45-test family need eight;
  seven pairs need fifteen. Without that number, "nothing survived correction" reads as an
  absence of signal when it is often an arithmetic impossibility. The remedy is a family
  declared *before* the scan (`declared_panel_membership`) with every family column recomputed
  at the new size (`recorrect_within_family`), never a laxer threshold;
- permutations and sampling are **seeded** and deterministic (`permanova_by_group`);
- guards (≥ 2 donors per arm) trigger an **explicit, recorded fallback** — one value per
  donor, then the t-test the design supports — rather than a silent crash or a
  misleading estimate;
- the table carries **one column that needs no assumption at all**:
  `n_donors_concordant`, how many of the paired donors moved in the direction of the
  reported effect. A p-value is a function of the sign pattern and an effect size is a
  function of the magnitudes, so neither distinguishes "every donor moved a little" from
  "one donor moved a lot and the rest sat still" — and those are different claims about a
  disease. `7/9` is also the one number in the table a reader can check by eye;
- **no row is blank or degraded without saying why.** `lmm_effect_sizes` returns a
  `method` (`lmm` / `paired_t` / `welch_t` / `none`), a `variance_components`
  (`donor+sample` / `donor`), a nullable `converged`, and a `reason` that is empty only
  for a clean mixed-model fit. Four consequences worth knowing before you read the table:
    - the fallback **matches the design**: donors spanning both arms are paired,
      independent donors get Welch. A paired test on an unpaired stratum has nothing to
      pair and returns a blank row, which is indistinguishable from a null;
    - a fit that did not converge is **kept and flagged**, not discarded — its fixed
      effect, standard error and CI are still valid, and dropping them silently
      substitutes a t-test for the model you asked for;
    - an undefined test **reports its effect and withholds only its p-value**. Zero (or
      floating-point-noise) spread among the paired differences makes the t statistic
      undefined, not infinitely significant, so no row ever carries `p_value == 0`;
    - a design whose sample component **cannot be estimated** falls back to the donor
      intercept, keeps the fit, and says in `reason` that its p-value is anticonservative
      and that `donor_p` is the one to read. Rows flagged `p_below_design_floor` stay in
      the FDR family — dropping them would change every other row's q-value.

```python
from cellquorum.stats import (
    lmm_effect_sizes,
    permanova_by_group,
    randomization_floor,
    signature_argmax_labels,
)

# score ~ condition + (1|donor) + (1|sample), per program x group. Returns effect, CI, p,
# FDR, n, the design's own floor and donor-level randomization p, plus
# method/variance_components/converged/reason so a blank or fallback row explains itself.
effects = lmm_effect_sizes(
    scores,
    metadata,
    donor_col="donor_id",
    condition_col="condition",
    group_col="lec_subtype",
    case="Disease",
    control="Normal",
    # The unit condition was assigned to. Omit it and donor x condition is reconstructed,
    # which is exact for one library per donor per arm and conservative otherwise.
    sample_col="sample_id",
)

# Read the parametric p beside the design's own answer, never on its own:
effects[
    [
        "group",
        "program",
        "effect",
        "p_value",
        "fdr",
        "n_pairs",
        "n_donors_concordant",
        "design_floor_p",
        "p_below_design_floor",
        "donor_p",
        "donor_test",
    ]
]

# The floor on its own, for a design you have not scored yet. Eight pairs cannot go
# below 2/2**8 = 0.0078 however many cells they contain; a single pair's floor is 1.0.
floor_p, n_pairs = randomization_floor(metadata["donor_id"], metadata["condition"].eq("Disease"))

# Multivariate condition effect per group: seeded permutation pseudo-F -> R2 + p.
r2 = permanova_by_group(
    scores,
    metadata,
    sample_col="sample_id",
    condition_col="condition",
    group_col="lec_subtype",
    case="Disease",
    control="Normal",
)

# Label each cluster by its dominant signature; below min_margin stays "unassigned".
labels = signature_argmax_labels(scores, cluster_labels, min_margin=0.25)
```

::: cellquorum.stats.lmm_effect_sizes

::: cellquorum.stats.randomization_floor

::: cellquorum.stats.fdr_floor_reachability

::: cellquorum.stats.permanova_by_group

::: cellquorum.stats.signature_argmax_labels

::: cellquorum.stats.signed_program_contrast_index

::: cellquorum.stats.leading_edge_jaccard

::: cellquorum.stats.bh_fdr

### When the family is too wide for the cohort

`fdr_floor_reachability` diagnoses a family BH cannot pass; these two are the only honest
remedy. A scan of 29,113 ligand–receptor pairs on nine donor pairs has a randomization floor
of 2/2⁹ = 0.0039, and its largest family of 7,555 tests would need **591 of them unanimous
together** before BH passed a single one. Loosening α does not fix that. A **smaller family,
declared before the scan ran**, does.

The rule that builds the smaller family must never read a p-value, which is what
`declared_panel_membership` enforces: it takes gene sets that already exist for another reason
(a manifest's declared modules) and keeps only the composite items whose *every* entity is a
declared gene. Requiring both ends is the point — "any declared gene" would admit most of the
resource, since a broadcast ligand touches hundreds of receptors. The check that the panel was
not quietly drawn around the winners is that the largest effect in the scan is usually *not*
in it, and that is visible in the returned frame rather than argued for.

`recorrect_within_family` then does the arithmetic that is easy to get half-right. BH depends
on the company an item keeps, so a restricted table carrying the scan's `sign_test_fdr` forward
reports a correction for a family it is no longer in — and carrying `family_size` or
`family_min_concordant` forward is the identical mistake one column over, which is easier to
miss because those columns look descriptive. All of them move together, per family, and the
incoming size is preserved as `n_scanned` so a reader can see what fraction of the tested space
was pre-specified and cannot mistake a panel result for a discovery. No p-value is recomputed;
only the multiplicity accounting moves.

```python
from cellquorum.stats import declared_panel_membership, recorrect_within_family

# "FN1->ITGAV_ITGB1" qualifies (three declared genes); "FN1->TIE1" does not (receptor
# undeclared), and neither does "ANGPT2->TIE1", the scan's largest single effect.
member = declared_panel_membership(modules, scan["lr_pair"], item_label="lr_pair")
panel = scan.merge(member, on="lr_pair").query("in_panel")

# One family per focus x direction: 7,555 -> 38 tests, 591 -> 3 needed unanimous.
panel = recorrect_within_family(panel, by=("focus", "flow"), alpha=0.05)
panel[["family_size", "n_scanned", "family_min_concordant", "sign_test_fdr_conservative"]]
```

::: cellquorum.stats.declared_panel_membership

::: cellquorum.stats.recorrect_within_family

### Reading a paired cohort's verdict

`paired_value_concordance` reports a donor-agreement `pattern` and a family-corrected FDR, and
they answer different questions: `pattern` is uncorrected and says nothing about how many other
items were asked the same thing, while an FDR says nothing about whether the donors agreed — an
item can clear it on one outlying donor. `mark_called` is the single boolean that clears both,
so a caption cannot quote one hurdle and imply the other. It gates on the *conservative* sign
FDR, the doubled one-sided value, because the direction being tested was read off the same donor
deltas.

`donor_unanimous` is the summary with no effect size in it at all, which is what makes it worth
putting beside one — and it is worthless unless it can be lost. `n_agree >= n_pairs` is
**vacuously true at 0/0** and nearly free at 3/3, so reported without a floor the
strongest-looking rows in a table are the ones with the least data behind them, and a figure
that selects rows this way will draw empty ones. The floor is the house
`MIN_PAIRED_BLOCKS` = 6, the same threshold that decides whether a grid cell may set a colour
scale, so "unanimous" denotes one thing in every table and figure of a run.

```python
from cellquorum.stats import donor_unanimous, mark_called

table = mark_called(paired_value_concordance(values, donors, conditions, case=..., control=...))
table["unanimous"] = donor_unanimous(table)  # False at 0/0 and at 3/3
```

::: cellquorum.stats.paired_value_concordance

::: cellquorum.stats.mark_called

::: cellquorum.stats.donor_unanimous

### Is this effect a biological effect or a library-size effect?

Every continuous per-cell readout the engine produces — pseudotime, potency, module
activity, any `obs` column a stage writes — is built by summing or ranking gene
detections, so deeper libraries score higher on all of them at once. If depth also
differs between the arms, which is a property of the cohort and not of the analysis,
then a condition effect on any of them has a second and uninteresting explanation that
**no amount of donor pairing removes**. The failure is not hypothetical: on a lymphatic
slice `dpt_pseudotime` moved with condition in 9/9 donors at p = 0.004 and correlated
with `n_genes_by_counts` at rho = −0.856; after adjustment it was 4/9 donors at p = 0.76.
The AUCell module index on the same cells was unchanged. Nothing about either name told
you which belonged in a paper.

Confounding is a three-way condition, so `depth_confound_audit` tests all three legs
rather than flagging correlation alone: whether **depth differs by condition** (which
gates the other two — on balanced arms nothing can be depth-driven however hard it tracks
depth), how strongly **the metric tracks depth**, and whether **the effect survives**
residualising on `log1p(depth)` and re-testing with identical machinery. The slope it
residualises on is estimated *within* donor-condition samples, not pooled: a metric with a
real condition effect inherits a pooled correlation with depth through condition alone, and
adjusting on the pooled slope removed 40% of a known true effect on a fixture built
independent of depth.

The verdict vocabulary is one column — `depth_balanced`, `robust`, `attenuated`,
`depth_driven`, `depth_masked`, `no_raw_effect`, `insufficient_pairs` — and the audit never
rewrites a result, so raw and adjusted sit side by side and the reader sees which claim is
which.

`depth_masked` is the one verdict that **adds** a result rather than removing one, and it
exists because confounding has a direction. Where a metric rises with depth and the deeper
arm is the case arm, depth pushes the case mean up and cancels part of a genuine *fall*: the
unadjusted test sees nothing, the adjusted one sees the whole effect. Filing that under
`no_raw_effect` would hide the one row the audit *found* rather than protected — a caller
filtering on the verdict column would never see it. It is reported as a **lead and not a
call**, because the unadjusted test is the one any declared FDR family was corrected over,
so the honest question it raises is whether the direction was predicted in advance. An
audit that can only ever remove a claim is under-reading its own output.

**The `module_remodeling` stage runs this itself**, on every run, and writes
`module_depth_audit.csv` beside its effect sizes: one row per program plus the contrast
index, a warning naming any `depth_driven` program, a note naming any `depth_masked` one,
`n_depth_driven` / `n_depth_attenuated` / `n_depth_masked` on its metrics, and
`depth unaudited` on the stage's headline when no depth column could be resolved. It looks for `depth_col`, then scanpy's
`n_genes_by_counts`, then `total_counts`. A `depth_col` that is absent from `obs` skips the
audit rather than falling back — an audit against a covariate the caller did not name would
be read as evidence about the one they did. The audit is run un-partitioned even when the
effects are per group, because leg one is a cohort property and splitting the donors again
would put most groups below `min_pairs`.

```python
from cellquorum.stats import depth_confound_audit

audit = depth_confound_audit(
    scores,                       # cells x metrics: module scores, pseudotime, potency
    adata.obs,
    donor_col="donor_id",
    condition_col="condition",
    case="Disease",
    control="Normal",
    depth_col="n_genes_by_counts",  # gene count over UMI count: less saturating
)
audit[["metric", "spearman_rho_vs_depth", "raw_delta", "adjusted_delta", "verdict"]]
```

`depth_stratified_abundance` is the companion for the compositional version of the same
problem, which residualisation cannot reach: cluster membership is categorical, so if a
cluster is largely a depth stratum then the deeper arm appears to gain it. It re-runs each
label's donor-paired proportion test inside global depth-quantile strata — the same bin
means the same depth range in both arms — and a shift that is an artefact loses its sign
consistency inside the bins while a real one keeps it in every stratum. No slope, so
nothing to under-correct.

::: cellquorum.stats.depth_confound_audit

::: cellquorum.stats.depth_stratified_abundance

### Are all of these clusters cells?

Ambient RNA, stripped nuclei and low-complexity debris do not spread themselves evenly
across an atlas. They collect, they get their own Leiden id, and from that point on they
are counted as a population: a composition figure reports one library's soup as a
condition-specific gain, and every lineage the debris leaked into carries its genes into
a differential test.

The usual handling is to notice a blob in the middle of the UMAP, look at it, and
hardcode its id into a mask. `cluster_artifact_audit` exists because that practice is
wrong in two ways at once, and both are silent.

**A debris cluster must be re-identified by criteria, never carried across partitions by
id.** Leiden numbering belongs to one clustering run, not to the cells. Re-cluster after
changing QC, integration or resolution and every id is reassigned. This is not a
hypothetical: a `{"18", "30", "40"}` mask written against one build of an atlas, applied
to the rebuilt one, would have deleted a real 1,902-cell cluster and left the actual
7,175-cell graveyard in — and the mask's presence reads as having handled it.
`verify_declared_debris` raises on a declared id that is not a cluster of the partition in
hand, and returns both other disagreements (declared-but-clean, audited-but-not-declared)
as rows with cell counts, so the cost of each is a number rather than an assumption.

**No single mark identifies debris; the conjunction does.** Each mark on its own is
routinely a real population, and the ones it would delete are the ones least likely to be
missed:

- *Low complexity* alone is a genuinely low-RNA cell type. Neutrophils and cornified
  keratinocytes sit at 0.21× and 0.37× the atlas median gene count and are populations.
- *Annotation-confidence collapse* alone is a doublet cluster or a real transitional state.
- *Lineage promiscuity* alone is a clustering that cut the data differently from the
  annotation, which is a labelling question, not a data-quality one.
- *Single-library dominance* alone is a donor-private population, unremarkable in a
  nine-donor cohort.

Ambient debris is mechanistically a shallow mixture of everything in the suspension, so it
is shallow **and** unassignable **and** every lineage at once. On the atlas this was
calibrated against, that conjunction picked out one cluster of thirty-nine; complexity
alone picked out six, five of them real.

Two signals are reported and deliberately **not** scored. **Condition dominance**, because
a cluster that is 93% cases is either the finding or the artifact and no column can tell
you which — an audit that downgraded clusters for being disease-specific would delete the
result. What disambiguates is whether it is also one *library*: 93% cases across nine
donors is biology, 93% cases because 77% came from one case library is that library. And
**embedding position**, because the UMAP is downstream of the same PCA the debris distorts,
so a central position is a symptom of the artifact and of the pipeline's response to it in
unknown proportion.

The verdicts (`ambient_debris`, `library_artifact`, `ambiguous_annotation`,
`low_complexity`, `promiscuous`, `low_confidence`, `single_library`, `clean`,
`insufficient_cells`) are leads; only the two conjunctions are treated as debris by
default. Every threshold is exposed twice, as the measured value and as the boolean it
produced, so a reader can move one without re-running anything — and `marks` is the column
to quote, since it is *why* where `verdict` is only *what*.

```python
from cellquorum.stats import cluster_artifact_audit, debris_clusters, verify_declared_debris

audit = cluster_artifact_audit(
    adata.obs["leiden"],
    complexity=adata.obs["n_genes_by_counts"],   # genes over counts: saturates more slowly
    confidence=adata.obs["cell_type_granular_confidence"],
    lineage=adata.obs["cell_type"],
    library=adata.obs["sample_id"],              # the library, not the donor
    condition=adata.obs["condition"],            # reported, never scored
    embedding=adata.obsm["X_umap"],              # reported, never scored
)

mask = debris_clusters(audit)                    # recomputed, so it cannot be stale
verify_declared_debris(audit, prior_run_mask)    # raises if the mask is from another partition
```

Acting on the finding is `input.exclude` in the config (or `exclude_on`/`exclude_values`
in a hypothesis manifest) — see
[Configuration](configuration.md#the-input-section). It drops the audited cluster at
load time rather than deleting cells from the shared object, which keeps the atlas one
file for every analysis that reads it and keeps the removal in the run's own provenance
(`n_excluded`) instead of in a build script's history. The exclusion is deliberately not
expressible as a subset: naming what to *keep* out of a thirty-nine-cluster partition
means listing thirty-eight ids, which is unreadable and silently incomplete the next
time the object gains a cluster.

::: cellquorum.stats.cluster_artifact_audit

::: cellquorum.stats.debris_clusters

::: cellquorum.stats.verify_declared_debris

### Is this panel of programs actually a panel?

A study that scores eleven programs and reports eleven effects is claiming eleven things
were measured. These three answer whether that is true, and they are deliberately hard
to misuse:

- **`set_overlap_tests` requires a `universe`.** A hypergeometric p-value is a function
  of how many genes *could* have been shared, so there is no default — the convenient
  default (the union of the sets) is the smallest defensible universe and therefore the
  most flattering one.
- **For hand-curated modules, read `exclusive_combinations`, not the p-value.** Two
  modules written from one literature share VIM more often than chance, so the test
  returns 1000-fold enrichments that are the arithmetic of having written FN1 into two
  lists. What is *not* foregone is how much of each module is its own.
- **`program_correlation_tests` names its unit.** Cells within a donor are not
  independent observations, so passing `sample_col` averages within each sample before
  correlating; passing nothing declares that the rows *are* the units, and the frame
  records which. There is no path through it that quietly treats cells as independent.

```python
from cellquorum.stats import (
    exclusive_combinations,
    program_correlation_tests,
    set_overlap_tests,
)

# Pairwise overlap with a hypergeometric p-value against a stated universe.
overlaps = set_overlap_tests(programs, universe=adata.var_names)

# How much of each program is its own — the readout for curated sets.
combos = exclusive_combinations(programs)

# Correlation at the donor level, condition-adjusted, with the overlap disclosed.
correlations = program_correlation_tests(
    scores,
    metadata,
    sample_col="sample_id",
    condition_col="condition",
    program_genes=adata.uns["state_aucell"]["genes"],
)
```

::: cellquorum.stats.set_overlap_tests

::: cellquorum.stats.exclusive_combinations

::: cellquorum.stats.set_sizes

::: cellquorum.stats.program_correlation_tests

### Superseded

Kept so existing callers keep working; each emits a `DeprecationWarning` naming its
replacement. All three return a similarity or a coefficient with no null, no universe
and no unit, which is the failure the three functions above exist to prevent.

::: cellquorum.stats.program_correlation_matrix

::: cellquorum.stats.module_gene_overlap

::: cellquorum.stats.upset_membership
