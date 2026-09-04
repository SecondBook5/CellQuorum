"""Reusable statistical primitives that sit on top of engine stage outputs.

These are study-agnostic building blocks — no lineage or study biology is
hardcoded. They take plain numpy/pandas inputs (a per-cell score matrix plus a
metadata frame) rather than an :class:`~anndata.AnnData`, so they are trivially
unit-testable and importable from any hypothesis repo. A caller extracts the
score matrix (e.g. ``obsm["X_state_aucell"]`` written by the ``state_scoring``
stage) and the design columns, then calls these functions.

The module-remodeling suite (donor-aware LMM effect sizes, PERMANOVA-by-group,
signature-argmax subtyping, the signed program-contrast index, leading-edge
concordance, and program correlations) makes rigorous — and reusable — the
by-hand statistics in the LEC "Module Remodeling" analysis. The study-specific
pieces (which gene modules, which subtype signatures, the EndoMT index
definition, figure titles) live in the hypothesis repo, not here.

The depth-confounding audit answers a question that applies to every continuous
per-cell readout the engine writes, not just module scores: does this metric's
condition effect survive adjustment for library depth, or is it a library-size
readout? It exists because a pseudotime shift that was unanimous across donors at
p=0.004 turned out to correlate with gene count at rho=-0.856 and to vanish
entirely under adjustment, while an AUCell index on the same object was
untouched. Nothing about either metric's name distinguished them.

Causal mediation asks the mechanistic follow-up to any contrast: of the total
change in an outcome program, how much travels through a candidate mediator
program? It is a decomposition, not a test of a new effect, and it fails in four
specific ways on data shaped like ours — fitted on cells instead of samples, fitted
on paired donors as if they were independent, reporting a mediated *fraction* of a
total effect that straddles zero, and scoring the mediator and the outcome from
overlapping gene sets. All four are guarded rather than left to the caller.

Gene-set overlap answers the question that every multi-program panel invites: are
these independent readouts, or restatements of each other? A similarity coefficient
alone cannot say — the same Jaccard can be strong evidence or none at all depending
on the set sizes and on how many genes could have been shared — so the overlap tests
always carry a hypergeometric p-value, and the universe they are tested against is a
required argument. There is no default because the convenient default (the union of
the sets) is the smallest defensible universe and therefore the most flattering one.

Program correlation is the same question asked of the *scores* rather than the gene
lists, and it is the table most often produced with a wrong n. A program-by-program
``DataFrame.corr()`` over per-cell scores gets the coefficient right and the sample
size wrong by two orders of magnitude, because cells within a donor are not
independent observations of anything. So the unit is named rather than assumed, the
condition-adjusted partial correlation is reported beside the raw one (two programs
both raised in disease correlate across samples without co-varying within either arm),
and the shared-gene count travels with each pair so a correlation that is arithmetic
cannot be read as one that is biological.

Partition agreement is for the moment a project has more than one clustering of the same
cells and a table keyed on one has to be read beside a figure keyed on another. A single
agreement index cannot tell a disagreement from a refinement — an eight-cluster partition
nesting inside a three-label one returns much the same ARI as two partitions that genuinely
cut the data differently — so every pair carries the two directional purities, and the
intersection the index was computed over is reported rather than assumed.

The paired-concordance audit is the same shape of question asked of an abundance
claim rather than a per-cell metric: every abundance method reports a cohort mean,
and a mean cannot say whether a shift happened in most donors or hugely in a few.
It exists because a compositional fit's one credible call on a 9-donor cohort moved
in the reported direction in only a minority of donors, while the two shifts that
were unanimous or near-unanimous were never called at all.

Claim support asks two questions of a finished results table, both properties of the
design rather than the data: could the test have cleared its own threshold, and is the
row's ratio measured on enough cells to mean anything. It exists because a Wilcoxon on
nine donors corrected over thirteen cell types cannot return an FDR below 0.051, and
"nothing clears FDR" was read off that arrangement and written down as a biological
null; and because a fold-change axis hands the top of a ranking to whichever group is
rarest, which on this cohort was a group with a median of 1.5 cells per sample.

The cluster-artifact audit asks the question that has to be settled before any of the
above is worth computing: are all of these clusters cells? Ambient debris collects
into its own cluster and is then counted as a population. It is usually found by eye
and masked by hardcoded Leiden id, which is wrong twice over — no single mark
identifies debris (low complexity alone is a real low-RNA lineage), and an id belongs
to one clustering run, so a mask carried to the next partition deletes whatever
inherited the number and leaves the artifact in.
"""

from __future__ import annotations

from cellquorum.stats.causal_mediation import (
    mediation_effects,
    mediation_grid,
)
from cellquorum.stats.claim_support import (
    FDR_REACHABILITY_COLUMNS,
    GROUP_RESOLUTION_COLUMNS,
    MIN_CELLS_PER_SAMPLE,
    annotate_fdr_reachability,
    group_resolution,
)
from cellquorum.stats.cluster_artifacts import (
    AMBIENT_DEBRIS,
    DEFAULT_DEBRIS_VERDICTS,
    LIBRARY_ARTIFACT,
    cluster_artifact_audit,
    debris_clusters,
    verify_declared_debris,
)
from cellquorum.stats.depth_confounding import (
    depth_confound_audit,
    depth_stratified_abundance,
)
from cellquorum.stats.gene_set_overlap import (
    cross_membership,
    cross_overlap_tests,
    exclusive_combinations,
    selection_overlap,
    set_overlap_tests,
    set_sizes,
)
from cellquorum.stats.module_remodeling import (
    DEFAULT_FDR_COLUMNS,
    FAMILY_COLUMNS,
    PANEL_MEMBERSHIP_COLUMNS,
    bh_fdr,
    declared_panel_membership,
    fdr_floor_reachability,
    leading_edge_jaccard,
    lmm_effect_sizes,
    module_gene_overlap,
    permanova_by_group,
    program_correlation_matrix,
    randomization_floor,
    recorrect_within_family,
    signature_argmax_labels,
    signed_program_contrast_index,
    upset_membership,
)
from cellquorum.stats.paired_concordance import (
    donor_unanimous,
    mark_called,
    paired_abundance_concordance,
    paired_value_concordance,
    qualify_abundance_calls,
)
from cellquorum.stats.partition_agreement import (
    align_partitions,
    cluster_group_support,
    label_composition,
    partition_agreement,
    partition_crosstab,
)
from cellquorum.stats.program_correlation import (
    program_correlation_tests,
)

__all__ = [
    "AMBIENT_DEBRIS",
    "DEFAULT_DEBRIS_VERDICTS",
    "DEFAULT_FDR_COLUMNS",
    "FAMILY_COLUMNS",
    "FDR_REACHABILITY_COLUMNS",
    "GROUP_RESOLUTION_COLUMNS",
    "LIBRARY_ARTIFACT",
    "MIN_CELLS_PER_SAMPLE",
    "PANEL_MEMBERSHIP_COLUMNS",
    "align_partitions",
    "annotate_fdr_reachability",
    "bh_fdr",
    "cluster_artifact_audit",
    "cluster_group_support",
    "cross_membership",
    "cross_overlap_tests",
    "debris_clusters",
    "declared_panel_membership",
    "depth_confound_audit",
    "depth_stratified_abundance",
    "donor_unanimous",
    "exclusive_combinations",
    "fdr_floor_reachability",
    "group_resolution",
    "label_composition",
    "leading_edge_jaccard",
    "lmm_effect_sizes",
    "mark_called",
    "mediation_effects",
    "mediation_grid",
    "module_gene_overlap",
    "paired_abundance_concordance",
    "paired_value_concordance",
    "partition_agreement",
    "partition_crosstab",
    "permanova_by_group",
    "program_correlation_matrix",
    "program_correlation_tests",
    "qualify_abundance_calls",
    "randomization_floor",
    "recorrect_within_family",
    "selection_overlap",
    "set_overlap_tests",
    "set_sizes",
    "signature_argmax_labels",
    "signed_program_contrast_index",
    "upset_membership",
    "verify_declared_debris",
]
