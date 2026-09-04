"""Configuration for the module-remodeling comparative stage.

Every field here is a column name, a design token, or a rendering knob. No gene
symbol, module name, lineage or condition label appears in this file or anywhere
else in the package: the eleven curated endothelial modules that motivated the
stage live in the hypothesis repo's manifest, under ``state_scoring.programs``,
and reach the stage as data. That boundary is what makes one capability serve
the LEC manuscript and the next study equally, and it is enforced by a test.
"""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class ModuleRemodelingConfig(StrictBaseModel):
    """Condition effects on per-cell module activity, resolved per subtype.

    Consumes what ``state_scoring`` already computes — a cells x programs score
    matrix — and answers the question that stage stops short of: for each module,
    in each subtype, how much did activity shift between conditions, with what
    confidence, and is the shift subtype-restricted or global. ``state_scoring``
    emits tables and no figures; this stage is where the module activity becomes
    a panel.

    The statistics come from :mod:`cellquorum.stats`, so they are the same
    donor-aware primitives a notebook or hypothesis repo can call directly.

    Attributes:
        enabled: Whether the stage runs.
        method: Analysis method (currently only ``module_remodeling``).
        score_key: ``obsm`` key holding the cells x programs score matrix
            (``state_scoring``'s AUCell output).
        programs_uns_key: ``uns`` key whose ``["programs"]` entry gives the
            score matrix's column order. Read rather than assumed, so a change
            to the scored module set cannot silently mislabel columns.
        programs: Optional subset/ordering of programs to analyse; empty means
            all columns of the score matrix.
        cluster_col: ``obs`` column holding the data-driven partition to label
            (the subclustering stage's ``key_added``).
        group_col: ``obs`` column to use as the group axis *directly*, skipping
            signature labelling. Set this when the groups are already named.
        subtype_signatures: Group name -> the programs that mark it. Each
            cluster is assigned the group whose signature scores highest at
            cluster level (more stable than a per-cell argmax). Empty means
            every program is its own candidate signature.
        min_margin: Minimum standardized gap between the best and second-best
            signature for a cluster to be labelled at all; below it the cluster
            is recorded ``unassigned`` rather than guessed.
        group_order: Explicit column order for the group axis. Empty means a
            natural order (cluster 2 before cluster 10, names after numbers),
            which is the right default for a data-driven partition; set it when
            the groups have a meaningful sequence — a maturation axis, a
            severity grade — that no sort can infer. Labels no cell carries are
            dropped rather than drawn as empty columns, and observed labels the
            list omits are appended rather than hidden.
        group_label: Axis name for the group columns — what the groups *are*
            ("LEC subcluster"). Empty falls back to the column name with its
            underscores opened up, because an unlabelled axis of cluster numbers
            is a panel the reader cannot interpret.
        program_labels: Program -> row label, for manuscript wording
            (``endomt_lec`` -> "EndoMT (LEC)"). Programs left out render as their
            name with underscores opened up. Display only: the statistics, the
            tables and the obs columns always use the program key.
        module_categories: Category -> programs, used to group and order the
            flagship figure's rows. Programs missing from every category are
            appended under an "other" band rather than dropped.
        index_label: Axis name for the contrast index — the *quantity*, not the
            formula ("EndoMT index"). Empty falls back to ``index_key`` with its
            underscores opened up.
        index_up: Programs entering the signed contrast index positively.
        index_down: Programs entering it negatively. Both empty disables the
            index (no obs column, no figure) instead of emitting a zero column.
        donor_col: ``obs`` column identifying a donor — the random intercept
            that absorbs pseudoreplication.
        condition_col: ``obs`` column holding the condition label.
        sample_col: ``obs`` column identifying a sample; PERMANOVA operates on
            per-sample module vectors, not per-cell rows.
        case: Case condition label.
        control: Control condition label.
        min_donors_per_arm: Below this the mixed model is not attempted and a
            recorded donor-level fallback is used instead.
        fdr_method: Multiple-testing method across the module x group family.
        n_permutations: PERMANOVA permutations.
        seed: Seed for the permutations; recorded in the output table.
        correlation_method: ``spearman`` (default) or ``pearson`` for the
            program-vs-program correlation matrix.
        depth_col: ``obs`` column holding the library-depth covariate the module
            effects are audited against. Empty looks for scanpy's standard
            ``n_genes_by_counts`` then ``total_counts``, so an object that has been
            through QC needs no setting at all. A name given here and absent from
            ``obs`` skips the audit with a warning rather than falling back to a
            different column, because an audit against a covariate the caller did
            not ask for would be read as evidence about the one they did.
        min_depth_pairs: Complete donor pairs below which the depth audit declines
            to reach a verdict. ``None`` uses the house floor (six), below which a
            paired test cannot reach 0.05 at all, so "no longer significant after
            adjustment" would describe the cohort size rather than depth.
        effect_cap: Symmetric colour cap for the flagship dot-grid, in effect
            units. ``None`` derives it from the data (98th percentile of \\|effect\\|),
            which keeps one outlier from flattening every other dot -- but derives
            it PER PANEL, so two arms of one comparison end up on two colour
            scales and a pale dot in the stronger arm can out-saturate a strong dot
            in the weaker one. Set this explicitly whenever panels are meant to be
            read side by side.
        max_dot_fdr_exponent: Ceiling on the ``-log10(FDR)`` that sets dot area,
            so an astronomically small FDR cannot produce a dot that covers its
            neighbours.
        alpha: Significance level the correlation panels mark pairs at. Display
            only — no table's FDR is recomputed from it.
        max_correlation_pairs: Cap on the program pairs drawn in the
            correlation-adjustment panel, whose rows are the pairs with the largest
            change under condition adjustment. ``None`` draws every pair, which is
            unreadable past a few dozen programs.
    """

    enabled: bool = True
    method: str = "module_remodeling"

    # ---- Inputs -----------------------------------------------------------
    score_key: str = "X_state_aucell"
    programs_uns_key: str = "state_aucell"
    programs: list[str] = []

    # ---- Group axis -------------------------------------------------------
    cluster_col: str | None = None
    group_col: str | None = None
    subtype_signatures: dict[str, list[str]] = {}
    min_margin: float = 0.0

    # ---- Figure grouping + derived axis -----------------------------------
    group_order: list[str] = []
    group_label: str | None = None
    program_labels: dict[str, str] = {}
    module_categories: dict[str, list[str]] = {}
    index_up: list[str] = []
    index_down: list[str] = []
    index_label: str | None = None

    # ---- Design -----------------------------------------------------------
    donor_col: str | None = None
    condition_col: str | None = None
    sample_col: str | None = None
    case: str | None = None
    control: str | None = None

    # ---- Statistics -------------------------------------------------------
    min_donors_per_arm: int = 2
    fdr_method: str = "fdr_bh"
    n_permutations: int = 999
    seed: int = 1337
    correlation_method: str = "spearman"
    depth_col: str | None = None
    min_depth_pairs: int | None = None

    # ---- obs additions ----------------------------------------------------
    key_added: str = "module_subtype"
    index_key: str = "contrast_index"

    # ---- Rendering --------------------------------------------------------
    effect_cap: float | None = None
    max_dot_fdr_exponent: float = 6.0
    alpha: float = 0.05
    max_correlation_pairs: int | None = None


__all__ = ["ModuleRemodelingConfig"]
