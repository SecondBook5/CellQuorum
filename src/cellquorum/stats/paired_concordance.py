"""Donor-level concordance audit for a cohort mean, compositional or otherwise.

Two entry points. :func:`paired_abundance_concordance` audits a composition, where
every cell type has a share of every sample and the donor count is one number for
the whole table. :func:`paired_value_concordance` audits any other per-sample
quantity — a communication edge's weight, a pathway score, a module mean — where an
item can simply be *absent* for a sample, so the donors that can be paired differ
per item and the common-support rule has to be enforced and reported per item. They
share the classifier, :func:`_classify_pattern`, so the two cannot drift in what
they call a result while describing it in their own units.


Every abundance method the engine ships — scCODA, propeller, a paired proportion
test — reports a *cohort-mean* effect per cell type. A mean is silent about the
one thing a reader needs in order to know what was found: whether the shift
happened in most donors, or happened enormously in a few and not at all in the
rest. Those are different biological claims, and they arrive in identical output.

Why this is its own audit rather than a footnote
------------------------------------------------
On a 9-donor, 13-lineage skin cohort a donor-paired scCODA fit returned exactly
one credible effect, T/NK, at +4.6 percentage points. The number is real and it
is not one outlier: dropping any single donor leaves the mean between +2.6 and
+5.9pp. But the direction holds in only 5 of 9 donors, with the five gains large
(+5.5 to +20.6pp) and the four losses small (-0.2 to -6.5pp). The honest reading
is a T-cell-infiltrated *subset* of donors, not a cohort-wide shift — a subgroup
hypothesis, which is interesting, rather than a headline effect, which it is not.

The same cohort shows the sharper version of the problem. Fibroblasts move -3.5pp
on the mean and +2.2pp on the median: the cohort average and the typical donor
disagree about the sign. Nothing in the abundance table says so.

So the mean is kept and three donor-level facts are reported beside it, each of
which rests on a comparison rather than on a tuned cutoff:

* Does the mean's direction survive dropping any one donor? If not, one donor is
  the result.
* Do the mean and the median agree on the sign? If not, the cohort average is
  describing a different population than the typical donor.
* Is the direction shared by more donors than chance would give? This is the sign
  test, which ignores magnitude entirely and so cannot be carried by a few large
  movers.

Reading the ``pattern`` column
-----------------------------
``consistent`` is the only pattern that supports "cell type X changes in disease"
as written. ``heterogeneous`` says the cohort mean moves but donors disagree, and
points at a subgroup. ``direction_inconsistent`` and ``single_donor_driven`` mean
the reported mean should not be quoted without the qualifier. ``underpowered``
means the cohort is too small for the question, not that the answer is no.

Power limits, stated up front: the sign test is magnitude-free and therefore
coarse. Its smallest attainable one-sided p on n pairs is 2**-n, so 6 pairs
(0.0156, requiring unanimity) is the floor at which ``consistent`` is reachable at
all, and at 9 pairs a cell type needs 8 of 9 donors moving the same way (p =
0.0195) to clear 0.05. A ``heterogeneous`` verdict on a small cohort is therefore a
statement about resolution as much as about biology, which is why the donor counts
are reported next to it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# MIN_PAIRED_BLOCKS is imported rather than re-derived: it is the one house floor
# for a donor-paired test, on the grounds that below six pairs an exact signed-rank
# test cannot reach 0.05 at all, so any verdict would be a statement about the
# sample size instead of about the data.
from cellquorum.core.labels import as_label_strings
from cellquorum.stats.depth_confounding import MIN_PAIRED_BLOCKS
from cellquorum.stats.module_remodeling import _floor_p, bh_fdr

# Pseudocount for the centred log-ratio, matching what scCODA adds internally and
# what the reference selector uses, so all three describe the same geometry.
LOG_RATIO_PSEUDOCOUNT = 0.5

# The conventional two-sided bar for the sign test. Not tuned for this engine --
# lowering it would manufacture consistency and raising it would call every small
# cohort heterogeneous, and neither is a defensible default.
SIGN_TEST_ALPHA = 0.05

CONCORDANCE_COLUMNS: tuple[str, ...] = (
    "cell_type",
    "n_pairs",
    "n_agree",
    "direction",
    "mean_delta_pp",
    "median_delta_pp",
    "mean_delta_clr",
    "median_delta_clr",
    "loo_mean_clr_min",
    "loo_mean_clr_max",
    "loo_sign_stable",
    "signs_agree",
    "sign_test_p",
    "sign_test_fdr",
    "wilcoxon_p",
    "wilcoxon_fdr",
    "pattern",
    "reason",
)

#: Schema of :func:`paired_value_concordance`. ``n_pairs`` and ``n_donors_one_arm`` are the
#: two columns a composition cannot have: they are per item, because an arbitrary measurement
#: can be missing for a sample and a share of a whole cannot.
VALUE_CONCORDANCE_COLUMNS: tuple[str, ...] = (
    "item",
    "n_pairs",
    "n_donors_one_arm",
    "n_agree",
    "direction",
    "mean_case",
    "mean_control",
    "mean_delta",
    "median_delta",
    "loo_mean_min",
    "loo_mean_max",
    "loo_sign_stable",
    "signs_agree",
    "sign_test_p",
    "sign_test_fdr",
    "sign_test_p_conservative",
    "sign_test_fdr_conservative",
    "wilcoxon_p",
    "wilcoxon_fdr",
    "design_floor_p",
    "pattern",
    "reason",
)


def _conservative_sign_p(sign_test_p: float) -> float:
    """Double a one-sided sign p whose direction came from the data it is testing.

    The sign test here asks whether the donors corroborate a direction, and that
    direction is read off the cohort mean of the same deltas. A genuinely one-sided
    test would have had the direction from somewhere else -- a prior result, a
    hypothesis stated before the data -- so the nominal one-sided p is anticonservative
    by up to a factor of two: at nine pairs it can reach ``2**-9`` = 0.00195, whereas
    the smallest p any relabelling of nine donors can produce is
    :func:`~cellquorum.stats.module_remodeling.randomization_floor`'s two-sided
    ``2/2**9`` = 0.0039. Doubling puts the two on the same yardstick.

    The one-sided value is what :func:`_classify_pattern` continues to use, because
    that gate is a statement about donor agreement and its threshold was chosen
    against the one-sided scale. This value is what a family-wide correction and any
    "called" decision should rest on. ``NaN`` (no donor moved, or no direction) stays
    ``NaN``: there is nothing to make conservative.
    """

    if not np.isfinite(sign_test_p):
        return float("nan")
    return min(1.0, 2.0 * float(sign_test_p))


def paired_abundance_concordance(
    counts: pd.DataFrame,
    donors: pd.Series,
    conditions: pd.Series,
    *,
    case: str,
    control: str,
    min_pairs: int = MIN_PAIRED_BLOCKS,
) -> pd.DataFrame:
    """
    Audit whether each cell type's mean abundance shift is shared across donors.

    Pairs each donor's case and control samples, measures the per-donor change on
    the centred-log-ratio scale (the scale compositional models actually work on)
    and in percentage points (the scale a reader interprets), then reports how
    much of the cohort mean is donor-consensus and how much is a few large movers.

    Args:
        counts: Samples (rows) × cell types (columns) integer count matrix, as
            produced by ``aggregate_celltype_counts``.
        donors: Per-sample donor labels, indexed by ``counts.index``.
        conditions: Per-sample condition labels, indexed by ``counts.index``.
        case: Condition label treated as the case/disease arm.
        control: Condition label treated as the control arm.
        min_pairs: Donors with both arms required before a pattern is called.

    Returns:
        One row per cell type with the columns in :data:`CONCORDANCE_COLUMNS`,
        ordered by ``sign_test_p`` then by descending absolute mean shift. Empty
        (with the full schema) when no donor contributes both arms.

    Notes:
        Direction is assessed on the log-ratio delta because that is what the
        abundance models estimate; ``*_pp`` columns are descriptive companions and
        can differ in sign from the log-ratio when one large population absorbs
        another's change. A donor contributing several samples to an arm has that
        arm averaged, which is the standard handling for technical replicates and
        keeps one donor from counting several times toward consensus.

        ``sign_test_p`` here is one-sided against a direction taken from the same
        deltas, so it is anticonservative by up to a factor of two for the reason set
        out in :func:`_conservative_sign_p`. This function's columns are left as they
        are because published tables are keyed on them; a reader comparing a cell
        type's ``sign_test_fdr`` against a nominal 0.05 should double it first, or read
        it beside ``randomization_floor`` for the same donor set. The newer
        :func:`paired_value_concordance` reports both scales as columns.
    """

    empty = pd.DataFrame(columns=list(CONCORDANCE_COLUMNS))

    if counts is None or counts.empty or counts.shape[1] == 0:
        return empty

    # Name the cell types the way the rest of the run names them. A count matrix
    # usually arrives from ``aggregate_celltype_counts`` with canonical string
    # columns, but this function is callable on any count matrix, and a raw numeric
    # state column would otherwise be rendered "1.0" here and "1" everywhere else.
    cell_types = list(as_label_strings(pd.Series(list(counts.columns))))
    matrix = counts.to_numpy(dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)

    # A sample with no cells has no composition; it cannot pair with anything.
    usable = (totals[:, 0] > 0) & np.isfinite(totals[:, 0])
    if not usable.any():
        return empty

    proportion = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)

    # The CLR: log share minus the per-sample mean log share. Subtracting that
    # mean is what removes the arbitrary sequencing total, so a difference of two
    # CLRs is a change in a cell type's share *relative to the composition* rather
    # than a change in how many cells happened to be captured.
    padded = matrix + LOG_RATIO_PSEUDOCOUNT
    log_share = np.log(padded / padded.sum(axis=1, keepdims=True))
    clr = log_share - log_share.mean(axis=1, keepdims=True)

    aligned_donors = donors.reindex(counts.index).astype(str)
    aligned_conditions = conditions.reindex(counts.index).astype(str)

    row_of = {sample: position for position, sample in enumerate(counts.index)}

    delta_clr: list[np.ndarray] = []
    delta_pp: list[np.ndarray] = []
    paired_donors: list[str] = []
    for donor, block in aligned_donors.groupby(aligned_donors):
        samples = [s for s in block.index if usable[row_of[s]]]
        case_rows = [row_of[s] for s in samples if aligned_conditions.get(s) == case]
        control_rows = [row_of[s] for s in samples if aligned_conditions.get(s) == control]
        if not case_rows or not control_rows:
            continue
        delta_clr.append(clr[case_rows].mean(axis=0) - clr[control_rows].mean(axis=0))
        delta_pp.append(
            100.0 * (proportion[case_rows].mean(axis=0) - proportion[control_rows].mean(axis=0))
        )
        paired_donors.append(str(donor))

    if not paired_donors:
        return empty

    clr_deltas = np.vstack(delta_clr)
    pp_deltas = np.vstack(delta_pp)
    n_pairs = clr_deltas.shape[0]

    rows = []
    for column, cell_type in enumerate(cell_types):
        rows.append(
            _describe_cell_type(
                cell_type,
                clr_deltas[:, column],
                pp_deltas[:, column],
                n_pairs=n_pairs,
                min_pairs=min_pairs,
            )
        )

    table = pd.DataFrame(rows)

    # One test family per statistic across the cell types of a single comparison.
    table["sign_test_fdr"] = bh_fdr(table["sign_test_p"].to_numpy())
    table["wilcoxon_fdr"] = bh_fdr(table["wilcoxon_p"].to_numpy())

    table = table.sort_values(
        ["sign_test_p", "mean_delta_clr"],
        key=lambda s: s.abs().mul(-1) if s.name == "mean_delta_clr" else s,
    ).reset_index(drop=True)

    return table[list(CONCORDANCE_COLUMNS)]


def _describe_cell_type(
    cell_type: str,
    clr_delta: np.ndarray,
    pp_delta: np.ndarray,
    *,
    n_pairs: int,
    min_pairs: int,
) -> dict:
    """Measure one cell type's donor-level agreement and name the pattern."""

    mean_clr = float(clr_delta.mean())
    median_clr = float(np.median(clr_delta))

    # Direction comes from the mean, because the mean is what the abundance model
    # reports and what this audit is qualifying.
    direction = int(np.sign(mean_clr))
    moved = clr_delta[clr_delta != 0.0]
    n_agree = int((np.sign(clr_delta) == direction).sum()) if direction != 0 else 0

    # Sign test: is the direction shared more often than a coin flip? Magnitude
    # plays no part, so a handful of large movers cannot carry it.
    #
    # One-sided, and that is not a shortcut for extra power. The direction is not
    # chosen by this test -- it is handed over from the model's reported mean, and
    # the only question asked is whether the donors corroborate it. A two-sided
    # test answers a different question and gets it dangerously wrong here:
    # binomtest(1, 9, 0.5) two-sided is p=0.039, so a cell type where eight of nine
    # donors moved *against* the reported mean would clear a two-sided bar and be
    # labelled consistent.
    if direction != 0 and moved.size:
        agree_among_moved = int((np.sign(moved) == direction).sum())
        sign_test_p = float(
            stats.binomtest(agree_among_moved, moved.size, 0.5, alternative="greater").pvalue
        )
    else:
        sign_test_p = np.nan

    # Signed rank: direction weighted by how big each donor's change was. Reported
    # beside the sign test rather than instead of it -- the gap between the two is
    # the diagnostic.
    wilcoxon_p = np.nan
    if moved.size >= 1:
        try:
            wilcoxon_p, _ = _floor_p(float(stats.wilcoxon(clr_delta).pvalue))
        except ValueError:
            wilcoxon_p = np.nan

    # Leave-one-donor-out: does the cohort mean keep its sign without any one
    # donor? A flip means that donor is the finding.
    if n_pairs > 1:
        total = clr_delta.sum()
        loo_means = (total - clr_delta) / (n_pairs - 1)
        loo_min, loo_max = float(loo_means.min()), float(loo_means.max())
        loo_sign_stable = bool(np.all(np.sign(loo_means) == direction)) if direction != 0 else False
    else:
        loo_min = loo_max = mean_clr
        loo_sign_stable = False

    signs_agree = bool(np.sign(median_clr) == direction) if direction != 0 else False

    pattern, reason = _name_pattern(
        n_pairs=n_pairs,
        min_pairs=min_pairs,
        n_agree=n_agree,
        direction=direction,
        signs_agree=signs_agree,
        loo_sign_stable=loo_sign_stable,
        sign_test_p=sign_test_p,
        mean_pp=float(pp_delta.mean()),
        median_pp=float(np.median(pp_delta)),
        mean_clr=mean_clr,
        median_clr=median_clr,
        loo_min=loo_min,
        loo_max=loo_max,
    )

    return {
        "cell_type": cell_type,
        "n_pairs": n_pairs,
        "n_agree": n_agree,
        "direction": direction,
        "mean_delta_pp": float(pp_delta.mean()),
        "median_delta_pp": float(np.median(pp_delta)),
        "mean_delta_clr": mean_clr,
        "median_delta_clr": median_clr,
        "loo_mean_clr_min": loo_min,
        "loo_mean_clr_max": loo_max,
        "loo_sign_stable": loo_sign_stable,
        "signs_agree": signs_agree,
        "sign_test_p": sign_test_p,
        "wilcoxon_p": wilcoxon_p,
        "pattern": pattern,
        "reason": reason,
    }


def _classify_pattern(
    *,
    n_pairs: int,
    min_pairs: int,
    direction: int,
    signs_agree: bool,
    loo_sign_stable: bool,
    sign_test_p: float,
) -> str:
    """Name the donor pattern from the four facts that decide it, and nothing else.

    The decision lives here alone so that the compositional audit and the general
    paired-value audit cannot drift apart in what they *call* a result while
    describing it in their own units. Each caller writes its own ``reason``, because
    a message has to quote the scale the decision was made on and those differ; the
    gate order does not.
    """

    if n_pairs < min_pairs:
        return "underpowered"
    if direction == 0:
        return "heterogeneous"
    corroborated = np.isfinite(sign_test_p) and sign_test_p < SIGN_TEST_ALPHA
    if corroborated and loo_sign_stable:
        return "consistent"
    if corroborated:
        # Donors agree on the direction, yet the mean crosses zero when one is
        # removed: a single dissenting donor is large enough to cancel the rest, so
        # the effect size is not the group's even though the direction is.
        return "single_donor_driven"
    if not signs_agree:
        return "direction_inconsistent"
    return "heterogeneous"


def _name_pattern(
    *,
    n_pairs: int,
    min_pairs: int,
    n_agree: int,
    direction: int,
    signs_agree: bool,
    loo_sign_stable: bool,
    sign_test_p: float,
    mean_pp: float,
    median_pp: float,
    mean_clr: float,
    median_clr: float,
    loo_min: float,
    loo_max: float,
) -> tuple[str, str]:
    """Classify a cell type's donor pattern.

    The sign test is the primary gate, because donor consensus is the property
    being audited. The other two facts then split each side of it: among cell types
    the donors *do* corroborate, leave-one-out separates a group effect from one
    where a single large dissenter nearly cancels the rest; among those they do
    not, the mean-versus-median sign separates a mean that is merely noisy from one
    that points the opposite way to the typical donor.

    Every decision here is made on the log-ratio scale, so each message quotes the
    log-ratio numbers it actually used. The percentage-point figures are carried
    alongside because they are what a reader interprets, and they can disagree with
    the log-ratio: a large population can gain several percentage points while its
    share of the composition barely moves, and quoting pp for a decision made on
    the log ratio would read as a contradiction.
    """

    pattern = _classify_pattern(
        n_pairs=n_pairs,
        min_pairs=min_pairs,
        direction=direction,
        signs_agree=signs_agree,
        loo_sign_stable=loo_sign_stable,
        sign_test_p=sign_test_p,
    )

    if pattern == "underpowered":
        return "underpowered", (
            f"only {n_pairs} donor(s) contribute both arms, below the {min_pairs} at which a "
            f"one-sided sign test can reach {SIGN_TEST_ALPHA:g} even on unanimous donors"
        )

    if direction == 0:
        return "heterogeneous", "the cohort mean is exactly zero, so there is no direction to check"

    if pattern == "consistent":
        return "consistent", (
            f"the shift is in the same direction in {n_agree} of {n_pairs} donors "
            f"(sign test p={sign_test_p:.3g}) and survives dropping any one donor, so the cohort "
            f"mean of {mean_pp:+.2f}pp describes the group"
        )

    if pattern == "single_donor_driven":
        return "single_donor_driven", (
            f"{n_agree} of {n_pairs} donors move the same way (sign test p={sign_test_p:.3g}), but "
            f"removing one donor takes the mean log-ratio shift across zero (leave-one-out range "
            f"{loo_min:+.3g} to {loo_max:+.3g} against a cohort mean of {mean_clr:+.3g}), so one "
            f"dissenting donor sets the effect size"
        )

    if pattern == "direction_inconsistent":
        return "direction_inconsistent", (
            f"the cohort mean and the typical donor disagree on the sign of the log-ratio change "
            f"(mean {mean_clr:+.3g} against median {median_clr:+.3g}, direction shared by only "
            f"{n_agree} of {n_pairs} donors), so the mean is describing a few large movers rather "
            f"than the cohort; the percentage-point change is {mean_pp:+.2f}pp on the mean and "
            f"{median_pp:+.2f}pp on the median"
        )

    return "heterogeneous", (
        f"the shift is in the same direction in only {n_agree} of {n_pairs} donors "
        f"(sign test p={sign_test_p:.3g}); the cohort mean is {mean_pp:+.2f}pp against a median of "
        f"{median_pp:+.2f}pp, so treat this as a subgroup hypothesis rather than a cohort-wide "
        f"change"
    )


def paired_value_concordance(
    values: pd.DataFrame,
    donors: pd.Series,
    conditions: pd.Series,
    *,
    case: str,
    control: str,
    min_pairs: int = MIN_PAIRED_BLOCKS,
    item_label: str = "item",
    unit: str = "",
) -> pd.DataFrame:
    """
    Audit whether each item's paired change is shared across donors, on common support.

    The same audit as :func:`paired_abundance_concordance`, for any per-sample
    quantity rather than a composition: a communication edge's weight, a pathway
    score, a module mean. One thing is genuinely new here and it is not cosmetic.
    **A composition is always complete and an arbitrary measurement is not.** Every
    cell type has a share of every sample, even if that share is zero, so the
    compositional audit can take one donor count for the whole table. An edge weight
    exists only where both of its populations cleared a detection floor in that
    sample; a pathway score exists only where the pathway's genes were seen. So the
    donors that can be paired differ *per item*, and ``n_pairs`` is a column here
    rather than a scalar.

    Averaging over whatever is present is the failure this exists to prevent. If an
    item scored in nine disease arms and six normal arms, the difference of the two
    arm means compares nine donors against six, and the gap can be entirely the
    three donors who appear on one side only. So a donor enters an item's test only
    if the item is finite in **both** of that donor's arms, and the donors that were
    dropped for that reason are reported in ``n_donors_one_arm``: a large value there
    is a detection statement, and it often *is* the finding.

    Args:
        values: Samples (rows) × items (columns). ``NaN`` means "not measured for
            this sample", which is not the same as zero and is never treated as one.
        donors: Per-sample donor labels, indexed by ``values.index``.
        conditions: Per-sample condition labels, indexed by ``values.index``.
        case: Condition label treated as the case/disease arm.
        control: Condition label treated as the control arm.
        min_pairs: Donors with both arms required before a pattern is called. Applied
            per item, since the count is per item.
        item_label: Name for the first column, so a caller's table reads in its own
            vocabulary (``"partner"``, ``"factor"``, ``"lr_pair"``).
        unit: Optional unit appended to the effect sizes quoted in ``reason``.

    Returns:
        One row per column of ``values`` with the columns in
        :data:`VALUE_CONCORDANCE_COLUMNS`, ordered by ``sign_test_p`` then by
        descending absolute mean change. Items no donor could pair are kept, with
        ``n_pairs`` 0 and pattern ``underpowered`` — a dropped row would make a
        detection failure look like an item that was never asked about.

    Notes:
        The BH families are the items of this one call, one per test. A caller whose
        items span several independent questions should call once per question; a
        caller who pools them has widened the family and will say so in a note or not
        at all. ``design_floor_p`` is the item's own randomization floor from
        :func:`~cellquorum.stats.module_remodeling.randomization_floor`, computed on
        the samples that actually carried the item, so it moves with the missingness
        rather than describing the cohort the run started with.

        ``pattern`` is a verdict about donor agreement and nothing else: it is
        uncorrected, and an item can read ``consistent`` at an FDR of 0.13. Deciding
        that an item *changed* takes two things — a pattern of ``consistent`` and a
        family-corrected p below the bar — and the column to use for the second is
        ``sign_test_fdr_conservative``, not ``sign_test_fdr``. See
        :func:`_conservative_sign_p` for why the doubling is the honest yardstick when
        the direction was read off the same deltas.
    """

    from cellquorum.stats.module_remodeling import randomization_floor

    empty = pd.DataFrame(columns=[item_label, *VALUE_CONCORDANCE_COLUMNS[1:]])

    if values is None or values.empty or values.shape[1] == 0:
        return empty

    aligned_donors = donors.reindex(values.index).astype(str)
    aligned_conditions = conditions.reindex(values.index).astype(str)
    numeric = values.apply(pd.to_numeric, errors="coerce")

    # The donor blocks are resolved to row positions once, outside the item loop. A
    # pair-level family here runs to thousands of items, and re-grouping the design
    # per item turns a seconds-long call into a minutes-long one for no difference in
    # the answer.
    position = {sample: index for index, sample in enumerate(numeric.index)}
    blocks: list[tuple[str, np.ndarray, np.ndarray]] = []
    for donor, block in aligned_donors.groupby(aligned_donors):
        arms = {
            arm: np.array(
                [position[s] for s in block.index if aligned_conditions.get(s) == arm], dtype=int
            )
            for arm in (case, control)
        }
        if arms[case].size or arms[control].size:
            blocks.append((str(donor), arms[case], arms[control]))

    matrix = numeric.to_numpy(dtype=float)

    rows: list[dict] = []
    for index, item in enumerate(numeric.columns):
        column = matrix[:, index]
        deltas: list[float] = []
        case_levels: list[float] = []
        control_levels: list[float] = []
        carried_donors: list[str] = []
        carried_is_case: list[bool] = []
        n_one_arm = 0
        for donor, case_rows, control_rows in blocks:
            case_values = column[case_rows]
            case_values = case_values[np.isfinite(case_values)]
            control_values = column[control_rows]
            control_values = control_values[np.isfinite(control_values)]
            carried_donors.extend([donor] * (case_values.size + control_values.size))
            carried_is_case.extend([True] * case_values.size + [False] * control_values.size)
            if not case_values.size or not control_values.size:
                # One arm only: the donor is dropped for this item and counted, which
                # is the difference between a reported common-support rule and a
                # silent one.
                n_one_arm += 1 if (case_values.size or control_values.size) else 0
                continue
            # A donor with several samples in an arm has that arm averaged, so one
            # donor counts once toward consensus however many samples it contributed.
            case_levels.append(float(case_values.mean()))
            control_levels.append(float(control_values.mean()))
            deltas.append(case_levels[-1] - control_levels[-1])

        floor_p, _ = randomization_floor(carried_donors, carried_is_case)
        rows.append(
            _describe_item(
                str(item),
                np.asarray(deltas, dtype=float),
                np.asarray(case_levels, dtype=float),
                np.asarray(control_levels, dtype=float),
                n_one_arm=n_one_arm,
                design_floor_p=float(floor_p),
                min_pairs=min_pairs,
                unit=unit,
            )
        )

    table = pd.DataFrame(rows)
    table["sign_test_fdr"] = bh_fdr(table["sign_test_p"].to_numpy())
    table["sign_test_fdr_conservative"] = bh_fdr(table["sign_test_p_conservative"].to_numpy())
    table["wilcoxon_fdr"] = bh_fdr(table["wilcoxon_p"].to_numpy())
    table = table.sort_values(
        ["sign_test_p", "mean_delta"],
        key=lambda s: s.abs().mul(-1) if s.name == "mean_delta" else s,
    ).reset_index(drop=True)
    return table[list(VALUE_CONCORDANCE_COLUMNS)].rename(columns={"item": item_label})


def _describe_item(
    item: str,
    delta: np.ndarray,
    case_level: np.ndarray,
    control_level: np.ndarray,
    *,
    n_one_arm: int,
    design_floor_p: float,
    min_pairs: int,
    unit: str,
) -> dict:
    """Measure one item's donor-level agreement and name the pattern."""

    n_pairs = int(delta.size)
    suffix = f" {unit}" if unit else ""

    if n_pairs == 0:
        return {
            "item": item,
            "n_pairs": 0,
            "n_donors_one_arm": n_one_arm,
            "n_agree": 0,
            "direction": 0,
            "mean_case": float(np.mean(case_level)) if case_level.size else np.nan,
            "mean_control": float(np.mean(control_level)) if control_level.size else np.nan,
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "loo_mean_min": np.nan,
            "loo_mean_max": np.nan,
            "loo_sign_stable": False,
            "signs_agree": False,
            "sign_test_p": np.nan,
            "sign_test_p_conservative": np.nan,
            "wilcoxon_p": np.nan,
            "design_floor_p": design_floor_p,
            "pattern": "underpowered",
            "reason": (
                f"no donor carried this {'item' if not unit else 'value'} in both arms"
                + (f"; {n_one_arm} donor(s) had it in one arm only" if n_one_arm else "")
            ),
        }

    mean_delta = float(delta.mean())
    median_delta = float(np.median(delta))
    direction = int(np.sign(mean_delta))
    moved = delta[delta != 0.0]
    n_agree = int((np.sign(delta) == direction).sum()) if direction != 0 else 0

    # One-sided, for the reason given in ``_describe_cell_type``: the direction is
    # handed over from the mean rather than chosen here, and the only question is
    # whether the donors corroborate it.
    if direction != 0 and moved.size:
        agree_among_moved = int((np.sign(moved) == direction).sum())
        sign_test_p = float(
            stats.binomtest(agree_among_moved, moved.size, 0.5, alternative="greater").pvalue
        )
    else:
        sign_test_p = np.nan

    wilcoxon_p = np.nan
    if moved.size >= 1:
        try:
            wilcoxon_p, _ = _floor_p(float(stats.wilcoxon(delta).pvalue))
        except ValueError:
            wilcoxon_p = np.nan

    if n_pairs > 1:
        loo_means = (delta.sum() - delta) / (n_pairs - 1)
        loo_min, loo_max = float(loo_means.min()), float(loo_means.max())
        loo_sign_stable = bool(np.all(np.sign(loo_means) == direction)) if direction != 0 else False
    else:
        loo_min = loo_max = mean_delta
        loo_sign_stable = False

    signs_agree = bool(np.sign(median_delta) == direction) if direction != 0 else False

    pattern = _classify_pattern(
        n_pairs=n_pairs,
        min_pairs=min_pairs,
        direction=direction,
        signs_agree=signs_agree,
        loo_sign_stable=loo_sign_stable,
        sign_test_p=sign_test_p,
    )
    support = f"; {n_one_arm} donor(s) carried it in one arm only" if n_one_arm else ""

    if pattern == "underpowered":
        reason = (
            f"only {n_pairs} donor(s) carry this in both arms, below the {min_pairs} at which a "
            f"one-sided sign test can reach {SIGN_TEST_ALPHA:g} even on unanimous donors{support}"
        )
    elif direction == 0:
        reason = "the cohort mean change is exactly zero, so there is no direction to check"
    elif pattern == "consistent":
        reason = (
            f"the change is in the same direction in {n_agree} of {n_pairs} donors "
            f"(sign test p={sign_test_p:.3g}) and survives dropping any one donor, so the mean of "
            f"{mean_delta:+.3g}{suffix} describes the group{support}"
        )
    elif pattern == "single_donor_driven":
        reason = (
            f"{n_agree} of {n_pairs} donors move the same way (sign test p={sign_test_p:.3g}), but "
            f"removing one donor takes the mean across zero (leave-one-out range {loo_min:+.3g} to "
            f"{loo_max:+.3g}{suffix} against a mean of {mean_delta:+.3g}{suffix}), so one "
            f"dissenting donor sets the effect size{support}"
        )
    elif pattern == "direction_inconsistent":
        reason = (
            f"the mean and the typical donor disagree on the sign (mean {mean_delta:+.3g}{suffix} "
            f"against median {median_delta:+.3g}{suffix}, direction shared by only {n_agree} of "
            f"{n_pairs} donors), so the mean describes a few large movers rather than the "
            f"cohort{support}"
        )
    else:
        reason = (
            f"the change is in the same direction in only {n_agree} of {n_pairs} donors "
            f"(sign test p={sign_test_p:.3g}); the mean is {mean_delta:+.3g}{suffix} against a "
            f"median of {median_delta:+.3g}{suffix}, so treat this as a subgroup hypothesis rather "
            f"than a cohort-wide change{support}"
        )

    return {
        "item": item,
        "n_pairs": n_pairs,
        "n_donors_one_arm": n_one_arm,
        "n_agree": n_agree,
        "direction": direction,
        "mean_case": float(case_level.mean()),
        "mean_control": float(control_level.mean()),
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "loo_mean_min": loo_min,
        "loo_mean_max": loo_max,
        "loo_sign_stable": loo_sign_stable,
        "signs_agree": signs_agree,
        "sign_test_p": sign_test_p,
        "sign_test_p_conservative": _conservative_sign_p(sign_test_p),
        "wilcoxon_p": wilcoxon_p,
        "design_floor_p": design_floor_p,
        "pattern": pattern,
        "reason": reason,
    }


def qualify_abundance_calls(
    effects: pd.DataFrame,
    concordance: pd.DataFrame,
    *,
    cell_type_col: str = "cell_type",
    called_col: str = "credible_effect",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Attach the donor-level pattern to an abundance table and flag fragile calls.

    Args:
        effects: Per-cell-type abundance results from any method.
        concordance: Output of :func:`paired_abundance_concordance`.
        cell_type_col: Cell-type column in ``effects``.
        called_col: Boolean column in ``effects`` marking a positive call.

    Returns:
        ``(annotated, notes)`` — ``effects`` with the concordance columns merged in
        (unchanged when either frame is empty or the cell-type column is absent),
        and one note per positive call whose donor pattern is not ``consistent``.
        Notes are the point: a call that the donors do not support should be
        visible in the run summary, not only in a column of a CSV.
    """

    if effects is None or effects.empty or concordance is None or concordance.empty:
        return (effects if effects is not None else pd.DataFrame()), []
    if cell_type_col not in effects.columns:
        return effects, []

    carried = [
        "n_pairs",
        "n_agree",
        "mean_delta_pp",
        "median_delta_pp",
        "loo_sign_stable",
        "signs_agree",
        "sign_test_p",
        "sign_test_fdr",
        "wilcoxon_p",
        "pattern",
        "reason",
    ]
    available = [column for column in carried if column in concordance.columns]

    # Join labels as labels. The two frames arrive from different places -- one from
    # a model's effect table, one from the count matrix -- and a label that looks
    # numeric can be float64 on one side and object on the other, which pandas
    # refuses to merge ("You are trying to merge on float64 and object columns")
    # rather than joining. Normalizing both keys through the one label helper makes
    # the join independent of which route the frame took.
    right = concordance[["cell_type", *available]].rename(columns={"cell_type": cell_type_col})
    effects = effects.copy()
    right = right.copy()
    effects[cell_type_col] = as_label_strings(effects[cell_type_col])
    right[cell_type_col] = as_label_strings(right[cell_type_col])

    annotated = effects.merge(
        right,
        on=cell_type_col,
        how="left",
        suffixes=("", "_concordance"),
    )

    notes: list[str] = []
    if called_col not in annotated.columns:
        return annotated, notes

    called = annotated[annotated[called_col].astype(bool)]
    for _, row in called.iterrows():
        pattern = row.get("pattern")
        if not isinstance(pattern, str) or pattern == "consistent":
            continue
        notes.append(
            f"{row[cell_type_col]}: called abundance change is {pattern} — {row['reason']}"
        )

    return annotated, notes


def mark_called(
    table: pd.DataFrame,
    *,
    alpha: float = SIGN_TEST_ALPHA,
    fdr_col: str = "sign_test_fdr_conservative",
    pattern_col: str = "pattern",
    consistent: str = "consistent",
    called_col: str = "called",
) -> pd.DataFrame:
    """Add one boolean that clears **both** hurdles, so no report can quote only one.

    ``pattern`` and the FDR answer different questions, and conflating them is how a result at
    an FDR of 0.13 ends up written up as a finding. ``pattern`` is an uncorrected verdict about
    whether the donors agree on a direction — it says nothing about how many other items were
    asked the same question. The FDR is a statement about the family and nothing about whether
    the donors agreed: an item can clear it on one outlying donor.

    Every note, caption and headline should be written off this one column rather than off
    either input, because a column is harder to quote selectively than a pair of thresholds a
    reader has to recombine.

    The default FDR column is the *conservative* one — the doubled one-sided sign p — because
    the direction being tested was read off the same donor deltas. See
    :func:`_conservative_sign_p`.

    Args:
        table: Output of :func:`paired_value_concordance` or
            :func:`paired_abundance_concordance`, optionally after
            :func:`~cellquorum.stats.module_remodeling.recorrect_within_family`.
        alpha: FDR level.
        fdr_col: Corrected p-value column to gate on.
        pattern_col: Donor-agreement verdict column.
        consistent: The value of ``pattern_col`` that counts as agreement.
        called_col: Name of the boolean to add.

    Returns:
        A copy with ``called_col`` added. When either input column is missing the column is
        added as all-``False`` rather than omitted, so a downstream figure that reads it fails
        loudly on an empty panel instead of silently on a missing key.
    """
    out = table.copy()
    if pattern_col not in out.columns or fdr_col not in out.columns:
        out[called_col] = False
        return out
    out[called_col] = (out[pattern_col] == consistent) & (
        pd.to_numeric(out[fdr_col], errors="coerce") < float(alpha)
    )
    return out


def donor_unanimous(
    table: pd.DataFrame,
    *,
    min_pairs: int = MIN_PAIRED_BLOCKS,
    pairs_col: str = "n_pairs",
    agree_col: str = "n_agree",
) -> pd.Series:
    """Which items moved the same way in **every** donor that could be paired on them.

    Unanimity is the one summary of a paired cohort with no effect size in it, which makes it
    the right thing to put beside an effect size — and it is worthless unless it can be lost.
    ``n_agree >= n_pairs`` is *vacuously true* at ``0/0``, and nearly free at ``3/3``: an item
    detected in three donors that happens to rise in all three is not evidence that the cohort
    agrees, it is evidence that three is a small number. Reported without a floor, the
    strongest-looking rows in a table are the ones with the least data behind them, and a figure
    that selects rows this way will draw empty ones.

    So the floor is the same :data:`~cellquorum.stats.depth_confounding.MIN_PAIRED_BLOCKS` used
    everywhere else in the house — below six pairs a signed-rank test cannot reach 0.05 at all —
    which means "unanimous" denotes one thing in every table and every figure of a run.

    Args:
        table: Any frame carrying the two count columns.
        min_pairs: Fewest paired donors an item needs before unanimity is claimable.
        pairs_col: Donors the item could be paired on.
        agree_col: Of those, how many moved in the cohort's direction.

    Returns:
        Boolean Series aligned to ``table.index``. Missing counts are ``False``.
    """
    if pairs_col not in table.columns or agree_col not in table.columns:
        return pd.Series(False, index=table.index)
    pairs = pd.to_numeric(table[pairs_col], errors="coerce")
    agree = pd.to_numeric(table[agree_col], errors="coerce")
    return ((pairs >= int(min_pairs)) & (agree >= pairs)).fillna(False)


__all__ = [
    "CONCORDANCE_COLUMNS",
    "SIGN_TEST_ALPHA",
    "VALUE_CONCORDANCE_COLUMNS",
    "donor_unanimous",
    "mark_called",
    "paired_abundance_concordance",
    "paired_value_concordance",
    "qualify_abundance_calls",
]
