"""Does a per-cell metric's condition effect survive adjustment for library depth?

Every continuous per-cell readout the engine produces — pseudotime, potency,
module scores, any ``obs`` column a stage writes — is a candidate for the same
failure: sequencing depth differs between arms, the metric is a function of
depth, and the resulting "condition effect" is a library-size readout wearing a
biological caption. It is not exotic. Diffusion pseudotime is computed on a
neighbour graph built from a depth-sensitive embedding, and CytoTRACE's
foundational signal is gene-count-correlated by construction.

The failure that motivated this module: on a lymphatic-endothelial slice,
``dpt_pseudotime`` moved with condition in 9/9 donors at p=0.004, and correlated
with ``n_genes_by_counts`` at Spearman rho = -0.856. After adjustment the effect
was -0.004 with 4/9 donors sharing the sign at p=0.76. Nothing survived. The
same object's AUCell-based module index correlated with depth at rho = +0.036 and
was unchanged by adjustment. One of those belonged in a paper and the other did
not, and no property of either name told you which.

Confounding is a three-way condition, so this module tests all three legs rather
than flagging correlation alone:

1. **depth ~ condition** — if the arms are depth-balanced, no metric can be
   confounded by depth no matter how strongly it tracks depth. This is a property
   of the dataset, evaluated once, and it gates everything else. Reporting
   ``rho`` without it manufactures alarm on balanced cohorts.
2. **metric ~ depth** — the coupling strength, reported as Spearman rho so a
   monotone non-linear dependence still registers.
3. **the adjusted effect** — the metric residualised on depth, re-tested with the
   same donor-paired machinery, so raw and adjusted differ only in the
   adjustment.

Both entry points here are audits: they never rewrite a result, they return a
verdict per metric with the evidence beside it. The verdict vocabulary is shared
(``depth_balanced``, ``robust``, ``attenuated``, ``depth_driven``,
``depth_masked``, ``no_raw_effect``, ``insufficient_pairs``) so a caller can
filter on one column.

``depth_masked`` is the one verdict that adds a result rather than removing one,
and it exists because confounding has a direction. Where a metric rises with
depth and the deeper arm is the case arm, depth pushes the case mean up and
cancels part of a genuine *fall* — so removing depth can expose an effect the
unadjusted test could not see. Filing that under ``no_raw_effect`` would hide the
one row the audit found rather than protected. It is still a lead and not a call:
the unadjusted test is the one a declared family was corrected over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from cellquorum.stats.module_remodeling import (
    _floor_p,
    _is_negligible_spread,
    bh_fdr,
)

# The house floor for a donor-paired test. Below six pairs a signed-rank test
# cannot reach 0.05 at all (2/2**5 = 0.0625), so an "adjusted effect is no longer
# significant" verdict would be a statement about the sample size rather than
# about depth.
MIN_PAIRED_BLOCKS = 6

# Below this share of the raw effect the adjusted effect is not "reduced", it is
# gone: the depth term explained the result. Chosen so that the motivating
# failure (4.8% retained) separates from a genuinely attenuated but surviving
# effect (44.7% retained) without either landing near the boundary.
DEFAULT_COLLAPSE_FRACTION = 0.25

# Coupling strong enough to be worth naming in the report even when the cohort is
# depth-balanced and therefore safe. |rho| >= 0.5 is a metric that is at least
# half depth by rank; it is fine here and dangerous in the next cohort.
NOTABLE_COUPLING_RHO = 0.5

# Verdicts that mean there was no unadjusted effect, and therefore no denominator worth
# dividing by. "Fraction of the effect retained" presupposes an effect: where the raw
# contrast was indistinguishable from zero the ratio is not a fraction, it is noise
# amplification, and it prints largest exactly where it means least -- a metric whose raw
# delta is -0.0019 and adjusted delta -0.0116 comes out as "620% retained". So the column
# is NaN on these rows rather than a number the reader has to know to distrust. The
# adjusted and raw deltas are both still there for anyone who wants the ratio anyway.
_NO_RAW_CLAIM = frozenset({"no_raw_effect", "depth_masked", "insufficient_pairs"})


def _same_cells(left: pd.Index, right: pd.Index) -> bool:
    """Do these two indexes name the same cells in the same order?

    Not ``Index.equals``, which also compares dtypes and so rejects a correctly aligned
    pair. AnnData holds ``obs_names`` as a pandas ``string`` index, and a per-cell frame
    built from those same names — through a scoring library, a parquet round trip, or a
    plain ``DataFrame`` constructor — comes back as ``object``. The barcodes are identical
    and the alignment is perfect, and a guard that refuses it sends a caller off to fix
    something that is not broken. Length and element order are what the callers below
    actually depend on, so that is what is checked.
    """
    if len(left) != len(right):
        return False
    if left.equals(right):
        return True
    return bool(np.array_equal(left.to_numpy(dtype=object), right.to_numpy(dtype=object)))


def _paired_by_sample(
    values: np.ndarray,
    donors: pd.Series,
    conditions: pd.Series,
    case: str,
    control: str,
) -> tuple[pd.DataFrame, str]:
    """Collapse cells to one value per donor-condition sample and pair on donor.

    Returns the wide frame (index donor, columns ``case``/``control``, complete
    pairs only) and a reason string, non-empty only when something was dropped.
    """
    frame = pd.DataFrame(
        {"donor": donors.to_numpy(), "condition": conditions.to_numpy(), "value": values}
    )
    frame = frame[frame["condition"].isin([case, control])].dropna(subset=["value"])
    wide = frame.groupby(["donor", "condition"])["value"].mean().unstack()
    for arm in (case, control):
        if arm not in wide.columns:
            wide[arm] = np.nan
    complete = wide[[case, control]].dropna()
    dropped = int(wide.shape[0] - complete.shape[0])
    reason = f"{dropped} donor(s) lacked one arm and were dropped" if dropped else ""
    return complete, reason


def _paired_test(wide: pd.DataFrame, case: str, control: str) -> dict[str, float | str]:
    """Paired t and Wilcoxon on one value per donor per arm.

    Both are reported because they fail differently: the t-test is undefined when
    the paired differences have no spread, and the signed-rank test saturates at
    ``2/2**n`` and cannot distinguish a large effect from an enormous one.
    """
    diff = (wide[case] - wide[control]).to_numpy(dtype=float)
    n = int(diff.size)
    out: dict[str, float | str] = {
        "n_pairs": n,
        "mean_case": float(wide[case].mean()),
        "mean_control": float(wide[control].mean()),
        "delta": float(np.mean(diff)) if n else np.nan,
        "n_donors_positive": int(np.sum(diff > 0)),
        "t_p": np.nan,
        "wilcoxon_p": np.nan,
        "reason": "",
    }
    if n < 2:
        out["reason"] = f"only {n} complete pair(s); no paired test is defined"
        return out

    explanations: list[str] = []
    spread = float(np.std(diff, ddof=1))
    if _is_negligible_spread(spread, diff):
        explanations.append(
            "the paired differences have no spread beyond floating-point residue, so the t "
            "statistic is undefined rather than infinitely significant"
        )
    else:
        _, raw_t = stats.ttest_rel(wide[case], wide[control])
        t_p, floor_reason = _floor_p(float(raw_t))
        out["t_p"] = t_p
        if floor_reason:
            explanations.append(floor_reason)

    if np.allclose(diff, 0.0):
        explanations.append("every paired difference is zero, so the signed-rank test is undefined")
    else:
        _, raw_w = stats.wilcoxon(wide[case], wide[control])
        w_p, floor_reason = _floor_p(float(raw_w))
        out["wilcoxon_p"] = w_p
        if floor_reason:
            explanations.append(floor_reason)

    out["reason"] = "; ".join(explanations)
    return out


def _varies(values: np.ndarray) -> bool:
    """Does ``values`` take more than one value? A constant has no rank correlation."""
    return bool(values.size) and bool(np.ptp(values) > 0.0)


def _center_within(values: np.ndarray, groups: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Subtract each group's mean from ``values``, over the ``finite`` cells only."""
    centered = np.full(values.shape, np.nan, dtype=float)
    frame = pd.DataFrame({"g": groups[finite], "v": values[finite]})
    centered[finite] = (frame["v"] - frame.groupby("g")["v"].transform("mean")).to_numpy()
    return centered


def _within_sample_depth_slope(
    values: np.ndarray, log_depth: np.ndarray, samples: np.ndarray, finite: np.ndarray
) -> float:
    """Slope of ``values`` on ``log_depth`` estimated *within* donor-condition samples.

    The pooled slope is the wrong quantity and using it silently destroys real
    findings. If the arms differ in depth and the metric has a genuine condition
    effect, then depth and the metric are correlated *through condition* even when
    depth has no influence on the metric whatsoever. Residualising on the pooled
    slope then subtracts part of the biology: on a fixture where the metric was
    constructed independent of depth, pooled adjustment removed 40% of a known
    true effect.

    Centering both variables within each donor-condition sample removes every
    between-sample difference — condition and donor alike — so what remains is
    only the cell-to-cell relationship inside a sample. That is the sole part of
    the association that depth could actually be causing. This is the slope an
    ANCOVA with sample intercepts would fit, obtained directly.
    """
    x = _center_within(log_depth, samples, finite)
    y = _center_within(values, samples, finite)
    usable = np.isfinite(x) & np.isfinite(y)
    denominator = float(np.sum(x[usable] ** 2))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(x[usable] * y[usable]) / denominator)


def _residualise(values: np.ndarray, depth: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Remove the within-sample log-depth trend from ``values``.

    ``log1p`` because depth spans an order of magnitude and the dependence of
    scores on it is closer to log-linear than linear. This is a monotone
    single-slope adjustment, so it will under-correct a strongly non-monotone
    dependence — which is why :func:`depth_stratified_abundance` provides an
    assumption-free alternative and why the coupling is reported beside the
    adjusted effect rather than replaced by it.
    """
    finite = np.isfinite(values) & np.isfinite(depth)
    out = np.full(values.shape, np.nan, dtype=float)
    if finite.sum() < 3:
        return out
    log_depth = np.full(depth.shape, np.nan, dtype=float)
    log_depth[finite] = np.log1p(depth[finite])
    slope = _within_sample_depth_slope(values, log_depth, samples, finite)
    reference = float(np.mean(log_depth[finite]))
    out[finite] = values[finite] - slope * (log_depth[finite] - reference)
    return out


def _within_sample_rho(
    values: np.ndarray, depth: np.ndarray, samples: np.ndarray, finite: np.ndarray
) -> float:
    """Mean per-sample Spearman rho between ``values`` and ``depth``.

    Reported alongside the pooled rho because the two answer different questions
    and only this one bears on confounding. A metric with a real condition effect
    inherits a pooled rho from the depth imbalance and looks depth-coupled when it
    is not; averaging the correlation computed separately inside each sample
    cannot pick that up.
    """
    frame = pd.DataFrame({"g": samples[finite], "v": values[finite], "d": depth[finite]})
    rhos: list[float] = []
    for _, block in frame.groupby("g"):
        if block.shape[0] < 5 or block["v"].nunique() < 2 or block["d"].nunique() < 2:
            continue
        rho, _ = stats.spearmanr(block["v"], block["d"])
        if np.isfinite(rho):
            rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else np.nan


def _verdict(
    *,
    depth_is_confounded: bool,
    n_pairs: int,
    raw_p: float,
    adjusted_p: float,
    raw_delta: float,
    adjusted_delta: float,
    alpha: float,
    collapse_fraction: float,
    min_pairs: int,
) -> tuple[str, str]:
    """Classify one metric. Returns ``(verdict, reason)``."""
    if n_pairs < min_pairs:
        return (
            "insufficient_pairs",
            f"{n_pairs} complete pair(s) is below min_pairs={min_pairs}; a non-significant "
            f"adjusted effect here would describe the sample size, not depth",
        )
    if not depth_is_confounded:
        return (
            "depth_balanced",
            "depth does not differ by condition in this cohort, so no depth-coupled metric can "
            "be confounded by it; the adjusted columns are reported for completeness",
        )
    if not np.isfinite(raw_p) or raw_p >= alpha:
        # An audit that can only ever remove a claim is under-reading its own output.
        # Confounding is directional: where a metric rises with depth and the deeper
        # arm is the case arm, depth pushes the case mean *up*, which cancels part of
        # a genuine fall. Removing depth can therefore expose an effect the unadjusted
        # test could not see, and filing that under "no claim to audit" hides the one
        # row the audit *found* rather than protected. It is still a lead and not a
        # call — the unadjusted test is the one any declared family was corrected
        # over — which is why it gets its own verdict instead of ``robust``.
        if (
            np.isfinite(raw_p)
            and np.isfinite(adjusted_p)
            and adjusted_p < alpha
            and np.isfinite(adjusted_delta)
        ):
            return (
                "depth_masked",
                f"there is no unadjusted effect (p={raw_p:.3g}) and there is an adjusted one "
                f"({adjusted_delta:+.4g}, p={adjusted_p:.3g}): depth was moving this metric "
                f"against its condition effect and hiding it. Report as a lead, not as a "
                f"finding — the unadjusted test is the one any declared family was corrected "
                f"over — and check whether the direction was predicted in advance",
            )
        return (
            "no_raw_effect",
            "the unadjusted condition effect is not significant, so there is no claim to audit",
        )
    if not np.isfinite(adjusted_delta) or not np.isfinite(adjusted_p):
        return (
            "depth_driven",
            "the depth-adjusted effect could not be estimated, so the unadjusted effect is "
            "unsupported",
        )

    retained = abs(adjusted_delta) / abs(raw_delta) if raw_delta else np.nan
    flipped = np.sign(adjusted_delta) != np.sign(raw_delta)
    if flipped:
        return (
            "depth_driven",
            f"the effect reverses sign under depth adjustment ({raw_delta:+.4g} -> "
            f"{adjusted_delta:+.4g}), so its direction was set by depth",
        )
    if np.isfinite(retained) and retained < collapse_fraction:
        return (
            "depth_driven",
            f"depth adjustment removes {(1 - retained) * 100:.0f}% of the effect "
            f"({raw_delta:+.4g} -> {adjusted_delta:+.4g}, adjusted p={adjusted_p:.3g}); "
            f"this metric is largely a library-size readout and must not be reported as biology",
        )
    if adjusted_p >= alpha:
        return (
            "attenuated",
            f"the effect keeps its sign and {retained * 100:.0f}% of its magnitude but is no "
            f"longer significant after adjustment (p={adjusted_p:.3g}); report it as suggestive "
            f"with the depth caveat, not as a finding",
        )
    return (
        "robust",
        "",
    )


def depth_confound_audit(
    metrics: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    donor_col: str,
    condition_col: str,
    case: str,
    control: str,
    depth_col: str,
    alpha: float = 0.05,
    fdr_method: str = "fdr_bh",
    min_pairs: int = MIN_PAIRED_BLOCKS,
    collapse_fraction: float = DEFAULT_COLLAPSE_FRACTION,
) -> pd.DataFrame:
    """Audit each continuous per-cell metric for depth confounding.

    Every test is donor-paired on one value per donor-condition sample, so no
    row is a per-cell test over many cells from few people. Raw and adjusted
    effects use identical machinery and differ only in whether the metric was
    residualised on ``log1p(depth)`` first, which is what makes the comparison
    between them meaningful.

    Args:
        metrics: Per-cell continuous metrics, one column per metric.
        metadata: Per-cell metadata aligned to ``metrics.index``; must contain
            ``donor_col``, ``condition_col`` and ``depth_col``.
        donor_col: Column holding the donor / pairing block.
        condition_col: Column holding the condition.
        case: Condition level treated as case.
        control: Condition level treated as control.
        depth_col: Column holding the depth covariate, e.g.
            ``n_genes_by_counts`` or ``total_counts``. Gene count is usually the
            better choice: it is the quantity most scores are monotone in, and it
            saturates less than UMI count.
        alpha: Significance level used for the verdicts.
        fdr_method: Passed to :func:`~cellquorum.stats.bh_fdr`. Raw and adjusted
            p-values are corrected as two separate families, because they are two
            tests of the same hypotheses rather than one family of size 2n.
        min_pairs: Below this many complete donor pairs the audit declines to
            reach a verdict.
        collapse_fraction: Share of the raw effect below which the adjusted
            effect counts as gone rather than reduced.

    Returns:
        One row per metric, ordered as ``metrics.columns``, with columns:
        ``metric``, ``n_cells``, ``spearman_rho_vs_depth``,
        ``spearman_p_vs_depth``, ``within_sample_rho_vs_depth``,
        ``depth_delta``, ``depth_n_donors_positive``,
        ``depth_t_p``, ``depth_wilcoxon_p``, ``depth_is_confounded``,
        ``n_pairs``, ``raw_mean_case``, ``raw_mean_control``, ``raw_delta``,
        ``raw_n_donors_positive``, ``raw_t_p``, ``raw_wilcoxon_p``, ``raw_fdr``,
        ``adjusted_delta``, ``adjusted_n_donors_positive``, ``adjusted_t_p``,
        ``adjusted_wilcoxon_p``, ``adjusted_fdr``, ``delta_retained_fraction``,
        ``verdict``, ``reason``. ``reason`` is non-empty for every row that is
        not a clean ``robust``. ``verdict`` is one of ``depth_balanced``,
        ``robust``, ``attenuated``, ``depth_driven``, ``depth_masked``,
        ``no_raw_effect``, ``insufficient_pairs`` — see the module docstring, and
        note that ``depth_masked`` reports a *gained* lead rather than a lost
        claim. ``delta_retained_fraction`` is NaN wherever there was no
        unadjusted effect to retain a fraction of, since a ratio to a contrast
        that was indistinguishable from zero is not a fraction; the raw and
        adjusted deltas are both present for anyone who wants it anyway.

    Raises:
        KeyError: If a required metadata column is absent.
        ValueError: If ``metrics`` and ``metadata`` are not aligned.
    """
    for column in (donor_col, condition_col, depth_col):
        if column not in metadata.columns:
            raise KeyError(f"metadata has no column {column!r}")
    if not _same_cells(metrics.index, metadata.index):
        raise ValueError("metrics and metadata must share an index; align them before calling")

    depth = pd.to_numeric(metadata[depth_col], errors="coerce").to_numpy(dtype=float)
    donors = metadata[donor_col].astype(str)
    conditions = metadata[condition_col].astype(str)

    # Leg 1, evaluated once: is depth itself associated with condition? Without
    # this there is nothing for depth to confound.
    depth_wide, depth_pair_reason = _paired_by_sample(depth, donors, conditions, case, control)
    depth_test = (
        _paired_test(depth_wide, case, control)
        if not depth_wide.empty
        else {
            "n_pairs": 0,
            "delta": np.nan,
            "n_donors_positive": 0,
            "t_p": np.nan,
            "wilcoxon_p": np.nan,
            "reason": "no complete donor pairs",
        }
    )
    depth_p = float(depth_test["t_p"])
    depth_is_confounded = bool(
        depth_test["n_pairs"] >= min_pairs and np.isfinite(depth_p) and depth_p < alpha
    )

    samples = (donors + "|" + conditions).to_numpy()

    rows: list[dict[str, object]] = []
    for metric in metrics.columns:
        values = pd.to_numeric(metrics[metric], errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(values) & np.isfinite(depth)

        rho, rho_p = (np.nan, np.nan)
        within_rho = np.nan
        # A constant metric has no correlation with anything -- which is the honest answer, not
        # an error. It happens whenever the metrics are genes rather than scores, because a gene
        # undetected in one group's cells is an all-zero column, so scipy's warning would fire
        # once per such column and bury the run's real output.
        if usable.sum() >= 3 and _varies(values[usable]) and _varies(depth[usable]):
            rho, rho_p = stats.spearmanr(values[usable], depth[usable])
            within_rho = _within_sample_rho(values, depth, samples, usable)

        raw_wide, raw_pair_reason = _paired_by_sample(values, donors, conditions, case, control)
        raw = (
            _paired_test(raw_wide, case, control)
            if not raw_wide.empty
            else _paired_test(pd.DataFrame({case: [], control: []}), case, control)
        )

        adjusted_values = _residualise(values, depth, samples)
        adj_wide, _ = _paired_by_sample(adjusted_values, donors, conditions, case, control)
        adjusted = (
            _paired_test(adj_wide, case, control)
            if not adj_wide.empty
            else _paired_test(pd.DataFrame({case: [], control: []}), case, control)
        )

        raw_delta = float(raw["delta"]) if np.isfinite(float(raw["delta"])) else np.nan
        adj_delta = float(adjusted["delta"]) if np.isfinite(float(adjusted["delta"])) else np.nan
        retained = abs(adj_delta) / abs(raw_delta) if raw_delta else np.nan

        verdict, verdict_reason = _verdict(
            depth_is_confounded=depth_is_confounded,
            n_pairs=int(raw["n_pairs"]),
            raw_p=float(raw["t_p"]),
            adjusted_p=float(adjusted["t_p"]),
            raw_delta=raw_delta,
            adjusted_delta=adj_delta,
            alpha=alpha,
            collapse_fraction=collapse_fraction,
            min_pairs=min_pairs,
        )

        notes = [n for n in (verdict_reason, raw_pair_reason, str(raw["reason"])) if n]
        if (
            verdict == "depth_balanced"
            and np.isfinite(within_rho)
            and abs(within_rho) >= NOTABLE_COUPLING_RHO
        ):
            notes.append(
                f"within-sample coupling to {depth_col} is strong (rho={within_rho:+.3f}); this "
                f"metric is safe in this cohort only because the arms are depth-balanced, and "
                f"would not be in one that is not"
            )
        # "Depth is balanced" rests on a paired test over a handful of donors, which
        # cannot rule out an imbalance too small to detect. So when the gate passes
        # a metric whose effect nonetheless moves under adjustment, say so rather
        # than let depth_balanced read as an all-clear on that specific claim.
        if (
            verdict == "depth_balanced"
            and np.isfinite(float(raw["t_p"]))
            and float(raw["t_p"]) < alpha
            and np.isfinite(retained)
            and (
                retained < collapse_fraction
                or not (np.isfinite(float(adjusted["t_p"])) and float(adjusted["t_p"]) < alpha)
            )
        ):
            notes.append(
                f"the arms are depth-balanced, yet adjustment still removes "
                f"{(1 - retained) * 100:.0f}% of this effect (adjusted p="
                f"{float(adjusted['t_p']):.3g}); a depth imbalance too small for a "
                f"{int(depth_test['n_pairs'])}-pair test to detect would account for that, so do "
                f"not treat this particular effect as cleared"
            )

        rows.append(
            {
                "metric": metric,
                "n_cells": int(usable.sum()),
                "spearman_rho_vs_depth": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p_vs_depth": float(rho_p) if np.isfinite(rho_p) else np.nan,
                "within_sample_rho_vs_depth": float(within_rho)
                if np.isfinite(within_rho)
                else np.nan,
                "depth_delta": float(depth_test["delta"]),
                "depth_n_donors_positive": int(depth_test["n_donors_positive"]),
                "depth_t_p": float(depth_test["t_p"]),
                "depth_wilcoxon_p": float(depth_test["wilcoxon_p"]),
                "depth_is_confounded": depth_is_confounded,
                "n_pairs": int(raw["n_pairs"]),
                "raw_mean_case": float(raw["mean_case"]),
                "raw_mean_control": float(raw["mean_control"]),
                "raw_delta": raw_delta,
                "raw_n_donors_positive": int(raw["n_donors_positive"]),
                "raw_t_p": float(raw["t_p"]),
                "raw_wilcoxon_p": float(raw["wilcoxon_p"]),
                "adjusted_delta": adj_delta,
                "adjusted_n_donors_positive": int(adjusted["n_donors_positive"]),
                "adjusted_t_p": float(adjusted["t_p"]),
                "adjusted_wilcoxon_p": float(adjusted["wilcoxon_p"]),
                "delta_retained_fraction": float(retained)
                if np.isfinite(retained) and verdict not in _NO_RAW_CLAIM
                else np.nan,
                "verdict": verdict,
                "reason": "; ".join(notes),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["raw_fdr"] = bh_fdr(table["raw_t_p"].to_numpy(), method=fdr_method)
    table["adjusted_fdr"] = bh_fdr(table["adjusted_t_p"].to_numpy(), method=fdr_method)
    ordered = [
        "metric",
        "n_cells",
        "spearman_rho_vs_depth",
        "spearman_p_vs_depth",
        "within_sample_rho_vs_depth",
        "depth_delta",
        "depth_n_donors_positive",
        "depth_t_p",
        "depth_wilcoxon_p",
        "depth_is_confounded",
        "n_pairs",
        "raw_mean_case",
        "raw_mean_control",
        "raw_delta",
        "raw_n_donors_positive",
        "raw_t_p",
        "raw_wilcoxon_p",
        "raw_fdr",
        "adjusted_delta",
        "adjusted_n_donors_positive",
        "adjusted_t_p",
        "adjusted_wilcoxon_p",
        "adjusted_fdr",
        "delta_retained_fraction",
        "verdict",
        "reason",
    ]
    return table[ordered]


def depth_stratified_abundance(
    labels: pd.Series,
    metadata: pd.DataFrame,
    *,
    donor_col: str,
    condition_col: str,
    case: str,
    control: str,
    depth_col: str,
    n_bins: int = 5,
    alpha: float = 0.05,
    fdr_method: str = "fdr_bh",
    min_pairs: int = MIN_PAIRED_BLOCKS,
) -> pd.DataFrame:
    """Re-test each label's donor-paired abundance shift inside depth strata.

    A compositional claim has its own version of the depth problem, and
    residualisation cannot address it: cluster membership is categorical, and if
    a cluster is largely a depth stratum then an arm with deeper cells will
    appear to gain it. The assumption-free check is to hold depth roughly fixed —
    bin all cells by global depth quantile and re-run the paired proportion test
    within each bin, where proportions are taken among that donor-sample's cells
    *in that bin*. A shift that is a depth artefact loses its sign consistency
    inside bins; a real one keeps it in every stratum.

    Strata are cut on global depth quantiles rather than per-sample ones so the
    same bin means the same depth range in both arms, which is the entire point.

    Args:
        labels: Per-cell cluster / state label. Cells with a null label are
            excluded from both numerator and denominator.
        metadata: Per-cell metadata aligned to ``labels.index``.
        donor_col: Column holding the donor / pairing block.
        condition_col: Column holding the condition.
        case: Condition level treated as case.
        control: Condition level treated as control.
        depth_col: Column holding the depth covariate.
        n_bins: Number of global depth quantile strata. Fewer, larger bins are
            usually better: the paired test inside a bin has the same donor count
            but fewer cells, so over-binning converts a real shift into a row of
            underpowered nulls.
        alpha: Significance level used for the per-stratum significance counts.
        fdr_method: Passed to :func:`~cellquorum.stats.bh_fdr`, applied across
            labels within each stratum.
        min_pairs: Below this many complete donor pairs no verdict is reached.

    Returns:
        One row per (label, stratum) with ``stratum == "all"`` for the unstratified
        test, plus columns ``n_pairs``, ``mean_control``, ``mean_case``,
        ``delta``, ``n_donors_positive``, ``t_p``, ``wilcoxon_p``, ``fdr``,
        ``reason``; and, on the ``all`` rows only, the summary columns
        ``n_strata``, ``n_strata_same_sign``, ``n_strata_significant`` and
        ``verdict``. Proportions are fractions, not percentages.

    Raises:
        KeyError: If a required metadata column is absent.
        ValueError: If ``labels`` and ``metadata`` are not aligned.
    """
    for column in (donor_col, condition_col, depth_col):
        if column not in metadata.columns:
            raise KeyError(f"metadata has no column {column!r}")
    if not _same_cells(labels.index, metadata.index):
        raise ValueError("labels and metadata must share an index; align them before calling")

    depth = pd.to_numeric(metadata[depth_col], errors="coerce")
    frame = pd.DataFrame(
        {
            "label": labels.astype("object"),
            "donor": metadata[donor_col].astype(str),
            "condition": metadata[condition_col].astype(str),
            "depth": depth,
        }
    )
    frame = frame[frame["condition"].isin([case, control])]
    frame = frame[frame["label"].notna() & frame["depth"].notna()]

    # Duplicate quantile edges (a depth distribution concentrated on few values)
    # would raise; dropping duplicate edges yields fewer strata, which is the
    # honest outcome and is reported via n_strata.
    try:
        frame["stratum"] = pd.qcut(frame["depth"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        frame["stratum"] = 0

    label_values = sorted(pd.unique(frame["label"]), key=str)
    strata = sorted(int(s) for s in pd.unique(frame["stratum"].dropna()))

    def rows_for(subset: pd.DataFrame, stratum_name: str) -> list[dict[str, object]]:
        totals = subset.groupby(["donor", "condition"]).size()
        out: list[dict[str, object]] = []
        for label in label_values:
            counts = subset[subset["label"] == label].groupby(["donor", "condition"]).size()
            proportion = (counts.reindex(totals.index, fill_value=0) / totals).unstack()
            for arm in (case, control):
                if arm not in proportion.columns:
                    proportion[arm] = np.nan
            complete = proportion[[case, control]].dropna()
            if complete.empty:
                out.append(
                    {
                        "label": str(label),
                        "stratum": stratum_name,
                        "n_pairs": 0,
                        "mean_control": np.nan,
                        "mean_case": np.nan,
                        "delta": np.nan,
                        "n_donors_positive": 0,
                        "t_p": np.nan,
                        "wilcoxon_p": np.nan,
                        "reason": "no complete donor pairs in this stratum",
                    }
                )
                continue
            test = _paired_test(complete, case, control)
            out.append(
                {
                    "label": str(label),
                    "stratum": stratum_name,
                    "n_pairs": int(test["n_pairs"]),
                    "mean_control": float(test["mean_control"]),
                    "mean_case": float(test["mean_case"]),
                    "delta": float(test["delta"]),
                    "n_donors_positive": int(test["n_donors_positive"]),
                    "t_p": float(test["t_p"]),
                    "wilcoxon_p": float(test["wilcoxon_p"]),
                    "reason": str(test["reason"]),
                }
            )
        return out

    blocks = [pd.DataFrame(rows_for(frame, "all"))]
    for stratum in strata:
        blocks.append(pd.DataFrame(rows_for(frame[frame["stratum"] == stratum], f"q{stratum + 1}")))

    table = pd.concat([b for b in blocks if not b.empty], ignore_index=True)
    if table.empty:
        return table
    table["fdr"] = np.nan
    # BH within each stratum: the unstratified rows and each quantile bin are separate
    # families of tests over the same labels, so pooling them would double-count.
    for index in table.groupby("stratum").groups.values():
        table.loc[index, "fdr"] = bh_fdr(table.loc[index, "t_p"].to_numpy(), method=fdr_method)

    stratum_rows = table[table["stratum"] != "all"]
    for position in table.index[table["stratum"] == "all"]:
        label = table.at[position, "label"]
        pooled_delta = table.at[position, "delta"]
        mine = stratum_rows[stratum_rows["label"] == label]
        estimable = mine[mine["n_pairs"] >= min_pairs]
        same_sign = (
            int(np.sum(np.sign(estimable["delta"].to_numpy()) == np.sign(pooled_delta)))
            if np.isfinite(pooled_delta)
            else 0
        )
        significant = int(np.sum(estimable["t_p"].to_numpy() < alpha))
        table.at[position, "n_strata"] = float(estimable.shape[0])
        table.at[position, "n_strata_same_sign"] = float(same_sign)
        table.at[position, "n_strata_significant"] = float(significant)

        n_strata = estimable.shape[0]
        pooled_p = table.at[position, "t_p"]
        if table.at[position, "n_pairs"] < min_pairs:
            verdict, reason = (
                "insufficient_pairs",
                (
                    f"{int(table.at[position, 'n_pairs'])} complete pair(s) is below "
                    f"min_pairs={min_pairs}"
                ),
            )
        elif not np.isfinite(pooled_p) or pooled_p >= alpha:
            verdict, reason = (
                "no_raw_effect",
                (
                    "the unstratified abundance shift is not significant, so there is no "
                    "claim to audit"
                ),
            )
        elif n_strata == 0:
            verdict, reason = (
                "insufficient_pairs",
                ("no depth stratum retained enough complete donor pairs to re-test"),
            )
        elif same_sign == n_strata and significant >= 1:
            verdict, reason = "robust", ""
        elif same_sign == n_strata:
            verdict, reason = (
                "attenuated",
                (
                    f"the shift keeps its direction in all {n_strata} depth strata but reaches "
                    f"significance in none of them, which at these per-stratum cell counts is "
                    f"expected for a real effect and cannot be distinguished from a weak one"
                ),
            )
        elif same_sign >= n_strata - 1:
            verdict, reason = (
                "attenuated",
                (
                    f"the shift holds its direction in {same_sign}/{n_strata} depth strata; "
                    f"the dissenting stratum is usually the sparsest, so report this label "
                    f"as supporting rather than primary"
                ),
            )
        else:
            verdict, reason = (
                "depth_driven",
                (
                    f"the shift holds its direction in only {same_sign}/{n_strata} depth "
                    f"strata, so holding depth fixed removes it; this label is substantially "
                    f"a depth stratum"
                ),
            )
        table.at[position, "verdict"] = verdict
        if reason:
            existing = str(table.at[position, "reason"])
            table.at[position, "reason"] = f"{existing}; {reason}" if existing else reason

    ordered = [
        "label",
        "stratum",
        "n_pairs",
        "mean_control",
        "mean_case",
        "delta",
        "n_donors_positive",
        "t_p",
        "wilcoxon_p",
        "fdr",
        "n_strata",
        "n_strata_same_sign",
        "n_strata_significant",
        "verdict",
        "reason",
    ]
    for column in ordered:
        if column not in table.columns:
            table[column] = np.nan
    return table[ordered]
