"""Pure-math module-remodeling statistics.

No I/O, no AnnData, no R — just numpy/pandas/scipy/statsmodels transformations
over a per-cell score matrix and a design frame. Each function is independently
testable with tiny synthetic fixtures. Mediation lives next door in
:mod:`cellquorum.stats.causal_mediation` — same shape, same guards, its own module
because its unit of analysis is a sample rather than a cell.

The house statistical bar is enforced here, not left to the caller:

* pseudoreplication is absorbed at the level the contrast actually varies at. A
  donor random intercept absorbs the donor *mean*, which is enough only when
  condition is a property of the donor. In a paired cohort condition varies
  *within* donor, at the sample, so the model carries a sample variance component
  nested in donor as well — without it the replicate count for the contrast is
  the cell count, and the standard error is wrong by the square root of the cells
  per sample (see :func:`lmm_effect_sizes`);
* every reported p-value is placed against the smallest one the design's own
  randomization set can reach (:func:`randomization_floor`), and the
  assumption-free donor-level randomization p is reported beside it, so a
  parametric p-value that rests entirely on the distributional assumption is
  visible as such rather than quotable as evidence;
* every test family is BH-FDR corrected — and where a family is wide enough that
  its own floor puts BH out of reach for any lone result
  (:func:`fdr_floor_reachability`), that is reported as a property of the design
  rather than left to read as an absence of signal;
* permutations and any sampling are seeded and deterministic;
* guards (>= 2 donors per arm) trigger an explicit, recorded fallback rather
  than a silent crash or a misleading estimate.
"""

from __future__ import annotations

import itertools
import warnings
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import special, stats
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_SEED = 1337
UNASSIGNED = "unassigned"

# Above this many pairs the sign-flip reference set is sampled rather than
# enumerated. 2**18 sign vectors of 18 doubles is ~38 MB, cached once per pair
# count; 2**20 would be 168 MB, which is not worth the last two decimal places of
# a p-value that is already far below any threshold anyone applies.
_MAX_EXACT_PAIRS = 18

# Same trade-off for the unpaired reference set, where the enumeration is over
# combinations and runs in Python rather than in one matrix product.
_MAX_EXACT_COMBINATIONS = 20_000

#: Draws used when a reference set is too large to enumerate. The resulting
#: p-value is never smaller than ``1 / _N_RANDOMIZATION_DRAWS``, which at these
#: sizes is still above the design's own floor.
_N_RANDOMIZATION_DRAWS = 50_000

# A spread this small *relative to the values themselves* is floating-point
# residue from the sums and means that produced them, not a measured difference.
# sqrt(machine epsilon) is the conventional line. Checking for an exact zero is
# not enough: whether six identical paired differences leave a spread of 0.0 or
# of 1e-16 is an accident of summation order, and 1e-16 divides through to a t
# statistic of 1e16 and a p-value of 1e-82 that reads as the strongest finding
# in the table.
_NEGLIGIBLE_RELATIVE_SPREAD = float(np.sqrt(np.finfo(float).eps))

# Smallest positive double. A p-value that underflows to exactly zero is real
# evidence reported at the floor of the arithmetic, so it is floored here rather
# than passed on as a zero that BH would carry into the FDR column.
_MIN_P = float(np.nextafter(0.0, 1.0))


def _is_negligible_spread(sd: float, values: np.ndarray) -> bool:
    """True when ``sd`` is at the floating-point noise floor of ``values``."""
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    return sd <= _NEGLIGIBLE_RELATIVE_SPREAD * scale


def _floor_p(p_value: float) -> tuple[float, str]:
    """Clamp an underflowed p-value to the smallest representable double."""
    if np.isfinite(p_value) and p_value == 0.0:
        return _MIN_P, (
            f"the p-value underflowed to zero and is reported at the smallest representable "
            f"double ({_MIN_P:.3g})"
        )
    return float(p_value), ""


def randomization_floor(donors: Sequence, is_case: Sequence[bool]) -> tuple[float, int]:
    """Smallest two-sided p-value this design's own randomization set can reach.

    Condition is not a property of a cell. It is assigned to a *sample*, and the
    only reassignments of it that respect a paired cohort are the ones that swap a
    donor's two arms. So the design admits ``2**k`` assignments from the ``k``
    donors that span both arms, times ``C(n, n_case)`` from any donors that sit in
    one arm only, and a two-sided test can place at most two of those at or beyond
    the observed statistic. That makes ``2 / n_assignments`` a floor on any
    assumption-free p-value for the contrast — whatever statistic is used, and
    however many cells were measured. Nine donor pairs cannot get below 0.0039;
    eight cannot get below 0.0078.

    A parametric p-value *can* fall below the floor: that is exactly what the
    distributional assumption buys, and a paired t-test on eight pairs will happily
    return 1e-6. What the floor gives is a scale for how much of a small p-value
    came from the design and how much came from the model. A p-value one or two
    orders of magnitude below it is leaning on the assumption; a p-value forty
    orders below it is not describing the cohort at all, and in practice means
    replicates are being counted at the wrong level.

    So the floor is a scale and **never a veto on its own**, and a caller that gates
    on ``p_below_design_floor`` alone has introduced a second error in the course of
    fixing the first. Nine unanimous pairs sit at the floor's own minimum of 0.0039
    while a correctly specified model on the same data reads 4e-05 — the strongest
    evidence such a cohort can produce, dropped by a one-column gate. The gate that
    is correct is the *conjunction*: below the floor **and** the assumption-free
    ``donor_p`` failing to corroborate it. That is why
    :func:`lmm_effect_sizes` reports the flag and the donor-level p side by side and
    rewrites neither.

    Args:
        donors: Donor identifier per observation.
        is_case: Whether each observation is in the case arm.

    Returns:
        ``(floor_p, n_pairs)``. ``n_pairs`` is the number of donors present in both
        arms — the count that sets the floor for a paired contrast.
    """
    frame = pd.DataFrame(
        {
            "donor": np.asarray(donors, dtype=object),
            "case": np.asarray(is_case, dtype=bool),
        }
    )
    if frame.empty:
        return np.nan, 0
    arms = frame.groupby("donor")["case"].nunique()
    spanning = arms.index[arms == 2]
    n_pairs = int(len(spanning))

    one_arm = frame[~frame["donor"].isin(spanning)].groupby("donor")["case"].first()
    n_case_only = int(one_arm.sum())
    n_control_only = int(len(one_arm) - n_case_only)
    unpaired_assignments = int(special.comb(n_case_only + n_control_only, n_case_only, exact=True))

    assignments = (2**n_pairs) * unpaired_assignments
    if assignments <= 0:
        return np.nan, n_pairs
    return float(min(1.0, 2.0 / assignments)), n_pairs


def fdr_floor_reachability(
    floor_p: float, n_tests: int, *, alpha: float = 0.05
) -> tuple[int, bool]:
    """How many tests must sit at the design floor together before BH passes any of them.

    A design floor is usually read as a statement about one test, and read that way it
    looks survivable: eight donor pairs reach 0.0078, comfortably under 0.05. Inside a
    family it is not, because BH compares the ``k``-th smallest p-value against
    ``alpha * k / n_tests``. If every p-value in the family is bounded below by
    ``floor_p``, then the smallest ``k`` that can clear its own threshold is the smallest
    one with ``floor_p <= alpha * k / n_tests`` — so **no single isolated result can be
    FDR-significant at all**, and significance is available only to a block of at least
    that many tests moving together.

    On eight pairs in a 45-test family that block is 8; on seven pairs it is 15, which is a
    third of the family. This is a property of the design and the family size alone — no
    data enters — so it can be computed before an analysis is run, and it is the honest
    answer to "why did nothing survive correction": often not because the effects were
    small but because the cohort could not have shown a lone one at that bar however large
    it was.

    An assumption-free test is bounded by the floor and so is bound by this. A parametric
    test is not, which is one more reason to read the two beside each other.

    Args:
        floor_p: The design's randomization floor, from :func:`randomization_floor`.
        n_tests: Size of the BH family the test sits in.
        alpha: FDR level the family is corrected at.

    Returns:
        ``(min_concordant, reachable)``. ``min_concordant`` is the smallest number of
        tests that must simultaneously reach the floor for BH to call any of them, and
        ``reachable`` says whether that many tests exist in the family at all. When the
        floor is not finite or the family is empty, ``(0, False)``.
    """
    if not np.isfinite(floor_p) or n_tests <= 0 or alpha <= 0.0:
        return 0, False
    min_concordant = int(np.ceil(float(floor_p) * int(n_tests) / float(alpha)))
    min_concordant = max(1, min_concordant)
    return min_concordant, bool(min_concordant <= int(n_tests))


#: Columns describing the BH family an item was corrected inside. They are a property of the
#: family and not of the item, so every one of them has to be recomputed when the family
#: changes — see :func:`recorrect_within_family`.
FAMILY_COLUMNS: tuple[str, ...] = (
    "family_size",
    "family_best_floor_p",
    "family_min_concordant",
    "family_floor_reachable",
)

#: Default ``p-value column -> FDR column`` map, matching what
#: :func:`cellquorum.stats.paired_concordance.paired_value_concordance` emits.
DEFAULT_FDR_COLUMNS: dict[str, str] = {
    "sign_test_p": "sign_test_fdr",
    "sign_test_p_conservative": "sign_test_fdr_conservative",
    "wilcoxon_p": "wilcoxon_fdr",
}

#: Columns naming the panel a composite item belongs to, from
#: :func:`declared_panel_membership`.
PANEL_MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "in_panel",
    "panel_sets",
    "n_entities",
    "n_declared_entities",
)


def declared_panel_membership(
    declared: dict[str, Sequence[str]],
    items: Sequence[str],
    *,
    item_label: str = "item",
    entity_sep: str = "->",
    member_sep: str = "_",
) -> pd.DataFrame:
    """Which declared sets each composite item lies inside, requiring *every* entity to.

    A scan whose family is too wide for BH to reach its own floor
    (:func:`fdr_floor_reachability`) is not rescued by a laxer threshold. It is rescued by a
    smaller family declared before the scan ran — and the only kind of restriction that is
    legitimate is one whose rule never reads a p-value. Gene sets that were written for a
    different purpose are such a rule, so this maps them onto the composite keys a
    communication or interaction scan tests: ``FN1->ITGAV_ITGB1`` names a ligand and a
    two-chain receptor, and it qualifies only when the ligand *and both chains* are declared
    genes.

    Requiring every entity rather than any is the whole point. "Any" makes the panel a
    statement about everything a declared gene happens to touch, which is most of the
    resource; "every" makes it a statement about the declared sets themselves.

    The rule is deliberately blind to the results: pass it the sets a manifest declares and
    the outcome does not depend on which items turned out to move. The check that it was not
    quietly drawn around the winners is that the largest effect in a scan is usually *not* in
    the panel, and that is visible in the returned frame rather than argued for.

    Args:
        declared: ``set name -> member entities``, e.g. the gene modules a manifest states.
        items: Composite keys to classify. Duplicates collapse.
        item_label: Name of the returned key column.
        entity_sep: Splits an item into its sides, e.g. ligand from receptor.
        member_sep: Splits one side into its members, e.g. a complex into chains.

    Returns:
        One row per unique item with :data:`PANEL_MEMBERSHIP_COLUMNS`: ``in_panel`` (every
        entity is declared), ``panel_sets`` (the declared sets involved, ``;``-joined and
        sorted, so the column is stable), and the entity counts behind the verdict, which are
        what let a reader see *why* an item was excluded rather than only that it was.
    """
    owner: dict[str, list[str]] = {}
    for name, members in declared.items():
        for member in members:
            owner.setdefault(str(member), []).append(str(name))

    rows = []
    for item in items:
        text = str(item)
        left, _, right = text.partition(entity_sep)
        entities = [left, *right.split(member_sep)] if right else [left]
        entities = [part for part in (e.strip() for e in entities) if part]
        hits = [owner.get(entity, []) for entity in entities]
        rows.append(
            {
                item_label: text,
                "in_panel": bool(entities) and all(names for names in hits),
                "panel_sets": ";".join(sorted({n for names in hits for n in names})),
                "n_entities": len(entities),
                "n_declared_entities": sum(1 for names in hits if names),
            }
        )
    frame = pd.DataFrame(rows, columns=[item_label, *PANEL_MEMBERSHIP_COLUMNS])
    return frame.drop_duplicates(subset=item_label).reset_index(drop=True)


def recorrect_within_family(
    table: pd.DataFrame,
    *,
    by: Sequence[str] = (),
    alpha: float = 0.05,
    fdr_columns: dict[str, str] | None = None,
    floor_col: str = "design_floor_p",
    scanned_col: str = "n_scanned",
) -> pd.DataFrame:
    """Recompute every family-dependent column after a family has been restricted.

    Restricting a scan to a pre-specified panel changes the BH family, and BH depends on the
    company an item keeps: the same p-value corrected among 7,500 items and among 30 is two
    different FDRs. So a restricted table that carries the scan's ``sign_test_fdr`` forward is
    reporting a correction for a family it is no longer in — and carrying ``family_size`` or
    ``family_min_concordant`` forward is the identical mistake one column over, which is easier
    to miss because those columns look descriptive.

    This recomputes all of them together, per family, so the two cannot drift apart. The
    incoming ``family_size`` is preserved as ``n_scanned`` rather than dropped, because a reader
    has to be able to see what fraction of the tested space was pre-specified; without it a
    panel result is indistinguishable from a discovery.

    Nothing here re-tests anything. Every p-value is left exactly as the original test produced
    it — only the multiplicity accounting moves.

    Args:
        table: An already-restricted table, e.g. the rows of a scan that lie inside a declared
            panel. Restriction is a boolean mask and belongs to the caller; the arithmetic that
            is easy to get wrong is what lives here.
        by: Columns identifying one BH family, e.g. ``("focus", "flow")``. Empty means the
            whole table is one family.
        alpha: FDR level.
        fdr_columns: ``p-value column -> FDR column``. Defaults to
            :data:`DEFAULT_FDR_COLUMNS`. Pairs whose p-value column is absent are skipped, so
            a table carrying only some of them is fine.
        floor_col: Per-item randomization floor, used for the family's reachability.
        scanned_col: Where the incoming ``family_size`` is recorded.

    Returns:
        A copy with :data:`FAMILY_COLUMNS`, ``scanned_col`` and every present FDR column
        recomputed within each family. Row order is preserved.
    """
    if table.empty:
        return table.copy()

    mapping = dict(DEFAULT_FDR_COLUMNS if fdr_columns is None else fdr_columns)
    keys = [str(column) for column in by]
    missing = [column for column in keys if column not in table.columns]
    if missing:
        raise KeyError(f"recorrect_within_family: no such grouping column(s): {missing}")

    out = table.copy()
    # Record what the incoming family was before overwriting it, and do it once for the whole
    # table: doing it per family would see the column exist as soon as the first family had
    # written it and leave every later family without one.
    if scanned_col not in out.columns:
        # With no incoming ``family_size`` there is nothing to preserve, and the honest
        # statement is that the table itself is all that was tested.
        incoming = out.get("family_size", pd.Series(len(out), index=out.index))
        out[scanned_col] = pd.to_numeric(incoming, errors="coerce").fillna(len(out)).to_numpy()

    # A single-family table goes through the same code path as a grouped one, so the two can
    # never diverge in how they count a family.
    blocks: list[tuple[object, pd.DataFrame]]
    blocks = list(out.groupby(keys, observed=True, sort=False)) if keys else [((), out)]
    for _, block in blocks:
        index = block.index
        out.loc[index, "family_size"] = len(block)

        if floor_col in block.columns:
            floors = pd.to_numeric(block[floor_col], errors="coerce").to_numpy(dtype=float)
            best = float(np.nanmin(floors)) if np.isfinite(floors).any() else float("nan")
        else:
            best = float("nan")
        min_concordant, reachable = fdr_floor_reachability(best, len(block), alpha=alpha)
        out.loc[index, "family_best_floor_p"] = best
        out.loc[index, "family_min_concordant"] = min_concordant
        out.loc[index, "family_floor_reachable"] = reachable

        for p_col, fdr_col in mapping.items():
            if p_col not in block.columns:
                continue
            out.loc[index, fdr_col] = bh_fdr(
                pd.to_numeric(block[p_col], errors="coerce").to_numpy(dtype=float)
            )

    for column in ("family_size", scanned_col, "family_min_concordant"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    out["family_floor_reachable"] = out["family_floor_reachable"].astype(bool)
    out["family_best_floor_p"] = pd.to_numeric(out["family_best_floor_p"], errors="coerce")
    return out


@cache
def _sign_matrix(n: int) -> np.ndarray:
    """All ``2**n`` sign vectors, one per row. Cached: the same ``n`` recurs per family."""
    codes = np.arange(2**n, dtype=np.int64)[:, None]
    return 1.0 - 2.0 * ((codes >> np.arange(n)) & 1).astype(float)


def _sign_flip_p(diff: np.ndarray, *, seed: int) -> tuple[float, str]:
    """Two-sided sign-flip randomization p for paired differences.

    This is the paired design's own reference distribution: the statistic is
    recomputed under every way the two arms could have been swapped within donors.
    It assumes nothing about the shape of the differences, which is why it is the
    right thing to print next to a model-based p-value.
    """
    n = int(diff.size)
    if n < 2:
        return np.nan, ""
    observed = float(abs(diff.sum()))
    tol = 1e-12 * max(1.0, float(np.abs(diff).sum()))
    if n <= _MAX_EXACT_PAIRS:
        totals = np.abs(_sign_matrix(n) @ diff)
        return float((totals >= observed - tol).mean()), "sign_flip_exact"
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(_N_RANDOMIZATION_DRAWS, n))
    # The observed assignment is a member of its own reference set.
    signs[0, :] = 1.0
    totals = np.abs(signs @ diff)
    return float((totals >= observed - tol).mean()), "sign_flip_mc"


def _label_permutation_p(
    case_values: np.ndarray, control_values: np.ndarray, *, seed: int
) -> tuple[float, str]:
    """Two-sided label-permutation p for independent per-donor values."""
    pooled = np.concatenate([case_values, control_values])
    n, k = int(pooled.size), int(case_values.size)
    if k < 2 or n - k < 2:
        return np.nan, ""
    observed = float(abs(case_values.mean() - control_values.mean()))
    tol = 1e-12 * max(1.0, float(np.abs(pooled).mean()))

    def _statistic(mask: np.ndarray) -> np.ndarray:
        left = (pooled * mask).sum(axis=1) / k
        right = (pooled * ~mask).sum(axis=1) / (n - k)
        return np.abs(left - right)

    total = int(special.comb(n, k, exact=True))
    if total <= _MAX_EXACT_COMBINATIONS:
        picks = np.array(list(itertools.combinations(range(n), k)), dtype=int)
        mask = np.zeros((picks.shape[0], n), dtype=bool)
        np.put_along_axis(mask, picks, True, axis=1)
        return float((_statistic(mask) >= observed - tol).mean()), "label_perm_exact"

    rng = np.random.default_rng(seed)
    mask = np.zeros((_N_RANDOMIZATION_DRAWS, n), dtype=bool)
    mask[:, :k] = True
    mask = rng.permuted(mask, axis=1)
    mask[0, :] = False
    mask[0, :k] = True
    return float((_statistic(mask) >= observed - tol).mean()), "label_perm_mc"


def bh_fdr(pvalues: np.ndarray | list[float], method: str = "fdr_bh") -> np.ndarray:
    """Benjamini-Hochberg FDR that tolerates NaN p-values.

    NaNs (a test family member that could not be computed) are held out of the
    correction and returned as NaN in place, so a single un-fittable row never
    drags the whole family's q-values around. Finite p-values are corrected
    among themselves exactly as :func:`statsmodels.stats.multitest.multipletests`
    would on that subset.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.any():
        out[finite] = multipletests(p[finite], method=method)[1]
    return out


def signature_argmax_labels(
    scores: pd.DataFrame,
    cluster_labels: pd.Series | np.ndarray | list,
    *,
    signatures: list[str] | None = None,
    min_margin: float = 0.0,
) -> pd.DataFrame:
    """Label each cluster by its dominant signature (cluster-level argmax).

    Cluster-level mean scores are more stable than per-cell argmax. Each
    signature is z-scored across clusters so signatures on different scales are
    comparable, then each cluster takes the signature with the highest z. If the
    gap between the top and second signature is below ``min_margin`` the cluster
    is left :data:`UNASSIGNED` — the ambiguity guard the spec requires.

    Parameters
    ----------
    scores
        Per-cell signature scores (cells x signatures); index aligns to
        ``cluster_labels``.
    cluster_labels
        Per-cell cluster id, aligned to ``scores.index``.
    signatures
        Columns of ``scores`` to consider (default: all).
    min_margin
        Minimum top-minus-second z gap required to assign a label.

    Returns
    -------
    DataFrame with columns: cluster, label, top_signature, top_z, second_z,
    margin (one row per cluster).
    """
    sigs = list(signatures) if signatures is not None else list(scores.columns)
    cl = pd.Series(np.asarray(cluster_labels), index=scores.index, name="cluster")
    cluster_means = scores[sigs].groupby(cl, observed=True).mean()

    # z-score each signature across clusters (population std, ddof=0); a
    # zero-variance signature contributes 0 to every cluster.
    mu = cluster_means.mean(axis=0)
    sd = cluster_means.std(axis=0, ddof=0).replace(0.0, np.nan)
    z = (cluster_means - mu) / sd
    z = z.fillna(0.0)

    rows = []
    for cluster, zrow in z.iterrows():
        order = zrow.sort_values(ascending=False)
        top_sig = order.index[0]
        top_z = float(order.iloc[0])
        second_z = float(order.iloc[1]) if len(order) > 1 else float("-inf")
        margin = top_z - second_z
        label = top_sig if margin >= min_margin else UNASSIGNED
        rows.append(
            {
                "cluster": cluster,
                "label": label,
                "top_signature": top_sig,
                "top_z": top_z,
                "second_z": second_z,
                "margin": margin,
            }
        )
    return pd.DataFrame(rows)


def signed_program_contrast_index(
    scores: pd.DataFrame,
    *,
    up: list[str],
    down: list[str],
) -> pd.Series:
    """Standardized signed contrast: ``z(sum(up)) - z(sum(down))`` per cell.

    The generalization of the EndoMT index. Up-programs and down-programs are
    each summed, standardized across cells (population std), and subtracted. The
    result is mean-centered by construction, monotone in the up-vs-down balance,
    and transparent.
    """

    def _z(cols: list[str]) -> np.ndarray:
        s = scores[cols].sum(axis=1).to_numpy(dtype=float)
        sd = s.std(ddof=0)
        if sd == 0:
            return np.zeros_like(s)
        return (s - s.mean()) / sd

    idx = _z(up) - _z(down)
    return pd.Series(idx, index=scores.index, name="contrast_index")


def lmm_effect_sizes(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    donor_col: str,
    condition_col: str,
    group_col: str,
    case: str,
    control: str,
    sample_col: str | None = None,
    programs: list[str] | None = None,
    groups: list[str] | None = None,
    fdr_method: str = "fdr_bh",
    min_donors_per_arm: int = 2,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Per group x program, fit ``score ~ condition + (1|donor) + (1|sample)``.

    The reported effect is the fixed-effect case-minus-control coefficient with a
    95% CI, p-value, and BH-FDR across the whole group x program family.

    **The sample variance component is not optional on a paired design.** A donor
    random intercept absorbs the donor *mean*, and that is the whole of the
    pseudoreplication only when condition is a property of the donor. When each
    donor contributes one case and one control sample, condition varies *within*
    donor: the replicate unit for the contrast is the sample, and a model with a
    donor intercept alone treats every cell in a sample as an independent
    replicate of that sample's condition. It gets the effect right and the
    standard error wrong by roughly the square root of the cells per sample. On
    real data that produced ``p = 3e-47`` from eight donor pairs in which one
    donor moved in the reported direction. Adding
    ``(1|sample nested in donor)`` recovers the design's own precision: on the
    same row, SE 0.20 -> 0.88 against a donor-level 0.94, and ``p`` 3e-47 ->
    1e-3 against a donor-level 1.6e-2.

    The component is added whenever the group holds more samples than donors,
    which is exactly when some donor spans more than one sample. When it does not
    — one sample per donor, condition assigned between donors — the donor
    intercept already sits at the level condition varies at, and a second
    component would be collinear with it.

    Every row also carries the design's own answer beside the model's:
    ``design_floor_p`` is the smallest two-sided p-value the cohort's
    randomization set can reach (:func:`randomization_floor`),
    ``p_below_design_floor`` says whether the reported p-value is below it — that
    is, resting on the distributional assumption rather than on the design — and
    ``donor_p`` is the assumption-free randomization p computed on one value per
    donor. Agreement between ``p_value`` and ``donor_p`` is what licenses quoting
    the model's smaller number.

    Args:
        scores: Cells x programs.
        metadata: Per-cell design frame, indexed like ``scores``.
        donor_col: Column naming the donor.
        condition_col: Column naming the arm.
        group_col: Column naming the group each row is fitted within.
        case: Value of ``condition_col`` for the case arm.
        control: Value of ``condition_col`` for the control arm.
        sample_col: Column naming the sample — the unit condition was assigned to.
            When ``None`` the sample is taken to be donor x condition, which is the
            right identification for a cohort with one sample per donor per arm and
            is conservative for any other, since it pools a donor's replicate
            samples within an arm rather than splitting them.
        programs: Subset and order of ``scores`` columns to fit. Defaults to all.
        groups: Subset and order of groups to fit. Defaults to order of appearance.
        fdr_method: Passed to :func:`bh_fdr` over the whole group x program family.
        min_donors_per_arm: Below this many donors in either arm the mixed model is
            not attempted and the donor-level fallback runs instead.
        seed: Seeds the randomization reference set when it is sampled rather than
            enumerated.

    Guards, all of them recorded in the table rather than applied silently:

    * a boundary ("singular") fit — a variance component estimated at zero, which
      ``MixedLM`` reports as ``converged=False`` — is *kept*, because its fixed
      effect, standard error and CI are all valid. ``converged=False`` and a
      ``reason`` say so;
    * a nested fit that cannot be obtained at all falls back to the donor
      intercept alone, and says so in ``reason`` — that fallback is the
      anticonservative model, so it is never silent;
    * a group x program with fewer than ``min_donors_per_arm`` donors in either
      arm, or whose mixed model yields no finite fixed effect, falls back to one
      value per donor and then the t-test the design supports: ``paired_t`` when
      at least two donors span both arms, otherwise Welch on independent
      per-donor means (``welch_t``). Matching the test to the design is what
      keeps an unpaired cohort from returning a blank row;
    * a degenerate test reports its effect and withholds only its p-value. Zero
      variance among the paired differences makes the t statistic undefined, not
      infinitely significant, so no row ever carries ``p_value == 0``;
    * nothing that could not be estimated is left unexplained: ``reason`` is
      non-empty for every degraded or blank row, and ``""`` for a clean fit.

    Returns
    -------
    DataFrame, one row per (group, program), with columns: group, program, effect,
    ci_low, ci_high, p_value, fdr, n_case, n_control, n_donors, n_pairs,
    design_floor_p, p_below_design_floor, donor_effect, donor_p, donor_test,
    n_donors_concordant, method, variance_components, converged, reason.
    ``method`` is one of ``lmm``, ``paired_t``, ``welch_t`` or ``none``;
    ``variance_components`` is ``donor+sample``, ``donor`` or ``""``;
    ``donor_test`` is ``sign_flip_exact``, ``sign_flip_mc``, ``label_perm_exact``,
    ``label_perm_mc`` or ``""``; ``converged`` is a nullable boolean, NA where no
    mixed model was fitted. ``n_donors_concordant`` is how many of ``n_pairs``
    donors moved in the direction of ``donor_effect``, NaN on an unpaired stratum —
    the one number in the table that needs no assumption, no correction and no
    explaining, and the one that distinguishes a cohort-wide shift from a large
    effect in one donor and nothing in the rest.
    """
    progs = list(programs) if programs is not None else list(scores.columns)
    meta = metadata.loc[scores.index]
    grp_values = list(groups) if groups is not None else list(pd.unique(meta[group_col]))

    if sample_col is not None:
        if sample_col not in meta.columns:
            raise KeyError(f"sample_col '{sample_col}' is not a column of metadata")
        sample_values = meta[sample_col].astype(str)
    else:
        # The sample is where condition was assigned. With one sample per donor per
        # arm this reconstructs it exactly; with several it pools them, which loses
        # power but cannot manufacture it.
        sample_values = meta[donor_col].astype(str) + "|" + meta[condition_col].astype(str)

    rows = []
    for group in grp_values:
        gmask = (meta[group_col] == group).to_numpy()
        gmeta = meta.loc[gmask]
        gscores = scores.loc[gmask]
        gsample = sample_values.loc[gmask]
        cond = gmeta[condition_col]
        arm_mask = cond.isin([case, control]).to_numpy()
        gmeta = gmeta.loc[arm_mask]
        gscores = gscores.loc[arm_mask]
        gsample = gsample.loc[arm_mask]
        cond = gmeta[condition_col]

        is_case = (cond == case).to_numpy()
        donors_case = set(gmeta[donor_col][is_case])
        donors_control = set(gmeta[donor_col][~is_case])
        n_donors = len(donors_case | donors_control)
        floor_p, n_pairs = randomization_floor(gmeta[donor_col].to_numpy(), is_case)

        for prog in progs:
            y = gscores[prog].to_numpy(dtype=float)
            df = pd.DataFrame(
                {
                    "score": y,
                    "cond": is_case.astype(float),
                    "donor": gmeta[donor_col].to_numpy(),
                    "sample": gsample.to_numpy(),
                }
            )
            n_case = int(is_case.sum())
            n_control = int((~is_case).sum())

            enough = (
                len(donors_case) >= min_donors_per_arm and len(donors_control) >= min_donors_per_arm
            )
            effect = ci_low = ci_high = p_value = np.nan
            method = "lmm"
            components = ""
            converged: bool | None = None
            reason = ""

            fit = _fit_effect(df) if enough else None
            if fit is not None:
                effect, ci_low, ci_high, raw_p, converged, components, nested_reason = fit
                p_value, floor_reason = _floor_p(raw_p)
                explanations = []
                if nested_reason:
                    explanations.append(nested_reason)
                if not converged:
                    explanations.append(
                        "the optimizer stopped before convergence; the fixed effect, CI and p are "
                        "retained because they are finite, but treat them as provisional"
                    )
                if floor_reason:
                    explanations.append(floor_reason)
                reason = "; ".join(explanations)
            else:
                if not enough:
                    reason = (
                        f"mixed model not attempted: {len(donors_case)} case and "
                        f"{len(donors_control)} control donor(s), below min_donors_per_arm="
                        f"{min_donors_per_arm}"
                    )
                else:
                    reason = "mixed model produced no finite fixed effect"
                method, effect, ci_low, ci_high, p_value, fallback_reason = _donor_level_effect(
                    df, case_val=1.0, control_val=0.0
                )
                if fallback_reason:
                    reason = f"{reason}; {fallback_reason}"

            donor_effect, donor_p, donor_test, n_concordant = _donor_randomization(
                df, case_val=1.0, control_val=0.0, seed=seed
            )

            rows.append(
                {
                    "group": group,
                    "program": prog,
                    "effect": effect,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "n_case": n_case,
                    "n_control": n_control,
                    "n_donors": n_donors,
                    "n_pairs": n_pairs,
                    "design_floor_p": floor_p,
                    "donor_effect": donor_effect,
                    "donor_p": donor_p,
                    "donor_test": donor_test,
                    "n_donors_concordant": n_concordant,
                    "method": method,
                    "variance_components": components,
                    "converged": converged,
                    "reason": reason,
                }
            )

    out = pd.DataFrame(rows)
    out["fdr"] = bh_fdr(out["p_value"].to_numpy(), method=fdr_method)
    # A p-value the design's randomization set cannot reach is flagged rather than
    # rewritten: the number is what the model said, and what it needs beside it is
    # the scale on which to read it. It is not excluded from the FDR family either,
    # because dropping rows from a family changes everyone else's q-value.
    out["p_below_design_floor"] = (
        out["p_value"].to_numpy() < out["design_floor_p"].to_numpy()
    ) & np.isfinite(out["p_value"].to_numpy())
    # Nullable boolean: NA means no mixed model was fitted, which is a different
    # statement from a model that was fitted and did not converge.
    out["converged"] = out["converged"].astype("boolean")
    return out[
        [
            "group",
            "program",
            "effect",
            "ci_low",
            "ci_high",
            "p_value",
            "fdr",
            "n_case",
            "n_control",
            "n_donors",
            "n_pairs",
            "design_floor_p",
            "p_below_design_floor",
            "donor_effect",
            "donor_p",
            "donor_test",
            "n_donors_concordant",
            "method",
            "variance_components",
            "converged",
            "reason",
        ]
    ]


def _is_nested_design(df: pd.DataFrame) -> bool:
    """Does this group hold more samples than donors — i.e. is condition within-donor?

    More samples than donors means some donor contributed more than one, which on a
    two-arm design means at least one donor spans both arms. That is the whole test
    for whether a sample variance component is (a) needed, because the contrast
    varies below the donor, and (b) identifiable, because it is not collinear with
    the donor intercept.
    """
    return "sample" in df.columns and df["sample"].nunique() > df["donor"].nunique()


def _between_within_dof(df: pd.DataFrame) -> int:
    """Denominator degrees of freedom for the condition contrast, from the design.

    ``MixedLM`` refers its fixed effects to a normal distribution — infinite
    denominator degrees of freedom — which is asymptotically right in the number of
    *replicates* and badly wrong at sixteen samples. On a null fixture with eight
    donor pairs it turns a correct ``p = 0.14`` into ``p = 0.046``: a false positive
    at the conventional threshold, produced by the reference distribution rather
    than by the data.

    So the Wald statistic is referred to a t instead, with the between-within
    denominator df the design supplies. Condition is estimated in the stratum where
    it varies, and that stratum's df is the number of units in it less the blocks it
    is nested in less the effect itself:

    * paired — condition varies within donor, so the stratum is samples within
      donors: ``n_samples - n_donors - 1``. Sixteen samples from eight donors gives
      7, which is the paired t-test's own df on those donors, and coincides with
      Satterthwaite's answer for a balanced design;
    * unpaired — condition varies between donors, so the stratum is the donors:
      ``n_donors - 2``. Four donors per arm gives 6, again the two-sample t df.

    A partly paired cohort is treated as paired, which charges it the full set of
    donor blocks and so understates its df. That is the conservative direction, and
    a design that cannot say whether its contrast is within or between donors should
    not be getting the benefit of the doubt.
    """
    arms_per_donor = df.groupby("donor")["cond"].nunique()
    n_donors = int(len(arms_per_donor))
    paired = bool((arms_per_donor == 2).any())
    if "sample" in df.columns:
        n_samples = int(df["sample"].nunique())
    else:
        # No sample column: the samples the design implies are donor x arm.
        n_samples = int(arms_per_donor.sum())
    dof = (n_samples - n_donors - 1) if paired else (n_donors - 2)
    return max(1, dof)


def _fit_effect(
    df: pd.DataFrame,
) -> tuple[float, float, float, float, bool, str, str] | None:
    """Fit the variance structure the design needs, falling back loudly.

    Returns ``(effect, ci_low, ci_high, p_value, converged, variance_components,
    reason)``, or ``None`` when no mixed model of any structure yields a finite
    fixed effect.

    The order matters and is not a preference. On a within-donor design the
    sample-nested model is the *correct* one and the donor-intercept-only model is
    anticonservative, so the nested fit is attempted first and a drop back to the
    simpler structure carries an explicit reason. Silently reporting the simpler
    model is how a standard error can come out an order of magnitude too small
    with nothing in the table to show it.
    """
    if _is_nested_design(df):
        nested = _mixedlm_effect(df, nested=True)
        if nested is not None:
            return (*nested, "donor+sample", "")
        plain = _mixedlm_effect(df, nested=False)
        if plain is not None:
            return (
                *plain,
                "donor",
                "the sample variance component could not be estimated, so this row is fitted "
                "with a donor intercept alone; condition varies within donor here, so the "
                "standard error is too small and the p-value is anticonservative — read "
                "donor_p instead",
            )
        return None

    plain = _mixedlm_effect(df, nested=False)
    if plain is None:
        return None
    return (*plain, "donor", "")


def _mixedlm_effect(
    df: pd.DataFrame, *, nested: bool | None = None, maxiter: int | None = None
) -> tuple[float, float, float, float, bool] | None:
    """Fit ``score ~ cond + (1|donor)``, optionally ``+ (1|sample)``; or None.

    Returns ``(effect, ci_low, ci_high, p_value, converged)``, or ``None`` when no
    finite fixed effect could be obtained at all — including when ``fit()`` raises,
    which is what a fully degenerate design does here: with no between-donor
    variance left the Hessian is singular and ``MixedLM`` cannot produce a result
    object at all, rather than returning one flagged as unconverged.

    The p-value and the CI are recomputed from the model's coefficient and standard
    error against a t distribution with the design's between-within denominator df
    (:func:`_between_within_dof`) rather than taken from ``result.pvalues``, which
    are normal-based. The estimate and its standard error are the model's; only the
    reference distribution is the design's.

    ``converged`` is therefore about the optimizer, and it is reported rather than
    used as a gate. A run stopped short of convergence still has a finite
    fixed-effect coefficient, standard error and CI; discarding those (which this
    module used to do) substituted a t-test on donor means for the
    pseudoreplication-aware model the caller asked for, and the downgrade was
    visible only as a changed value in the ``method`` column.

    ``nested`` adds a sample variance component inside the donor grouping, which is
    the level condition varies at on a paired design. Default ``None`` decides from
    the frame (:func:`_is_nested_design`), so a caller passing only
    ``score``/``cond``/``donor`` gets the two-level model it always got.

    ``maxiter`` is passed through only when set, so the default optimizer
    behaviour is untouched; it exists to make non-convergence testable.
    """
    if nested is None:
        nested = _is_nested_design(df)
    try:
        # Imported *before* the suppression, not inside it. ``statsmodels`` runs
        # ``warnings.simplefilter("always", ConvergenceWarning)`` at import time, which
        # inserts a filter at the front of the list — so importing it inside the block
        # re-arms the very warning the block just turned off, and every fit from then
        # on warns again. The import has to land outside so ``catch_warnings`` saves
        # that filter and ``simplefilter("ignore")`` clears it.
        import statsmodels.formula.api as smf

        # The whole body is inside the suppression, not just ``fit()``: a boundary
        # variance component is a normal and expected outcome here (many programs
        # genuinely have no sample-level spread), the fact is already reported in
        # ``converged``, and one statsmodels ConvergenceWarning per program x group
        # would bury a run's real output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_kwargs: dict[str, object] = {"reml": False, "method": "lbfgs"}
            if maxiter is not None:
                fit_kwargs["maxiter"] = int(maxiter)
            model_kwargs: dict[str, object] = {}
            if nested:
                # Samples are nested in donors, so the component is declared inside the
                # donor grouping rather than as a second crossed grouping factor:
                # MixedLM takes one ``groups`` and expresses the rest as variance
                # components evaluated within it.
                model_kwargs["re_formula"] = "1"
                model_kwargs["vc_formula"] = {"sample": "0 + C(sample)"}
            result = smf.mixedlm("score ~ cond", df, groups=df["donor"], **model_kwargs).fit(
                **fit_kwargs
            )
            effect = float(result.params.get("cond", np.nan))
            se = float(result.bse.get("cond", np.nan))
            if not (np.isfinite(effect) and np.isfinite(se)) or se <= 0.0:
                return None
            dof = _between_within_dof(df)
            p_value = float(2.0 * stats.t.sf(abs(effect / se), df=dof))
            tcrit = float(stats.t.ppf(0.975, df=dof))
            return (
                effect,
                effect - tcrit * se,
                effect + tcrit * se,
                p_value,
                bool(getattr(result, "converged", True)),
            )
    except Exception:
        return None


def _donor_means(
    df: pd.DataFrame, *, case_val: float, control_val: float
) -> tuple[pd.Series, pd.Series, pd.Index]:
    """One value per donor per arm, plus the donors that appear in both.

    Shared by the donor-level t-test and the donor-level randomization test so the
    two can never disagree about what the design is.
    """
    case = df[df["cond"] == case_val].groupby("donor")["score"].mean()
    control = df[df["cond"] == control_val].groupby("donor")["score"].mean()
    return case, control, case.index.intersection(control.index)


def _donor_randomization(
    df: pd.DataFrame, *, case_val: float, control_val: float, seed: int
) -> tuple[float, float, str, float]:
    """The design's own answer, assuming nothing: ``(effect, p, test, n_concordant)``.

    One value per donor, then the reference distribution the randomization itself
    defines — sign flips within donor when donors are paired, label permutation
    when they are independent. It borrows no distributional assumption and no
    strength from the cell count, so it is the number a model-based p-value has to
    be read against. It is computed on every row regardless of which model ran,
    because its whole use is as a check on the model.

    ``n_concordant`` is how many paired donors moved in the direction of the mean
    difference. It is the most legible statistic a small cohort produces — "seven of
    nine donors went up" needs no distributional assumption, no correction and no
    explanation, and it is what separates an effect from an average of one donor and
    eight zeros at the same p-value. It is NaN for an unpaired stratum, where there
    are no pairs to count rather than zero of them, and 0 for a row with no direction
    to agree with.
    """
    case, control, shared = _donor_means(df, case_val=case_val, control_val=control_val)
    if len(shared) >= 2:
        diff = (case.loc[shared] - control.loc[shared]).to_numpy(dtype=float)
        p_value, test = _sign_flip_p(diff, seed=seed)
        mean_diff = float(diff.mean())
        # Ties count as discordant: a donor that did not move is not evidence that it
        # moved the reported way. Sign is taken against the mean rather than against
        # zero so the count answers "how many agree with the reported effect".
        #
        # A mean difference of exactly zero has no direction, so nothing can agree with
        # it and the count is 0 rather than a comparison of two zero signs. Real data
        # produces this row: a program or gene with no expression in either arm gives
        # every donor a difference of 0, and `sign(0) == sign(0)` would report it as
        # unanimous — the strongest-looking value in the column, on the one row that
        # measured nothing at all.
        direction = np.sign(mean_diff)
        concordant = 0.0 if direction == 0 else float(np.sum(np.sign(diff) == direction))
        return mean_diff, p_value, test, concordant
    if len(case) >= 2 and len(control) >= 2:
        case_values = case.to_numpy(dtype=float)
        control_values = control.to_numpy(dtype=float)
        p_value, test = _label_permutation_p(case_values, control_values, seed=seed)
        return (
            float(case_values.mean() - control_values.mean()),
            p_value,
            test,
            np.nan,
        )
    return np.nan, np.nan, "", np.nan


def _donor_level_effect(
    df: pd.DataFrame, *, case_val: float, control_val: float
) -> tuple[str, float, float, float, float, str]:
    """One value per donor, then the t-test the design actually supports.

    Returns ``(method, effect, ci_low, ci_high, p_value, reason)``.

    Collapsing to per-donor means is what removes pseudoreplication when the
    mixed model is unavailable. Which t-test follows is a property of the design,
    not a preference: donors present in both arms are paired, and independent
    donors get Welch. This module used to run the paired test unconditionally, so
    an unpaired cohort — and any subtype stratum whose case and control cells
    happen to come from disjoint donors — found nothing to pair and returned NaN.
    A planted effect came back as a blank row labelled ``paired_t``,
    indistinguishable from a null.
    """
    case, control, shared = _donor_means(df, case_val=case_val, control_val=control_val)

    if len(shared) >= 2:
        diff = (case.loc[shared] - control.loc[shared]).to_numpy(dtype=float)
        effect = float(diff.mean())
        sd = float(diff.std(ddof=1))
        if _is_negligible_spread(sd, diff):
            # A near-infinite t statistic is not evidence; it is an undefined test.
            return (
                "paired_t",
                effect,
                np.nan,
                np.nan,
                np.nan,
                f"the paired differences are identical across all {len(shared)} donors to within "
                "floating-point error, so the t statistic is undefined; the effect size is "
                "reported without a p-value",
            )
        _, raw_p = stats.ttest_rel(case.loc[shared], control.loc[shared])
        p_value, reason = _floor_p(float(raw_p))
        se = sd / np.sqrt(diff.size)
        tcrit = float(stats.t.ppf(0.975, df=diff.size - 1))
        return ("paired_t", effect, effect - tcrit * se, effect + tcrit * se, p_value, reason)

    if len(case) >= 2 and len(control) >= 2:
        case_values = case.to_numpy(dtype=float)
        control_values = control.to_numpy(dtype=float)
        effect = float(case_values.mean() - control_values.mean())
        se = float(
            np.sqrt(
                case_values.var(ddof=1) / case_values.size
                + control_values.var(ddof=1) / control_values.size
            )
        )
        reason = (
            f"no donor spans both arms ({len(case)} case-only, {len(control)} control-only), so "
            "the donor-level test is unpaired (Welch) rather than paired"
        )
        pooled = np.concatenate([case_values, control_values])
        if _is_negligible_spread(se, pooled):
            return (
                "welch_t",
                effect,
                np.nan,
                np.nan,
                np.nan,
                f"{reason}; the per-donor means have no within-arm spread beyond floating-point "
                "error, so the t statistic is undefined",
            )
        _, raw_p = stats.ttest_ind(case_values, control_values, equal_var=False)
        if not np.isfinite(raw_p):
            return (
                "welch_t",
                effect,
                np.nan,
                np.nan,
                np.nan,
                f"{reason}; the t test returned no p-value",
            )
        p_value, floor_reason = _floor_p(float(raw_p))
        dof = max(case_values.size + control_values.size - 2, 1)
        tcrit = float(stats.t.ppf(0.975, df=dof))
        return (
            "welch_t",
            effect,
            effect - tcrit * se,
            effect + tcrit * se,
            p_value,
            f"{reason}; {floor_reason}" if floor_reason else reason,
        )

    return (
        "none",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        f"not estimable at donor level: {len(shared)} donor(s) in both arms, {len(case)} "
        f"case-only and {len(control)} control-only; a donor-level test needs 2 paired donors "
        "or 2+2 independent ones",
    )


def permanova_by_group(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_col: str,
    condition_col: str,
    group_col: str,
    case: str,
    control: str,
    programs: list[str] | None = None,
    groups: list[str] | None = None,
    n_permutations: int = 999,
    fdr_method: str = "fdr_bh",
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Multivariate condition effect per group, PERMANOVA (Anderson 2001).

    For each group, per-sample module-score vectors (mean score per program per
    sample) form the observations; the squared-Euclidean pseudo-F between the
    case and control label sets is tested against ``n_permutations`` seeded label
    shuffles. Self-implemented (skbio is not a dependency); deterministic under
    ``seed``.

    One call tests every group, which makes the groups a FAMILY: ``fdr`` is
    ``p_value`` BH-adjusted across them. Unadjusted, ten null groups give a ~40%
    chance that at least one permutes below 0.05, and a per-group bar chart stars
    it — the same reason :func:`lmm_effect_sizes` adjusts across module x group.

    Returns
    -------
    DataFrame, one row per group: group, pseudo_F, R2, p_value, fdr, n_samples,
    n_case, n_control, n_perm, seed.
    """
    progs = list(programs) if programs is not None else list(scores.columns)
    meta = metadata.loc[scores.index]
    grp_values = list(groups) if groups is not None else list(pd.unique(meta[group_col]))

    rows = []
    for group in grp_values:
        gmask = (meta[group_col] == group).to_numpy()
        gmeta = meta.loc[gmask]
        gscores = scores.loc[gmask, progs]
        cond = gmeta[condition_col]
        arm_mask = cond.isin([case, control]).to_numpy()
        gmeta = gmeta.loc[arm_mask]
        gscores = gscores.loc[arm_mask]

        # collapse cells -> per-sample mean vectors
        sample = gmeta[sample_col]
        per_sample = gscores.groupby(sample, observed=True).mean()
        sample_cond = gmeta.groupby(sample, observed=True)[condition_col].first()
        sample_cond = sample_cond.loc[per_sample.index]
        labels = (sample_cond == case).to_numpy()

        n_samples = int(per_sample.shape[0])
        n_case = int(labels.sum())
        n_control = int((~labels).sum())

        if n_case < 1 or n_control < 1 or n_samples < 3:
            rows.append(
                {
                    "group": group,
                    "pseudo_F": np.nan,
                    "R2": np.nan,
                    "p_value": np.nan,
                    "fdr": np.nan,
                    "n_samples": n_samples,
                    "n_case": n_case,
                    "n_control": n_control,
                    "n_perm": n_permutations,
                    "seed": seed,
                }
            )
            continue

        X = per_sample.to_numpy(dtype=float)
        d2 = _sq_euclidean(X)
        f_obs, r2 = _pseudo_f(d2, labels)

        rng = np.random.default_rng(seed)
        ge = 1  # include the observed statistic (Anderson convention)
        for _ in range(n_permutations):
            perm = rng.permutation(labels)
            f_perm, _ = _pseudo_f(d2, perm)
            if f_perm >= f_obs:
                ge += 1
        p_value = ge / (n_permutations + 1)

        rows.append(
            {
                "group": group,
                "pseudo_F": float(f_obs),
                "R2": float(r2),
                "p_value": float(p_value),
                "fdr": np.nan,  # filled once the whole family is known
                "n_samples": n_samples,
                "n_case": n_case,
                "n_control": n_control,
                "n_perm": n_permutations,
                "seed": seed,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = bh_fdr(out["p_value"].to_numpy(dtype=float), method=fdr_method)
    return out


def _sq_euclidean(X: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distance matrix."""
    sq = np.sum(X**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    return np.maximum(d2, 0.0)


def _pseudo_f(d2: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Anderson pseudo-F and R2 for a two-group partition on a distance matrix.

    Uses the sum-of-squared-distances decomposition:
    ``SST = sum_{i<j} d2_ij / N``; ``SSW = sum_g sum_{i<j in g} d2_ij / n_g``;
    ``SSA = SST - SSW``; ``F = (SSA/(a-1)) / (SSW/(N-a))`` with ``a=2`` groups.
    """
    n = d2.shape[0]
    triu = np.triu_indices(n, k=1)
    sst = d2[triu].sum() / n

    ssw = 0.0
    for val in (True, False):
        idx = np.where(labels == val)[0]
        ng = idx.size
        if ng < 1:
            continue
        sub = d2[np.ix_(idx, idx)]
        ssw += sub[np.triu_indices(ng, k=1)].sum() / ng

    ssa = sst - ssw
    a = 2
    denom = ssw / (n - a) if (n - a) > 0 and ssw > 0 else np.nan
    f = (ssa / (a - 1)) / denom if denom and np.isfinite(denom) else np.nan
    r2 = ssa / sst if sst > 0 else np.nan
    return f, r2


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def leading_edge_jaccard(
    module_genes: dict[str, list[str]],
    leading_edges: dict[str, list[str]],
) -> pd.DataFrame:
    """Jaccard overlap between each module and each GSEA leading-edge set.

    Returns a modules (rows) x pathways (cols) matrix of Jaccard indices.
    """
    mods = {k: set(v) for k, v in module_genes.items()}
    edges = {k: set(v) for k, v in leading_edges.items()}
    data = {
        mod: {path: _jaccard(genes, egenes) for path, egenes in edges.items()}
        for mod, genes in mods.items()
    }
    return pd.DataFrame(data).T.reindex(index=list(mods), columns=list(edges))


def module_gene_overlap(module_genes: dict[str, list[str]]) -> pd.DataFrame:
    """Symmetric module x module Jaccard overlap matrix (unit diagonal).

    .. deprecated::
        Use :func:`cellquorum.stats.gene_set_overlap.set_overlap_tests`, which returns
        the same Jaccard alongside the intersection, the expected intersection, the fold
        enrichment and a hypergeometric p-value against a stated universe. A similarity
        coefficient on its own cannot be interpreted: the same Jaccard of 0.06 is
        overwhelming evidence between two 8-gene modules and nothing at all between two
        500-gene ones, so a matrix coloured by Jaccard alone ranks pairs wrong. This
        function is kept only so existing callers keep working.
    """
    warnings.warn(
        "module_gene_overlap returns a similarity with no null and no universe, so its "
        "values cannot be ranked against each other; use "
        "cellquorum.stats.set_overlap_tests(sets, universe=...) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    mods = {k: set(v) for k, v in module_genes.items()}
    keys = list(mods)
    mat = pd.DataFrame(0.0, index=keys, columns=keys)
    for a in keys:
        for b in keys:
            mat.loc[a, b] = _jaccard(mods[a], mods[b])
    return mat


def upset_membership(sets: dict[str, list[str]]) -> pd.DataFrame:
    """Element x set boolean membership matrix (the basis for an UpSet plot).

    Rows are the union of all elements; columns are the set names; each cell is
    True if the element belongs to that set. Rows are sorted for determinism.

    .. deprecated::
        Use :func:`cellquorum.stats.gene_set_overlap.exclusive_combinations`, which
        collapses this matrix to the one readout an UpSet plot is drawn to show — how
        many elements each occupied combination holds, and which — and names the genes
        in each. The boolean matrix leaves the caller to do that reduction, and the
        reduction is where the finding is.
    """
    warnings.warn(
        "upset_membership leaves the combination counting to the caller; use "
        "cellquorum.stats.exclusive_combinations(sets) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    setmap = {k: set(v) for k, v in sets.items()}
    elements = sorted(set().union(*setmap.values())) if setmap else []
    data = {name: [el in members for el in elements] for name, members in setmap.items()}
    return pd.DataFrame(data, index=elements)


def program_correlation_matrix(
    scores: pd.DataFrame,
    *,
    method: str = "spearman",
) -> pd.DataFrame:
    """Program x program correlation matrix across cells (Spearman default).

    .. deprecated::
        Use :func:`cellquorum.stats.program_correlation.program_correlation_tests`. This
        function correlates *cells*, and cells within a donor are not independent
        observations, so any p-value attached to its coefficients uses an n one or two
        orders of magnitude too large. It also cannot distinguish a correlation between
        two programs from the condition contrast read twice, or from the arithmetic of two
        programs sharing genes. The replacement names the unit, adjusts for the condition,
        and reports the shared-gene count per pair.
    """
    warnings.warn(
        "program_correlation_matrix correlates cells, which are not independent units, "
        "and carries no significance; use cellquorum.stats.program_correlation_tests("
        "scores, metadata, sample_col=...) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return scores.corr(method=method)
