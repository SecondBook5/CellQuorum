"""Pure-math module-remodeling statistics.

No I/O, no AnnData, no R — just numpy/pandas/scipy/statsmodels transformations
over a per-cell score matrix and a design frame. Each function is independently
testable with tiny synthetic fixtures. The heavier R ``mediation`` call lives in
a separate backend bridge, not here, so this module imports nothing optional.

The house statistical bar is enforced here, not left to the caller:

* pseudoreplication is absorbed by a donor random intercept (LMM), never a raw
  per-cell test;
* every test family is BH-FDR corrected;
* permutations and any sampling are seeded and deterministic;
* guards (>= 2 donors per arm) trigger an explicit, recorded fallback rather
  than a silent crash or a misleading estimate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

DEFAULT_SEED = 1337
UNASSIGNED = "unassigned"


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
    programs: list[str] | None = None,
    groups: list[str] | None = None,
    fdr_method: str = "fdr_bh",
    min_donors_per_arm: int = 2,
) -> pd.DataFrame:
    """Per group x program, fit ``score ~ condition + (1|donor)``.

    The donor random intercept absorbs pseudoreplication (many cells per donor);
    the reported effect is the fixed-effect case-minus-control coefficient with
    a 95% CI, p-value, and BH-FDR across the whole group x program family.

    Guards: a group x program with fewer than ``min_donors_per_arm`` donors in
    either arm, or whose mixed model fails to converge, falls back to a paired
    t-test on per-donor mean scores (donors present in both arms) and records
    ``method='paired_t'``. Everything else records ``method='lmm'``.

    Returns
    -------
    DataFrame, one row per (group, program), with columns: group, program,
    effect, ci_low, ci_high, p_value, fdr, n_case, n_control, n_donors, method.
    """
    progs = list(programs) if programs is not None else list(scores.columns)
    meta = metadata.loc[scores.index]
    grp_values = list(groups) if groups is not None else list(pd.unique(meta[group_col]))

    rows = []
    for group in grp_values:
        gmask = (meta[group_col] == group).to_numpy()
        gmeta = meta.loc[gmask]
        gscores = scores.loc[gmask]
        cond = gmeta[condition_col]
        arm_mask = cond.isin([case, control]).to_numpy()
        gmeta = gmeta.loc[arm_mask]
        gscores = gscores.loc[arm_mask]
        cond = gmeta[condition_col]

        is_case = (cond == case).to_numpy()
        donors_case = set(gmeta[donor_col][is_case])
        donors_control = set(gmeta[donor_col][~is_case])
        n_donors = len(donors_case | donors_control)

        for prog in progs:
            y = gscores[prog].to_numpy(dtype=float)
            df = pd.DataFrame(
                {
                    "score": y,
                    "cond": is_case.astype(float),
                    "donor": gmeta[donor_col].to_numpy(),
                }
            )
            n_case = int(is_case.sum())
            n_control = int((~is_case).sum())

            enough = (
                len(donors_case) >= min_donors_per_arm and len(donors_control) >= min_donors_per_arm
            )
            effect = ci_low = ci_high = p_value = np.nan
            method = "lmm"
            fit_ok = False
            if enough:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        import statsmodels.formula.api as smf

                        model = smf.mixedlm("score ~ cond", df, groups=df["donor"])
                        res = model.fit(reml=False, method="lbfgs")
                    if getattr(res, "converged", True) and np.isfinite(
                        res.params.get("cond", np.nan)
                    ):
                        effect = float(res.params["cond"])
                        p_value = float(res.pvalues["cond"])
                        ci = res.conf_int().loc["cond"]
                        ci_low, ci_high = float(ci[0]), float(ci[1])
                        fit_ok = np.isfinite(p_value)
                except Exception:
                    fit_ok = False

            if not fit_ok:
                method = "paired_t"
                effect, ci_low, ci_high, p_value = _paired_t_effect(
                    df, case_val=1.0, control_val=0.0
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
                    "method": method,
                }
            )

    out = pd.DataFrame(rows)
    out["fdr"] = bh_fdr(out["p_value"].to_numpy(), method=fdr_method)
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
            "method",
        ]
    ]


def _paired_t_effect(
    df: pd.DataFrame, *, case_val: float, control_val: float
) -> tuple[float, float, float, float]:
    """Paired t-test on per-donor mean scores; donors present in both arms.

    Returns (effect, ci_low, ci_high, p_value). ``effect`` is the mean paired
    difference (case - control); the CI is the t-based CI on that difference.
    """
    case = df[df["cond"] == case_val].groupby("donor")["score"].mean()
    control = df[df["cond"] == control_val].groupby("donor")["score"].mean()
    common = case.index.intersection(control.index)
    if len(common) < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    diff = (case.loc[common] - control.loc[common]).to_numpy(dtype=float)
    effect = float(diff.mean())
    n = diff.size
    sd = diff.std(ddof=1)
    if sd == 0:
        p_value = 0.0 if effect != 0 else 1.0
        return (effect, effect, effect, p_value)
    se = sd / np.sqrt(n)
    tstat, p_value = stats.ttest_rel(case.loc[common], control.loc[common])
    tcrit = stats.t.ppf(0.975, df=n - 1)
    return (effect, effect - tcrit * se, effect + tcrit * se, float(p_value))


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
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Multivariate condition effect per group, PERMANOVA (Anderson 2001).

    For each group, per-sample module-score vectors (mean score per program per
    sample) form the observations; the squared-Euclidean pseudo-F between the
    case and control label sets is tested against ``n_permutations`` seeded label
    shuffles. Self-implemented (skbio is not a dependency); deterministic under
    ``seed``.

    Returns
    -------
    DataFrame, one row per group: group, pseudo_F, R2, p_value, n_samples,
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
                "n_samples": n_samples,
                "n_case": n_case,
                "n_control": n_control,
                "n_perm": n_permutations,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


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
    """Symmetric module x module Jaccard overlap matrix (unit diagonal)."""
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
    """
    setmap = {k: set(v) for k, v in sets.items()}
    elements = sorted(set().union(*setmap.values())) if setmap else []
    data = {name: [el in members for el in elements] for name, members in setmap.items()}
    return pd.DataFrame(data, index=elements)


def program_correlation_matrix(
    scores: pd.DataFrame,
    *,
    method: str = "spearman",
) -> pd.DataFrame:
    """Program x program correlation matrix across cells (Spearman default)."""
    return scores.corr(method=method)
