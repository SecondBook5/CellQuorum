"""Synthetic-fixture unit tests for the reusable module-remodeling stats.

Pure numpy/pandas fixtures built inline — no AnnData, no Rscript, no /mnt/e,
no skip markers. These always run and never trip the real-data skipif leak
(templated on test_mcp_diagnostics.py). They pin the statistical contract the
LEC manuscript depends on: donor-aware effect sizes, a multivariate PERMANOVA,
signature-argmax subtyping with an ambiguity guard, the signed EndoMT-style
contrast index, leading-edge concordance, and program correlations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from cellquorum.stats.module_remodeling import (
    bh_fdr,
    leading_edge_jaccard,
    lmm_effect_sizes,
    module_gene_overlap,
    permanova_by_group,
    program_correlation_matrix,
    signature_argmax_labels,
    signed_program_contrast_index,
)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _paired_frame(effect: float, *, n_donors: int = 6, n_cells: int = 8, seed: int = 0):
    """Build a paired (donor × condition) per-cell score frame for ONE program.

    Each donor contributes ``n_cells`` cells per arm. Within a donor the SAME
    noise vector is mirrored across the two arms, so the only condition
    difference is the planted ``effect`` (added to every case cell). This makes
    the null (effect=0) condition contrast exactly ~0 with non-zero residual
    variance — deterministic, not flaky.
    """
    rng = np.random.default_rng(seed)
    rows = []
    idx = []
    k = 0
    for d in range(n_donors):
        donor = f"D{d}"
        base = rng.normal(0.0, 1.0)  # donor random intercept
        noise = rng.normal(0.0, 0.4, size=n_cells)  # mirrored across arms
        for cond, bump in (("Normal", 0.0), ("Lymphedema", effect)):
            for j in range(n_cells):
                rows.append(
                    {
                        "score": base + noise[j] + bump,
                        "donor_id": donor,
                        "condition": cond,
                        "sample_id": f"{donor}_{cond}",
                        "subtype": "CV",
                    }
                )
                idx.append(f"c{k}")
                k += 1
    meta = pd.DataFrame(rows, index=idx)
    scores = pd.DataFrame({"prog": meta.pop("score")}, index=idx)
    return scores, meta


# --------------------------------------------------------------------------- #
# bh_fdr                                                                        #
# --------------------------------------------------------------------------- #
def test_bh_fdr_matches_statsmodels_on_finite_and_passes_nan_through():
    p = np.array([0.001, 0.2, np.nan, 0.04, 0.5])
    out = bh_fdr(p)
    assert np.isnan(out[2])  # NaN position preserved, not corrected
    finite = np.array([0.001, 0.2, 0.04, 0.5])
    expected = multipletests(finite, method="fdr_bh")[1]
    got = out[[0, 1, 3, 4]]
    assert np.allclose(got, expected)


# --------------------------------------------------------------------------- #
# signature_argmax_labels                                                       #
# --------------------------------------------------------------------------- #
def test_signature_argmax_labels_assigns_dominant_and_guards_ambiguous():
    # Four CHOIR clusters; each of c0/c1/c2 dominant in one signature; c3 is a
    # perfect tie between sig_a and sig_b -> must be left 'unassigned'.
    cell_clusters = {
        "c0": (1.0, 0.0, 0.0),
        "c1": (0.0, 1.0, 0.0),
        "c2": (0.0, 0.0, 1.0),
        "c3": (0.5, 0.5, 0.0),
    }
    rows, clusters, idx = [], [], []
    n = 0
    for cl, (a, b, c) in cell_clusters.items():
        for _ in range(3):  # 3 cells/cluster; mean == the planted value
            rows.append({"sig_a": a, "sig_b": b, "sig_c": c})
            clusters.append(cl)
            idx.append(f"x{n}")
            n += 1
    scores = pd.DataFrame(rows, index=idx)
    cluster_labels = pd.Series(clusters, index=idx)

    out = signature_argmax_labels(scores, cluster_labels, min_margin=0.1)
    lab = out.set_index("cluster")["label"].to_dict()
    assert lab["c0"] == "sig_a"
    assert lab["c1"] == "sig_b"
    assert lab["c2"] == "sig_c"
    assert lab["c3"] == "unassigned"


# --------------------------------------------------------------------------- #
# signed_program_contrast_index (EndoMT index generalization)                   #
# --------------------------------------------------------------------------- #
def test_signed_contrast_index_orders_along_planted_axis_and_centers():
    # Cells arranged on a monotone axis: up programs rise, down program falls.
    n = 20
    t = np.linspace(0, 1, n)
    scores = pd.DataFrame(
        {
            "endomt_lec": t,
            "mesenchymal_gain": t,
            "lec_identity": 1.0 - t,
        },
        index=[f"c{i}" for i in range(n)],
    )
    idx = signed_program_contrast_index(
        scores, up=["endomt_lec", "mesenchymal_gain"], down=["lec_identity"]
    )
    # Monotone increasing along the axis; the mesenchymal end > the LEC end.
    assert idx.iloc[-1] > idx.iloc[0]
    assert np.all(np.diff(idx.values) > 0)
    # Standardized contrast is mean-centered.
    assert abs(float(idx.mean())) < 1e-9


# --------------------------------------------------------------------------- #
# lmm_effect_sizes                                                              #
# --------------------------------------------------------------------------- #
def test_lmm_effect_sizes_recovers_planted_positive_effect():
    scores, meta = _paired_frame(effect=1.5, seed=1)
    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
    )
    row = out.iloc[0]
    assert row["program"] == "prog" and row["group"] == "CV"
    assert row["effect"] > 1.0  # near the planted +1.5
    assert row["ci_low"] > 0.0  # CI excludes 0
    assert row["p_value"] < 0.05
    assert row["fdr"] < 0.05
    assert row["method"] == "lmm"
    assert row["n_donors"] == 6


def test_lmm_effect_sizes_null_effect_is_nonsignificant_and_near_zero():
    scores, meta = _paired_frame(effect=0.0, seed=2)
    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
    )
    row = out.iloc[0]
    assert abs(row["effect"]) < 0.1  # mirrored noise -> ~0 contrast
    assert row["p_value"] > 0.05


def test_lmm_effect_sizes_falls_back_to_paired_t_when_singleton_donors():
    # One donor per arm -> the mixed model's random-intercept grouping is
    # degenerate; the guard must record a paired-t fallback rather than crash.
    scores, meta = _paired_frame(effect=1.0, n_donors=1, seed=3)
    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        min_donors_per_arm=2,
    )
    assert out.iloc[0]["method"] == "paired_t"


# --------------------------------------------------------------------------- #
# permanova_by_group                                                            #
# --------------------------------------------------------------------------- #
def _sample_frame(separation: float, seed: int):
    """Per-sample module vectors for one subtype, 4 case + 4 control samples."""
    rng = np.random.default_rng(seed)
    rows, idx = [], []
    n = 0
    for cond, shift in (("Normal", 0.0), ("Lymphedema", separation)):
        for _s in range(4):
            v = rng.normal(0.0, 0.2, size=3) + np.array([shift, shift, 0.0])
            rows.append({"m1": v[0], "m2": v[1], "m3": v[2], "condition": cond})
            idx.append(f"s{n}")
            n += 1
    meta = pd.DataFrame(
        {"condition": [r.pop("condition") for r in rows]},
        index=idx,
    )
    meta["sample_id"] = idx
    meta["subtype"] = "CV"
    scores = pd.DataFrame(rows, index=idx)
    return scores, meta


def test_permanova_detects_planted_separation_and_ignores_null():
    s_sep, m_sep = _sample_frame(separation=3.0, seed=10)
    out_sep = permanova_by_group(
        s_sep,
        m_sep,
        sample_col="sample_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        n_permutations=199,
        seed=1337,
    )
    assert out_sep.iloc[0]["p_value"] < 0.05
    assert out_sep.iloc[0]["R2"] > 0.5

    s_null, m_null = _sample_frame(separation=0.0, seed=11)
    out_null = permanova_by_group(
        s_null,
        m_null,
        sample_col="sample_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        n_permutations=199,
        seed=1337,
    )
    assert out_null.iloc[0]["p_value"] > 0.05


def test_permanova_is_deterministic_under_fixed_seed():
    s, m = _sample_frame(separation=2.0, seed=12)
    kw = dict(
        sample_col="sample_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        n_permutations=199,
        seed=1337,
    )
    a = permanova_by_group(s, m, **kw).iloc[0]
    b = permanova_by_group(s, m, **kw).iloc[0]
    assert a["pseudo_F"] == b["pseudo_F"]
    assert a["p_value"] == b["p_value"]


# --------------------------------------------------------------------------- #
# concordance + correlations                                                    #
# --------------------------------------------------------------------------- #
def test_leading_edge_jaccard_numerics():
    modules = {"A": ["g1", "g2", "g3"]}
    edges = {"P": ["g2", "g3", "g4"]}
    out = leading_edge_jaccard(modules, edges)
    # |{g2,g3}| / |{g1,g2,g3,g4}| = 2/4
    assert abs(out.loc["A", "P"] - 0.5) < 1e-12


def test_module_gene_overlap_is_symmetric_with_unit_diagonal():
    modules = {"A": ["g1", "g2"], "B": ["g2", "g3", "g4"]}
    out = module_gene_overlap(modules)
    assert out.loc["A", "A"] == 1.0 and out.loc["B", "B"] == 1.0
    assert abs(out.loc["A", "B"] - out.loc["B", "A"]) < 1e-12
    # |{g2}| / |{g1,g2,g3,g4}| = 1/4
    assert abs(out.loc["A", "B"] - 0.25) < 1e-12


def test_program_correlation_matrix_spearman_signs():
    scores = pd.DataFrame(
        {
            "p1": [1.0, 2.0, 3.0, 4.0],
            "p2": [2.0, 4.0, 6.0, 8.0],  # monotone increasing with p1
            "p3": [4.0, 3.0, 2.0, 1.0],  # monotone decreasing with p1
        }
    )
    out = program_correlation_matrix(scores, method="spearman")
    assert abs(out.loc["p1", "p2"] - 1.0) < 1e-12
    assert abs(out.loc["p1", "p3"] + 1.0) < 1e-12
