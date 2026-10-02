# tests/stages/state_scoring/test_state_scoring_viz_plots.py
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def _demo_df(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cluster in ["0", "1"]:
        for condition, shift in (("Normal", 0.0), ("Lymphedema", 0.3)):
            n = 40
            vals = rng.normal(0.4 + shift, 0.05, size=n)
            rows.append(pd.DataFrame({"cluster": cluster, "condition": condition, "score": vals}))
    return pd.concat(rows, ignore_index=True)


def test_state_score_violin_returns_figure_with_one_column_per_cluster():
    from cellquorum.stages.state_scoring.viz.plots import state_score_violin

    df = _demo_df()
    fig = state_score_violin(
        df,
        cluster_col="cluster",
        condition_col="condition",
        score_col="score",
        case="Lymphedema",
        control="Normal",
        case_color="#C41E3A",
        control_color="#1B4F8A",
        score_label="fibrosis_ecm (ucell)",
    )
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1"]


def test_state_score_violin_annotates_pvalue_when_both_conditions_present():
    # The whole point of this figure over the UMAP: a real number behind
    # "higher in Lymphedema", not just a color impression.
    from cellquorum.stages.state_scoring.viz.plots import state_score_violin

    df = _demo_df()
    fig = state_score_violin(
        df,
        cluster_col="cluster",
        condition_col="condition",
        score_col="score",
        case="Lymphedema",
        control="Normal",
        case_color="#C41E3A",
        control_color="#1B4F8A",
        score_label="score",
    )
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("p" in t.lower() or "*" in t or "n.s." in t for t in texts)


def test_state_score_violin_applies_cluster_labels():
    from cellquorum.stages.state_scoring.viz.plots import state_score_violin

    df = _demo_df()
    fig = state_score_violin(
        df,
        cluster_col="cluster",
        condition_col="condition",
        score_col="score",
        case="Lymphedema",
        control="Normal",
        case_color="#C41E3A",
        control_color="#1B4F8A",
        score_label="score",
        cluster_labels={"0": "Perivascular", "1": "Reticular"},
    )
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Perivascular", "Reticular"]


def test_state_score_violin_handles_a_cluster_missing_one_condition():
    from cellquorum.stages.state_scoring.viz.plots import state_score_violin

    df = _demo_df()
    df = df[~((df["cluster"] == "1") & (df["condition"] == "Lymphedema"))]
    fig = state_score_violin(
        df,
        cluster_col="cluster",
        condition_col="condition",
        score_col="score",
        case="Lymphedema",
        control="Normal",
        case_color="#C41E3A",
        control_color="#1B4F8A",
        score_label="score",
    )
    assert isinstance(fig, Figure)  # must not raise


def test_state_score_violin_empty_guarded():
    from cellquorum.stages.state_scoring.viz.plots import state_score_violin

    fig = state_score_violin(
        pd.DataFrame(columns=["cluster", "condition", "score"]),
        cluster_col="cluster",
        condition_col="condition",
        score_col="score",
        case="Lymphedema",
        control="Normal",
        case_color="#C41E3A",
        control_color="#1B4F8A",
        score_label="score",
    )
    assert isinstance(fig, Figure)
