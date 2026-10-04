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


def _donor_df(
    *, effect: float, donor_noise: float, n_donors: int = 9, cells: int = 400, seed: int = 0
):
    """Matched design: every donor contributes both conditions, many cells each. ``donor_noise`` is
    a per-(donor, condition) offset, the between-donor variation that cells within a donor share."""
    rng = np.random.default_rng(seed)
    rows = []
    for cluster in ["0", "1"]:
        for d in range(n_donors):
            base = rng.normal(0.4, 0.05)
            for condition, shift in (("Normal", 0.0), ("Lymphedema", effect)):
                offset = rng.normal(0.0, donor_noise)
                vals = rng.normal(base + shift + offset, 0.05, size=cells)
                rows.append(
                    pd.DataFrame(
                        {
                            "cluster": cluster,
                            "condition": condition,
                            "donor": f"P{d}",
                            "score": vals,
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def _violin(df, **kw):
    from cellquorum.stages.state_scoring.viz.plots import state_score_violin

    return state_score_violin(
        df,
        cluster_col="cluster",
        condition_col="condition",
        score_col="score",
        case="Lymphedema",
        control="Normal",
        case_color="#C41E3A",
        control_color="#1B4F8A",
        score_label="score",
        **kw,
    )


def _annotations(fig):
    return [t.get_text() for t in fig.axes[0].texts]


def test_state_score_violin_stars_come_from_donors_not_cells():
    """Real bug: the stars were a cell-level Mann-Whitney U, so with hundreds of cells per donor
    every cluster reached '***' however little the nine donors agreed. Here there is NO condition
    effect: donors differ only by random per-condition offsets, which hundreds of cells per donor
    cannot average away. A test that treats cells as replicates calls this significant; donors
    do not."""
    df = _donor_df(effect=0.0, donor_noise=0.05)
    from scipy.stats import mannwhitneyu

    sub = df[df.cluster == "0"]
    cell_p = mannwhitneyu(
        sub[sub.condition == "Normal"].score, sub[sub.condition == "Lymphedema"].score
    ).pvalue
    assert cell_p < 0.01, "fixture must reproduce the pseudoreplication false positive"

    fig = _violin(df, donor_col="donor", paired=True)
    assert not any("*" in t for t in _annotations(fig))
    assert any("n.s." in t for t in _annotations(fig))


def test_state_score_violin_marks_an_effect_every_donor_shows():
    df = _donor_df(effect=0.3, donor_noise=0.01)
    fig = _violin(df, donor_col="donor", paired=True)
    assert any("*" in t for t in _annotations(fig))


def test_state_score_violin_draws_no_stars_without_a_donor_column():
    """No donor column means no valid test. A cell-level p-value must never stand in for one."""
    df = _donor_df(effect=0.3, donor_noise=0.01).drop(columns="donor")
    fig = _violin(df)
    assert not any("*" in t or "n.s." in t for t in _annotations(fig))


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
