# tests/stages/comparative/differential_expression/test_de_viz_plots.py
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from cellquorum.stages.comparative.differential_expression.viz import plots


def _demo_df():
    return pd.DataFrame(
        {
            "gene": [f"G{i}" for i in range(6)],
            "logFC": [2.0, -2.5, 0.1, 3.0, -0.05, 1.5],
            "FDR": [1e-6, 1e-4, 0.5, 1e-8, 0.9, 0.2],
        }
    )


def test_volcano_returns_figure_with_expected_axis_labels():
    fig = plots.volcano(
        _demo_df(),
        fc_cut=1.0,
        fdr_cut=0.05,
        case_color="#C41E3A",
        control_color="#1B4F8A",
        x_label="log2 fold change (case - control)",
    )
    ax = fig.axes[0]
    assert ax.get_ylabel() == "-log10(FDR)"
    assert "log2 fold change" in ax.get_xlabel()
    # Top and right spines removed per the contract.
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()


def test_volcano_handles_fdr_zero_without_inf():
    df = _demo_df()
    df.loc[0, "FDR"] = 0.0  # would be inf under -log10 without the floor clip
    fig = plots.volcano(
        df,
        fc_cut=1.0,
        fdr_cut=0.05,
        case_color="#C41E3A",
        control_color="#1B4F8A",
        x_label="x",
    )
    # All y data finite (floor clip applied).
    ys = np.concatenate(
        [c.get_offsets()[:, 1] for c in fig.axes[0].collections if c.get_offsets().size]
    )
    assert np.all(np.isfinite(ys))


def test_volcano_significant_counts_match_mask():
    # 3 significant (G0 up, G1 down, G3 up), G5 fails fc, others fail fdr.
    df = _demo_df()
    fig = plots.volcano(
        df,
        fc_cut=1.0,
        fdr_cut=0.05,
        case_color="#C41E3A",
        control_color="#1B4F8A",
        x_label="x",
    )
    # The stats box text records "2 up, 1 down".
    box_texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("2 up" in t and "1 down" in t for t in box_texts)


def test_volcano_includes_extreme_effects_and_nonsignificant_scale():
    frame = pd.DataFrame({"gene": ["large", "small"], "logFC": [9.0, -8.0], "FDR": [1.0, 1.0]})
    fig = plots.volcano(
        frame,
        fc_cut=1,
        fdr_cut=0.05,
        case_color="red",
        control_color="blue",
        x_label="Effect",
        top_n_labels=0,
    )
    ax = fig.axes[0]
    assert ax.get_xlim()[0] < -8 and ax.get_xlim()[1] > 9
    assert ax.get_ylim()[1] > -np.log10(0.05)


def test_volcano_highlights_marker_genes_even_if_not_significant():
    # G2 defines this cluster's identity but isn't DE between conditions --
    # top_n_labels=0 means no gene would normally get a text label, so any
    # label present must come from highlight_genes, not the significance path.
    df = _demo_df()
    fig = plots.volcano(
        df,
        fc_cut=1.0,
        fdr_cut=0.05,
        case_color="#C41E3A",
        control_color="#1B4F8A",
        x_label="x",
        top_n_labels=0,
        highlight_genes={"G2"},
    )
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert "G2" in labels


def test_volcano_highlight_markers_are_filled_yellow_not_hollow():
    # A hollow black-outline star at this size reads as noise next to the
    # colored dots -- a real, opaque fill is what actually pops.
    df = _demo_df()
    fig = plots.volcano(
        df,
        fc_cut=1.0,
        fdr_cut=0.05,
        case_color="#C41E3A",
        control_color="#1B4F8A",
        x_label="x",
        top_n_labels=0,
        highlight_genes={"G2"},
    )
    # The highlight scatter is the last ax.scatter() call in volcano() --
    # after the main up/down/ambient scatters -- so it's the last collection.
    star = fig.axes[0].collections[-1]
    facecolor = star.get_facecolor()
    assert len(facecolor) > 0, "highlight marker has no fill at all (facecolors='none')"
    assert tuple(facecolor[0])[:3] != (0.0, 0.0, 0.0)  # not black/hollow
    assert facecolor[0][3] > 0.0  # not fully transparent


def test_volcano_highlight_labels_are_always_visually_distinct():
    # A highlighted gene that also happens to be DE-significant previously
    # fell back to the same case/control color as every other label,
    # defeating the point of highlighting it.
    df = _demo_df()
    fig = plots.volcano(
        df,
        fc_cut=0.1,
        fdr_cut=0.5,
        case_color="#C41E3A",
        control_color="#1B4F8A",
        x_label="x",
        top_n_labels=0,
        highlight_genes={"G1"},  # significant AND highlighted
    )
    label = next(t for t in fig.axes[0].texts if t.get_text() == "G1")
    assert label.get_color() not in ("#C41E3A", "#1B4F8A")


def test_volcano_rejects_invalid_probabilities_and_thresholds():
    import pytest

    defaults = dict(
        fc_cut=1, fdr_cut=0.05, case_color="red", control_color="blue", x_label="Effect"
    )
    for value in (-0.1, 1.1):
        frame = _demo_df()
        frame.loc[0, "FDR"] = value
        with pytest.raises(ValueError, match="FDR values"):
            plots.volcano(frame, **defaults)
    for override in ({"fc_cut": -1}, {"fdr_cut": 0}, {"fdr_cut": 1}):
        with pytest.raises(ValueError):
            plots.volcano(_demo_df(), **(defaults | override))


def test_a_bad_volcano_cutoff_names_the_value_that_was_passed():
    import numpy as np
    import pandas as pd
    import pytest

    from cellquorum.stages.comparative.differential_expression.viz.plots import volcano

    table = pd.DataFrame({"gene": ["a"], "logFC": [1.0], "FDR": [0.01]})
    kwargs = dict(case_color="r", control_color="b", x_label="x")
    with pytest.raises(ValueError, match="fc_cut.*-1"):
        volcano(table, fc_cut=-1, fdr_cut=0.05, **kwargs)
    with pytest.raises(ValueError, match="fdr_cut.*1.5"):
        volcano(table, fc_cut=1, fdr_cut=1.5, **kwargs)
    with pytest.raises(ValueError, match="fc_cut.*nan"):
        volcano(table, fc_cut=np.nan, fdr_cut=0.05, **kwargs)
