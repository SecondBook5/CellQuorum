# tests/test_de_viz_plots.py
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from cellquorum.de_viz import plots


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
