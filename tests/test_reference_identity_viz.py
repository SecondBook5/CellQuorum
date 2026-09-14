"""Tests for reference-identity diagnostic figures (one plot per file)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stats.reference_identity import ambiguous_identity_mask, top_reference_probabilities
from cellquorum.visualization.reference_identity import (
    plot_identity_diagnostic_embeddings,
    plot_lineage_confidence_bar,
)


def _adata() -> ad.AnnData:
    n = 20
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.normal(size=(n, 5)))
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    a.obs["cell_type"] = (["Fibroblasts"] * 10) + (["DC"] * 10)
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    a.obs["top_refprob"] = rng.uniform(0.2, 1.0, size=n)
    a.obs["doublet_score"] = rng.uniform(0.0, 1.0, size=n)
    a.obs["n_genes"] = rng.integers(200, 5000, size=n)
    return a


def test_plot_identity_diagnostic_embeddings_applies_the_house_theme(tmp_path, monkeypatch):
    """Without this, the panels render in bare matplotlib defaults instead of
    CellQuorum's house style -- caught by visual review, not by a test, the
    first time this module was written. Regression-guard it now."""
    from cellquorum.visualization import reference_identity as viz_module

    calls = []
    monkeypatch.setattr(viz_module, "apply_theme", lambda: calls.append(True))

    plot_identity_diagnostic_embeddings(
        _adata(),
        basis="X_umap",
        cell_type_col="cell_type",
        top_prob_col="top_refprob",
        doublet_col="doublet_score",
        n_genes_col="n_genes",
        out_dir=tmp_path,
    )

    assert calls, "plot_identity_diagnostic_embeddings must apply the shared house theme"


def test_plot_identity_diagnostic_embeddings_writes_four_standalone_files(tmp_path):
    written = plot_identity_diagnostic_embeddings(
        _adata(),
        basis="X_umap",
        cell_type_col="cell_type",
        top_prob_col="top_refprob",
        doublet_col="doublet_score",
        n_genes_col="n_genes",
        out_dir=tmp_path,
    )

    assert set(written) == {"cell_type", "top_refprob", "doublet_score", "n_genes"}
    for path in written.values():
        assert path.exists()
        assert path.with_suffix(".pdf").exists()


def test_plot_identity_diagnostic_embeddings_skips_missing_columns(tmp_path):
    a = _adata()
    del a.obs["doublet_score"]

    written = plot_identity_diagnostic_embeddings(
        a,
        basis="X_umap",
        cell_type_col="cell_type",
        top_prob_col="top_refprob",
        doublet_col="doublet_score",
        n_genes_col="n_genes",
        out_dir=tmp_path,
    )

    assert "doublet_score" not in written
    assert set(written) == {"cell_type", "top_refprob", "n_genes"}


def test_end_to_end_stats_into_viz_on_a_real_refprob_matrix(tmp_path):
    """The full path: raw refprob columns -> top_reference_probabilities ->
    ambiguous_identity_mask -> a real figure, not just isolated unit checks."""
    n = 12
    rng = np.random.default_rng(1)
    probs = pd.DataFrame(
        rng.dirichlet(np.ones(3), size=n),
        columns=["refprob_Fibroblasts", "refprob_DC", "refprob_TNK"],
        index=[f"c{i}" for i in range(n)],
    )

    a = ad.AnnData(X=rng.normal(size=(n, 4)))
    a.obs_names = probs.index
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    top = top_reference_probabilities(probs)
    a.obs["cell_type"] = pd.Categorical(top["top_label"].str.replace("refprob_", "", regex=False))
    a.obs["top_refprob"] = top["top_prob"].to_numpy()
    a.obs["is_ambiguous"] = ambiguous_identity_mask(top["top_prob"], threshold=0.5).to_numpy()
    a.obs["doublet_score"] = rng.uniform(0, 1, size=n)
    a.obs["n_genes"] = rng.integers(200, 5000, size=n)

    written = plot_identity_diagnostic_embeddings(
        a,
        basis="X_umap",
        cell_type_col="cell_type",
        top_prob_col="top_refprob",
        doublet_col="doublet_score",
        n_genes_col="n_genes",
        out_dir=tmp_path,
    )

    assert len(written) == 4
    for path in written.values():
        assert path.stat().st_size > 0


def test_plot_lineage_confidence_bar_writes_png_and_a_vector_twin(tmp_path):
    entropy = pd.Series([0.1, 0.2, 0.8, 0.9, 0.5], index=[f"c{i}" for i in range(5)])
    cell_type = pd.Series(
        ["Fibroblasts", "Fibroblasts", "DC", "DC", "T/NK"],
        index=entropy.index,
    )

    out = plot_lineage_confidence_bar(entropy, cell_type, tmp_path / "confidence_bar.png")

    assert out == tmp_path / "confidence_bar.png"
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_lineage_confidence_bar_orders_bars_by_ascending_mean_entropy(tmp_path, monkeypatch):
    """Most-confident (lowest entropy) lineage first, matching the reference figure."""
    from cellquorum.visualization import reference_identity as viz_module

    captured = {}

    def fake_save(fig, out_dir, stem, **kwargs):
        captured["fig"] = fig
        return [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]

    monkeypatch.setattr(viz_module, "save_figure", fake_save)

    entropy = pd.Series([0.9, 0.9, 0.1, 0.1, 0.5], index=[f"c{i}" for i in range(5)])
    cell_type = pd.Series(
        ["Fibroblasts", "Fibroblasts", "DC", "DC", "T/NK"],
        index=entropy.index,
    )

    plot_lineage_confidence_bar(entropy, cell_type, tmp_path / "bar.png")

    ax = captured["fig"].axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["DC", "T/NK", "Fibroblasts"]
