"""HVGMethod: writes var['highly_variable'], honors layer discipline + excludes."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.core.contracts import CellQuorumContractError
from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.stages.preprocessing.feature_selection.hvg import HVGMethod


def _counts_adata(n=200, g=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(g - 3)] + ["MT-CO1", "RPS2", "HBB"]
    a.layers["counts"] = X.copy()
    # A PFlog1pPF-like centered lognorm layer (has negatives).
    a.layers["cellquorum_normalized"] = (X - X.mean(0)).astype("float32")
    # Tag the layers using the real helper so contracts see the correct format.
    set_layer_tag(a, "counts", kind="counts")
    set_layer_tag(a, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    return a


def test_seurat_v3_flags_highly_variable_from_counts():
    a = _counts_adata()
    cfg = {
        "method": "seurat_v3",
        "n_top_genes": 20,
        "counts_layer": "counts",
        "lognorm_layer": "cellquorum_normalized",
        "exclude_gene_patterns": [],
    }
    result = HVGMethod().run(a, cfg, context=None)
    assert "highly_variable" in result.adata.var.columns
    assert int(result.adata.var["highly_variable"].sum()) >= 1


def test_exclude_patterns_remove_genes_from_hvg():
    a = _counts_adata()
    cfg = {
        "method": "seurat_v3",
        "n_top_genes": 50,
        "counts_layer": "counts",
        "lognorm_layer": "cellquorum_normalized",
        "exclude_gene_patterns": ["^MT-", "^RP[SL]", "^HB[AB]"],
    }
    result = HVGMethod().run(a, cfg, context=None)
    hv = result.adata.var["highly_variable"]
    assert not bool(hv.get("MT-CO1", False))
    assert not bool(hv.get("RPS2", False))
    assert not bool(hv.get("HBB", False))


def test_seurat_v3_contract_rejects_centered_layer_as_counts():
    # Point counts_layer at the centered (negative) layer -> counts contract must fail.
    a = _counts_adata()
    cfg = {"method": "seurat_v3", "n_top_genes": 20, "counts_layer": "cellquorum_normalized"}
    with pytest.raises(CellQuorumContractError):
        HVGMethod().run(a, cfg, context=None)


def test_pearson_residuals_flags_hvg_from_counts():
    a = _counts_adata()
    cfg = {
        "method": "pearson_residuals",
        "n_top_genes": 20,
        "counts_layer": "counts",
        "lognorm_layer": "cellquorum_normalized",
        "exclude_gene_patterns": [],
    }
    result = HVGMethod().run(a, cfg, context=None)
    assert "highly_variable" in result.adata.var.columns
    assert int(result.adata.var["highly_variable"].sum()) >= 1


def test_seurat_flavor_flags_hvg_from_lognorm():
    a = _counts_adata()
    cfg = {
        "method": "seurat",
        "n_top_genes": 20,
        "counts_layer": "counts",
        "lognorm_layer": "cellquorum_normalized",
        "exclude_gene_patterns": [],
    }
    result = HVGMethod().run(a, cfg, context=None)
    assert "highly_variable" in result.adata.var.columns
    assert int(result.adata.var["highly_variable"].sum()) >= 1


class _Paths:
    def __init__(self, figures):
        self.figures = figures


class _Context:
    def __init__(self, figures):
        self.paths = _Paths(figures)


def test_writes_hvg_figure_by_default_when_context_provides_a_figures_dir(tmp_path):
    a = _counts_adata()
    cfg = {
        "method": "seurat_v3",
        "n_top_genes": 20,
        "counts_layer": "counts",
        "lognorm_layer": "cellquorum_normalized",
        "exclude_gene_patterns": [],
    }
    result = HVGMethod().run(a, cfg, context=_Context(tmp_path))

    figure_artifacts = [art for art in result.artifacts if art.kind == "figure"]
    assert len(figure_artifacts) == 1
    assert figure_artifacts[0].path.exists()


def test_no_figure_written_without_a_usable_context(tmp_path):
    a = _counts_adata()
    cfg = {
        "method": "seurat_v3",
        "n_top_genes": 20,
        "counts_layer": "counts",
        "lognorm_layer": "cellquorum_normalized",
        "exclude_gene_patterns": [],
    }
    result = HVGMethod().run(a, cfg, context=None)
    assert not any(art.kind == "figure" for art in result.artifacts)


def test_write_figures_false_skips_the_figure(tmp_path):
    a = _counts_adata()
    cfg = {
        "method": "seurat_v3",
        "n_top_genes": 20,
        "counts_layer": "counts",
        "lognorm_layer": "cellquorum_normalized",
        "exclude_gene_patterns": [],
        "write_figures": False,
    }
    result = HVGMethod().run(a, cfg, context=_Context(tmp_path))
    assert not any(art.kind == "figure" for art in result.artifacts)


# ═══ A singular seurat_v3 trend fit retries loudly; nothing else is softened ═══════════════


def _singular_below(
    monkeypatch, *, min_genes_ok: int, message: str = "reciprocal condition number 5e-15"
):
    """Make scanpy's HVG raise a loess failure while the gene set exceeds ``min_genes_ok``."""
    from cellquorum.stages.preprocessing.feature_selection import hvg

    real = hvg.sc.pp.highly_variable_genes
    calls: list[int] = []

    def fake(adata, **kwargs):
        calls.append(adata.n_vars)
        if adata.n_vars > min_genes_ok:
            raise ValueError(message)
        real(adata, **kwargs)

    monkeypatch.setattr(hvg.sc.pp, "highly_variable_genes", fake)
    return calls


def _sparse_counts_adata(seed: int = 0) -> ad.AnnData:
    """Genes 0-39 are well expressed; genes 40-79 are expressed in only a handful of cells."""
    rng = np.random.default_rng(seed)
    X = np.zeros((300, 80), dtype="float32")
    X[:, :40] = rng.poisson(3.0, size=(300, 40))
    for gene in range(40, 80):
        X[rng.choice(300, 4, replace=False), gene] = 1.0
    a = ad.AnnData(X=X.copy())
    a.var_names = [f"g{i}" for i in range(80)]
    a.layers["counts"] = X.copy()
    set_layer_tag(a, "counts", kind="counts")
    return a


def test_a_working_first_attempt_is_not_retried_and_reports_no_retry(monkeypatch) -> None:
    calls = _singular_below(monkeypatch, min_genes_ok=10_000)
    result = HVGMethod().run(
        _counts_adata(),
        {"method": "seurat_v3", "n_top_genes": 20, "counts_layer": "counts"},
        context=None,
    )
    assert calls == [60]
    assert result.metrics["retry_min_cells"] is None
    assert not result.warnings


def test_a_singular_fit_retries_on_well_expressed_genes_and_says_so(monkeypatch) -> None:
    calls = _singular_below(monkeypatch, min_genes_ok=40)
    result = HVGMethod().run(
        _sparse_counts_adata(),
        {"method": "seurat_v3", "n_top_genes": 20, "counts_layer": "counts"},
        context=None,
    )
    assert calls == [80, 40]
    assert result.metrics["retry_min_cells"] == 10
    assert any("repeated on genes expressed in at least 10 cells" in w for w in result.warnings)
    var = result.adata.var
    assert int(var["highly_variable"].sum()) == 20
    # The rarely expressed genes were not eligible, so none of them is flagged.
    assert not var.loc[[f"g{i}" for i in range(40, 80)], "highly_variable"].any()


def test_a_different_error_is_not_softened(monkeypatch) -> None:
    _singular_below(monkeypatch, min_genes_ok=0, message="something else went wrong")
    with pytest.raises(ValueError, match="something else"):
        HVGMethod().run(
            _sparse_counts_adata(),
            {"method": "seurat_v3", "n_top_genes": 20, "counts_layer": "counts"},
            context=None,
        )


def test_the_original_error_is_raised_when_no_cell_bar_makes_the_fit_work(monkeypatch) -> None:
    _singular_below(monkeypatch, min_genes_ok=0)
    with pytest.raises(ValueError, match="condition number"):
        HVGMethod().run(
            _sparse_counts_adata(),
            {"method": "seurat_v3", "n_top_genes": 20, "counts_layer": "counts"},
            context=None,
        )
