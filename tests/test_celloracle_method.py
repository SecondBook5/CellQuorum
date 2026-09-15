"""Tests for the CellOracle perturbation method (fake backend; no real celloracle)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.gene_regulation.perturbation.celloracle_method import (
    CellOracleMethod,
)


def _adata(n: int = 300, g: int = 40, condition: bool = False) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    obs = {"cell_type": ["A" if i % 2 else "B" for i in range(n)]}
    if condition:
        obs["condition"] = ["disease" if i % 2 else "healthy" for i in range(n)]
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame(obs, index=[f"cell_{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(g)]),
    )
    a.layers["counts"] = X.copy()
    a.obsm["X_umap"] = rng.random((n, 2)).astype("float32")
    return a


def _ctx(tmp_path: Path, backend):
    paths = SimpleNamespace(
        results=tmp_path / "res", figures=tmp_path / "figs", scratch=tmp_path / "scr"
    )
    reg = SimpleNamespace(get=lambda name: backend)
    return SimpleNamespace(paths=paths, backend_registry=reg, config=None)


def test_skips_when_too_few_cells(tmp_path: Path) -> None:
    res = CellOracleMethod()._run(_adata(n=10), {"min_cells_total": 200}, _ctx(tmp_path, object()))
    assert isinstance(res, MethodSkip)
    assert "too few cells" in res.reason.lower()


def test_skips_when_backend_missing(tmp_path: Path) -> None:
    cfg = {"launcher": "python"}  # resolves on PATH
    res = CellOracleMethod()._run(_adata(), cfg, _ctx(tmp_path, None))
    assert isinstance(res, MethodSkip)


def test_skips_on_timeout(tmp_path: Path) -> None:
    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _s, _a, *, timeout=None):  # noqa: ANN001, ANN003
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    res = CellOracleMethod()._run(_adata(), {"launcher": "python"}, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)


def test_input_contract_does_not_require_condition_or_cluster() -> None:
    m = CellOracleMethod()
    contract = m.input_contract({})
    assert contract.required_obs == []
    assert m.requires_obs({}) == []


def _fake_success_backend(direction: str):
    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _script, args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(args[args.index("--out-dir") + 1])
            tag = args[args.index("--tag") + 1]
            out_dir.mkdir(parents=True, exist_ok=True)
            np.random.seed(0)
            pd.DataFrame(
                {
                    "tf": ["PROX1", "PIEZO1"],
                    "score": [0.9, 0.4],
                    "n_cells": [300, 300],
                    "direction": [direction, direction],
                }
            ).to_csv(out_dir / "perturbation_ranking.csv", index=False)
            # producer's real grn_summary schema (cluster/tf/n_targets)
            pd.DataFrame(
                {
                    "cluster": ["A", "A", "B", "B"],
                    "tf": ["PROX1", "PIEZO1", "PROX1", "PIEZO1"],
                    "n_targets": [12, 5, 8, 3],
                }
            ).to_csv(out_dir / "grn_summary.csv", index=False)
            # a shift-vector parquet so the shift-field figure has input; producer's
            # real d0/d1 column names (not dx/dy)
            idx = [f"cell_{i}" for i in range(300)]
            pd.DataFrame(
                {"d0": np.random.rand(300), "d1": np.random.rand(300)}, index=idx
            ).to_parquet(out_dir / "shift_vectors_PROX1.parquet")
            _ = tag
            return SimpleNamespace(returncode=0, stderr="")

    return FakeBackend()


def test_success_directional_builds_artifacts_and_metrics(tmp_path: Path) -> None:
    res = CellOracleMethod()._run(
        _adata(condition=True),
        {"launcher": "python", "condition_key": "condition", "healthy_label": "healthy"},
        _ctx(tmp_path, _fake_success_backend("directional")),
    )
    assert not isinstance(res, MethodSkip)
    assert res.metrics["n_tfs_screened"] == 2
    assert res.metrics["condition_scored"] is True
    assert res.metrics["cluster_key"] == "cell_type"
    assert res.metrics["n_obs"] == 300
    names = {a.name for a in res.artifacts}
    assert "ranking" in names


def test_success_magnitude_when_no_condition(tmp_path: Path) -> None:
    res = CellOracleMethod()._run(
        _adata(),
        {"launcher": "python"},
        _ctx(tmp_path, _fake_success_backend("magnitude")),
    )
    assert not isinstance(res, MethodSkip)
    assert res.metrics["condition_scored"] is False


def test_graceful_skip_marker_returns_methodskip(tmp_path: Path) -> None:
    """write_skip writes an empty ranking + SKIPPED marker -> must be a MethodSkip."""

    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _script, args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(args[args.index("--out-dir") + 1])
            tag = args[args.index("--tag") + 1]
            out_dir.mkdir(parents=True, exist_ok=True)
            # header-only ranking (mimics write_skip)
            pd.DataFrame(columns=["tf", "score", "n_cells", "direction"]).to_csv(
                out_dir / "perturbation_ranking.csv", index=False
            )
            (out_dir / f"perturbation_SKIPPED_{tag}.txt").write_text(
                "CellOracle skipped: base GRN unavailable\n", encoding="utf-8"
            )
            return SimpleNamespace(returncode=0, stderr="")

    res = CellOracleMethod()._run(_adata(), {"launcher": "python"}, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)
    assert "skipped" in res.reason.lower()


def test_scratch_h5ad_keeps_obsm_but_not_unrelated_layers(tmp_path: Path) -> None:
    """Writing the scratch h5ad must not duplicate the WHOLE cohort.

    adata.copy() (the old code) deep-copies every layer to change which matrix
    is X -- the same "12 GB in a minute, VM died" failure mode already fixed
    once in this exact stage category (grn/pyscenic_method.py,
    coexpression/hdwgcna_method.py). Unlike those two, celloracle_ko.py DOES
    read adata.obsm[embedding_key] (verified directly against the script), so
    obsm must be preserved even while unrelated LAYERS -- the actual multi-GB
    cost, since each is a full gene-space matrix -- are not.
    """

    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, script, args, *, timeout=None):  # noqa: ANN001, ANN003
            return SimpleNamespace(returncode=1, stderr="stop after write")

    a = _adata()
    a.layers["huge_unrelated_layer"] = np.zeros_like(a.layers["counts"])

    res = CellOracleMethod()._run(a, {"launcher": "python"}, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)

    written = ad.read_h5ad(tmp_path / "scr" / "perturbation_input.h5ad")
    assert "huge_unrelated_layer" not in written.layers
    assert "X_umap" in written.obsm
    assert written.obsm["X_umap"].shape == a.obsm["X_umap"].shape


def test_scratch_h5ad_is_restricted_to_highly_variable_genes(tmp_path: Path) -> None:
    """Real bug, confirmed live: CellOracle's own perform_PCA() runs a dense
    SVD over the FULL gene panel handed to it. On a 68,148-cell x 33,417-gene
    fibroblast object that is 2,277,301,716 elements -- just over LAPACK's
    32-bit dense-indexing ceiling -- and the whole stage failed with
    "Indexing a matrix... would incur an in integer overflow in LAPACK".
    CellOracle's own tutorials restrict to a few thousand HVGs before ever
    building the Oracle object; this pipeline never did. Real genes with
    real different variances here (not just a slice) so the test proves HVG
    selection ran, not an arbitrary truncation to the first N columns.
    """
    rng = np.random.default_rng(1)
    n, g = 200, 50
    # A realistic spread of per-gene baseline rates (not all genes sharing
    # one mean -- seurat_v3's loess mean-variance fit needs that spread to
    # be numerically stable; a tiny, artificially uniform toy panel makes it
    # near-singular). Genes 0-4 additionally get a marker-like pattern: a
    # 15% subpopulation expresses them far more strongly -- overdispersed
    # relative to the fitted Poisson trend, which is what seurat_v3 actually
    # ranks on (raw variance alone is not enough: Poisson variance already
    # scales with the mean, so a uniform-rate gene looks "expected").
    base_rates = rng.uniform(1, 8, size=g)
    X = rng.poisson(base_rates[None, :], size=(n, g)).astype("float32")
    marker_mask = rng.random(n) < 0.15
    for j in range(5):
        X[marker_mask, j] = rng.poisson(80, size=int(marker_mask.sum()))
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(g)]),
    )
    a.layers["counts"] = X.copy()
    a.obsm["X_umap"] = rng.random((n, 2)).astype("float32")

    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, script, args, *, timeout=None):  # noqa: ANN001, ANN003
            return SimpleNamespace(returncode=1, stderr="stop after write")

    res = CellOracleMethod()._run(
        a,
        {"launcher": "python", "n_top_genes": 5, "min_cells_total": 10},
        _ctx(tmp_path, FakeBackend()),
    )
    assert isinstance(res, MethodSkip)

    written = ad.read_h5ad(tmp_path / "scr" / "perturbation_input.h5ad")
    assert written.n_vars == 5
    # At least one of the deliberately marker-like genes was kept -- proof
    # real variance-based selection ran (they dominate variance overall;
    # seurat_v3's own local mean-variance normalization can rank the rest of
    # that cluster against each other rather than against the whole panel).
    kept = set(written.var_names)
    assert kept & {f"gene_{j}" for j in range(5)}
    # And it is not the naive "first n_top_genes columns" truncation this
    # guards against: gene_5 (a baseline, non-marker gene) sits right after
    # the marker block, so a truncation bug would never reach it or anything
    # past it, while real selection routinely does.
    assert any(int(name.split("_")[1]) >= 5 for name in kept)


def test_scratch_h5ad_untouched_when_gene_count_already_under_the_cap(tmp_path: Path) -> None:
    """A gene panel already at or below n_top_genes is written as-is -- no
    HVG computation forced on an object too small for it to be meaningful."""

    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, script, args, *, timeout=None):  # noqa: ANN001, ANN003
            return SimpleNamespace(returncode=1, stderr="stop after write")

    a = _adata(g=40)
    res = CellOracleMethod()._run(
        a, {"launcher": "python", "n_top_genes": 2000}, _ctx(tmp_path, FakeBackend())
    )
    assert isinstance(res, MethodSkip)

    written = ad.read_h5ad(tmp_path / "scr" / "perturbation_input.h5ad")
    assert written.n_vars == 40


def test_fate_summary_figure_produced(tmp_path: Path) -> None:
    """The in-process fate summary (FIX 4) is computed and wired into artifacts."""
    res = CellOracleMethod()._run(
        _adata(condition=True),
        {"launcher": "python", "condition_key": "condition", "healthy_label": "healthy"},
        _ctx(tmp_path, _fake_success_backend("directional")),
    )
    assert not isinstance(res, MethodSkip)
    fate_figs = list((tmp_path / "figs" / "perturbation").glob("perturbation_ko_fate_*"))
    assert fate_figs, "expected a perturbation_ko_fate_* figure on disk"
