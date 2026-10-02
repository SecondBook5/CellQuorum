"""Tests for PyscenicMethod's radial-GRN wiring: independence from cisTarget,
and the skip-vs-partial-result decision when pySCENIC itself can't run.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.gene_regulation.grn.pyscenic_method import PyscenicMethod

dc = pytest.importorskip("decoupler")


class _Paths:
    def __init__(self, tmp):
        self.root = tmp
        self.results = tmp / "results"
        self.figures = tmp / "figures"
        self.scratch = tmp / "scratch"
        self.results.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)
        self.backend_registry = None


def _adata(n_obs: int = 6) -> ad.AnnData:
    rng = np.random.default_rng(0)
    return ad.AnnData(X=rng.normal(size=(n_obs, 5)))


def _net() -> pd.DataFrame:
    genes = [f"G{i}" for i in range(20)]
    return pd.DataFrame(
        {
            "source": ["TF1"] * 6 + ["TF2"] * 6,
            "target": genes[0:6] + genes[4:10],
            "weight": [1.0] * 12,
        }
    )


def _write_de(tmp, genes) -> None:
    rng = np.random.default_rng(1)
    pd.DataFrame(
        {
            "gene": genes,
            "logFC": rng.normal(size=len(genes)),
            "PValue": rng.uniform(1e-4, 0.5, size=len(genes)),
            "FDR": rng.uniform(1e-3, 0.5, size=len(genes)),
        }
    ).to_csv(tmp / "results" / "de_pseudobulk_edger.csv", index=False)


def _patch_get_net(monkeypatch, net) -> None:
    monkeypatch.setattr(
        "cellquorum.stages.gene_regulation.grn.pyscenic_method.get_net",
        lambda collection, **kw: net.copy(),
    )


def test_render_radial_grn_warns_and_produces_nothing_without_de_csv(tmp_path):
    ctx = _Ctx(tmp_path)
    figs: list = []
    warnings: list = []

    PyscenicMethod()._render_radial_grn(
        context=ctx,
        figures_dir=ctx.paths.figures / "grn",
        radial_collection="dorothea",
        radial_de_results_filename="de_pseudobulk_edger.csv",
        radial_min_size=2,
        radial_n_tf=3,
        radial_per_tf=4,
        radial_n_sectors=2,
        radial_case_color=None,
        radial_control_color=None,
        organism="human",
        group_by="all",
        case=None,
        control=None,
        figs=figs,
        warnings=warnings,
    )

    assert figs == []
    assert any("no DE results" in w for w in warnings)


def test_render_radial_grn_renders_a_real_figure_file(tmp_path, monkeypatch):
    ctx = _Ctx(tmp_path)
    net = _net()
    genes = sorted(set(net["target"]) | {"Gextra1", "Gextra2"})
    _write_de(tmp_path, genes)
    _patch_get_net(monkeypatch, net)
    figs: list = []
    warnings: list = []

    PyscenicMethod()._render_radial_grn(
        context=ctx,
        figures_dir=ctx.paths.figures / "grn",
        radial_collection="dorothea",
        radial_de_results_filename="de_pseudobulk_edger.csv",
        radial_min_size=2,
        radial_n_tf=3,
        radial_per_tf=4,
        radial_n_sectors=2,
        radial_case_color=None,
        radial_control_color=None,
        organism="human",
        group_by="all",
        case="Tumor",
        control="Adjacent",
        figs=figs,
        warnings=warnings,
    )

    assert len(figs) > 0
    assert all(p.exists() for p in figs)


def test_run_returns_partial_result_when_radial_succeeds_but_pyscenic_guard_fails(
    tmp_path, monkeypatch
):
    """A real, common case: cisTarget DBs (or the launcher) aren't configured,
    but the radial GRN figure needs neither -- the stage must still report a
    real StageResult carrying that figure, not a bare MethodSkip that would
    silently discard it."""
    ctx = _Ctx(tmp_path)
    net = _net()
    genes = sorted(set(net["target"]) | {"Gextra1", "Gextra2"})
    _write_de(tmp_path, genes)
    _patch_get_net(monkeypatch, net)

    out = PyscenicMethod()._run(
        _adata(),
        {
            "launcher": "definitely_not_a_real_launcher_xyz",
            "radial_min_size": 2,
            "radial_n_tf": 3,
            "radial_per_tf": 4,
        },
        ctx,
    )

    assert not isinstance(out, MethodSkip)
    assert out.metrics.get("pyscenic_skipped")
    assert any("pySCENIC inference skipped" in w for w in out.warnings)
    assert any(a.kind == "figure" for a in out.artifacts)


def test_run_returns_plain_skip_when_nothing_was_produced(tmp_path):
    """No DE table (so the radial figure also can't render) plus a bad
    launcher -- must fall back to the original plain MethodSkip behavior,
    not fabricate a StageResult with nothing in it."""
    ctx = _Ctx(tmp_path)

    out = PyscenicMethod()._run(_adata(), {"launcher": "definitely_not_a_real_launcher_xyz"}, ctx)

    assert isinstance(out, MethodSkip)
