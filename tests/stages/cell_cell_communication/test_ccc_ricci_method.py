from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from optional_deps import requires

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.cell_cell_communication.network.ricci_method import RicciMethod


class _Paths:
    def __init__(self, tmp):
        self.root = tmp
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)
        self.config = None


def _adata():
    obs = pd.DataFrame({"sample": ["s1", "s2"], "condition": ["case", "control"]})
    a = ad.AnnData(X=np.ones((2, 2)), obs=obs)
    a.uns["liana_res"] = pd.DataFrame(
        {
            "sample": ["s1", "s1", "s2", "s2"],
            "source": ["A", "B", "A", "C"],
            "target": ["B", "C", "B", "A"],
            "ligand_complex": ["L1", "L2", "L1", "L3"],
            "receptor_complex": ["R1", "R2", "R1", "R3"],
            "magnitude_rank": [0.1, 0.3, 0.2, 0.4],
        }
    )
    return a


def _config():
    return {
        "source_key": "liana_res",
        "sample_col": "sample",
        "build_gci": True,
        "ricci_alpha": 0.5,
        "seed": 42,
    }


def test_ricci_absent_skips(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("GraphRicciCurvature"):
            raise ImportError("simulated: no GraphRicciCurvature")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    res = RicciMethod().run(_adata(), _config(), _Ctx(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "curvature" in res.reason.lower() or "graphricci" in res.reason.lower()


def test_ricci_skips_when_source_absent(tmp_path):
    a = ad.AnnData(X=np.ones((1, 2)), obs=pd.DataFrame({"sample": ["s1"]}))
    res = RicciMethod().run(a, _config(), _Ctx(tmp_path))
    assert isinstance(res, MethodSkip)


def test_ricci_computes_curvature_when_available(tmp_path):
    pytest.importorskip("GraphRicciCurvature")
    a = _adata()
    res = RicciMethod().run(a, _config(), _Ctx(tmp_path))
    assert not isinstance(res, MethodSkip)
    assert (tmp_path / "results" / "ccc_network" / "curvature_cci_edges.csv").exists()
    assert "curvature" in res.adata.uns["ccc_network"]


def _complete_liana_res(n_clusters: int = 6) -> pd.DataFrame:
    """A LIANA result covering EVERY ordered cluster pair -- the real, common
    shape (LIANA scores nearly every pair), which built a complete CCI graph
    and made Ollivier-Ricci curvature degenerate (every edge identical)
    before sparsification, confirmed directly on a real 9-cluster cohort."""
    rows = []
    clusters = [str(i) for i in range(n_clusters)]
    for i, src in enumerate(clusters):
        for j, tgt in enumerate(clusters):
            if src == tgt:
                continue
            rows.append(
                {
                    "sample": "s1",
                    "source": src,
                    "target": tgt,
                    "ligand_complex": "L1",
                    "receptor_complex": "R1",
                    "magnitude_rank": 0.01 + 0.05 * abs(i - j),
                }
            )
    return pd.DataFrame(rows)


def test_ricci_sparsifies_the_cci_graph_before_computing_curvature(tmp_path):
    """Without sparsifying first, curvature on a complete graph is
    mathematically degenerate: every edge gets (near) the identical value
    regardless of weight, because every node's neighborhood IS every other
    node -- confirmed directly on a real 9-cluster cohort (36/36 possible
    edges, weights ranging 2061-4347, every edge curved to exactly 0.5625).
    The structural, solver-noise-proof check: a complete 9-cluster LIANA
    result (36 unique unordered pairs) must not still be complete by the
    time curvature is computed -- some edges must have been dropped."""
    pytest.importorskip("GraphRicciCurvature")
    a = ad.AnnData(
        X=np.ones((1, 2)), obs=pd.DataFrame({"sample": ["s1"], "condition": ["control"]})
    )
    a.uns["liana_res"] = _complete_liana_res(n_clusters=9)

    res = RicciMethod().run(a, _config(), _Ctx(tmp_path))

    assert not isinstance(res, MethodSkip)
    edges = pd.read_csv(tmp_path / "results" / "ccc_network" / "curvature_cci_edges.csv")
    complete_graph_edge_count = 9 * 8 // 2
    assert len(edges) < complete_graph_edge_count


def _break_the_solver(monkeypatch):
    """Make Ollivier-Ricci raise, as the optimal-transport solver does on a bad graph."""
    import GraphRicciCurvature.OllivierRicci as orm

    def broken(self, *_args, **_kwargs):
        raise RuntimeError("transport plan did not converge")

    monkeypatch.setattr(orm.OllivierRicci, "compute_ricci_curvature", broken)


@requires("GraphRicciCurvature")
def test_a_solver_failure_is_an_error_not_an_empty_table(monkeypatch):
    """An empty curvature table means "no edges"; a solver failure must not look like that."""
    import networkx as nx

    from cellquorum.stages.cell_cell_communication.network.ricci_method import (
        compute_ricci_curvature,
    )

    _break_the_solver(monkeypatch)
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=1.0)
    graph.add_edge("B", "C", weight=1.0)
    with pytest.raises(RuntimeError, match="did not converge"):
        compute_ricci_curvature(graph)


@requires("GraphRicciCurvature")
def test_a_solver_failure_is_reported_as_a_warning_by_the_stage(tmp_path, monkeypatch):
    _break_the_solver(monkeypatch)
    res = RicciMethod().run(_adata(), _config(), _Ctx(tmp_path))
    assert not isinstance(res, MethodSkip)
    assert any("curvature" in w and "did not converge" in w for w in res.warnings), res.warnings
