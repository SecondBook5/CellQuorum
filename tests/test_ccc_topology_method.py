from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.cell_cell_communication.network.topology_method import TopologyMethod
from cellquorum.methods.base import MethodSkip


class _Paths:
    def __init__(self, tmp):
        self.root = tmp
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)
        self.config = None


def _liana_res():
    return pd.DataFrame(
        {
            "sample": ["s1", "s1", "s2", "s2"],
            "source": ["A", "B", "A", "B"],
            "target": ["B", "A", "B", "A"],
            "ligand_complex": ["L1", "L2", "L1", "L2"],
            "receptor_complex": ["R1", "R2", "R1", "R2"],
            "magnitude_rank": [0.1, 0.5, 0.2, 0.4],
        }
    )


def _adata_with_obs():
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "condition": ["case", "control"],
        }
    )
    a = ad.AnnData(X=np.ones((2, 2)), obs=obs)
    a.uns["liana_res"] = _liana_res()
    return a


def _config():
    return {
        "source_key": "liana_res",
        "sample_col": "sample",
        "build_gci": True,
        "seed": 42,
        "pagerank_alpha": 0.01,
    }


def test_topology_writes_whole_cohort(tmp_path):
    a = _adata_with_obs()
    res = TopologyMethod().run(a, _config(), _Ctx(tmp_path))
    assert not isinstance(res, MethodSkip)
    assert "ccc_network" in res.adata.uns
    assert "topology" in res.adata.uns["ccc_network"]
    assert "cci" in res.adata.uns["ccc_network"]["topology"]
    assert (tmp_path / "results" / "ccc_network" / "topology_cci.csv").exists()


def test_topology_skips_when_source_absent(tmp_path):
    a = ad.AnnData(X=np.ones((2, 2)), obs=pd.DataFrame({"sample": ["s1", "s2"]}))
    res = TopologyMethod().run(a, _config(), _Ctx(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "liana_res" in res.reason or "source" in res.reason.lower()


def test_topology_comparative_when_design_present(tmp_path):
    a = _adata_with_obs()
    cfg = _config() | {"condition_col": "condition", "case": "case", "control": "control"}
    res = TopologyMethod().run(a, cfg, _Ctx(tmp_path))
    assert not isinstance(res, MethodSkip)
    assert (tmp_path / "results" / "ccc_network" / "comparative_cci.csv").exists()


def test_topology_min_edges_skips_level(tmp_path):
    """FIX 4: min_edges gate skips a level cleanly when edge count is below threshold."""
    a = _adata_with_obs()
    # The toy graph has very few edges; set min_edges above the count.
    cfg = _config() | {"min_edges": 1000}
    res = TopologyMethod().run(a, cfg, _Ctx(tmp_path))
    assert not isinstance(res, MethodSkip)
    # CCI should be skipped due to min_edges gate.
    assert "cci" not in res.adata.uns.get("ccc_network", {}).get("topology", {})
