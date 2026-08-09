from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.ccc_network.stage import CCCNetworkStage
from cellquorum.methods.registry import METHOD_REGISTRY


class _Paths:
    def __init__(self, tmp):
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Design:
    condition_col = "condition"
    case = "case"
    control = "control"
    donor_col = "patient_id"
    paired = False


class _Config:
    def __init__(self):
        self.design = _Design()
        self.cohort = None
        self.ccc_network = {}


class _Ctx:
    def __init__(self, tmp, adata):
        self.paths = _Paths(tmp)
        self.config = _Config()
        self._adata = adata

    def require_adata(self):
        return self._adata


def _adata():
    obs = pd.DataFrame({"sample": ["s1", "s2"], "condition": ["case", "control"]})
    a = ad.AnnData(X=np.ones((2, 2)), obs=obs)
    a.uns["liana_res"] = pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "source": ["A", "A"],
            "target": ["B", "B"],
            "ligand_complex": ["L1", "L1"],
            "receptor_complex": ["R1", "R1"],
            "magnitude_rank": [0.1, 0.2],
        }
    )
    return a


def test_both_methods_registered():
    assert METHOD_REGISTRY.has("ccc_network", "topology")
    assert METHOD_REGISTRY.has("ccc_network", "ricci")


def test_stage_default_injects_both_methods():
    stage = CCCNetworkStage()
    aug = stage._augment_config(_Ctx.__new__(_Ctx), {})  # design bridge tolerates missing config
    assert [m["method"] for m in aug["methods"]] == ["topology", "ricci"]


def test_stage_runs_topology_end_to_end(tmp_path):
    a = _adata()
    ctx = _Ctx(tmp_path, a)
    result = CCCNetworkStage().run(ctx)
    # Topology always produces output; ricci may skip if dep absent (recorded as warning).
    assert "ccc_network" in result.adata.uns
    assert "topology" in result.adata.uns["ccc_network"]


def test_stage_determinism(tmp_path):
    r1 = CCCNetworkStage().run(_Ctx(tmp_path / "a", _adata()))
    r2 = CCCNetworkStage().run(_Ctx(tmp_path / "b", _adata()))
    t1 = r1.adata.uns["ccc_network"]["topology"]["cci"]
    t2 = r2.adata.uns["ccc_network"]["topology"]["cci"]
    pd.testing.assert_frame_equal(t1, t2)


def test_ccc_network_e2e_populates_uns_and_csvs(tmp_path):
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s2", "s3", "s4"],
            "condition": ["case", "case", "control", "control"],
        }
    )
    a = ad.AnnData(X=np.ones((4, 3)), obs=obs)
    rng = np.random.default_rng(0)
    rows = []
    cts = ["A", "B", "C"]
    for s in ["s1", "s2", "s3", "s4"]:
        for src in cts:
            for tgt in cts:
                if src == tgt:
                    continue
                rows.append(
                    {
                        "sample": s,
                        "source": src,
                        "target": tgt,
                        "ligand_complex": f"L_{src}",
                        "receptor_complex": f"R_{tgt}",
                        "magnitude_rank": float(rng.random()),
                    }
                )
    a.uns["liana_res"] = pd.DataFrame(rows)

    ctx = _Ctx(tmp_path, a)
    result = CCCNetworkStage().run(ctx)

    store = result.adata.uns["ccc_network"]
    assert "topology" in store and "cci" in store["topology"]
    assert (tmp_path / "results" / "ccc_network" / "topology_cci.csv").exists()
    # Comparative present because a design contrast (case vs control) is declared.
    assert (tmp_path / "results" / "ccc_network" / "comparative_cci.csv").exists()
    # No crash, well-formed topology frame.
    assert list(store["topology"]["cci"].columns) == [
        "node",
        "Listener",
        "Influencer",
        "Mediator",
        "Pagerank",
    ]
