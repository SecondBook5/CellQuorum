from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.cell_cell_communication.network.stage import CCCNetworkStage


class _Paths:
    def __init__(self, tmp):
        self.root = tmp
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


def test_comparative_cci_subtracts_control(tmp_path):
    """FIX 1 stage-level test: comparative degrees are case-control, proving NEGATIVE values."""
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "condition": ["case", "control"],
        }
    )
    a = ad.AnnData(X=np.ones((2, 2)), obs=obs)
    # Design the LR table so control has stronger A->B than case.
    a.uns["liana_res"] = pd.DataFrame(
        [
            # Case: A->B (L1/R1) with magnitude_rank=0.8 -> weight=0.2 (WEAK in case)
            {
                "sample": "s1",
                "source": "A",
                "target": "B",
                "ligand_complex": "L1",
                "receptor_complex": "R1",
                "magnitude_rank": 0.8,
            },
            # Control: A->B (L1/R1) with magnitude_rank=0.1 -> weight=0.9 (STRONG in control)
            {
                "sample": "s2",
                "source": "A",
                "target": "B",
                "ligand_complex": "L1",
                "receptor_complex": "R1",
                "magnitude_rank": 0.1,
            },
        ]
    )
    ctx = _Ctx(tmp_path, a)
    result = CCCNetworkStage().run(ctx)

    comp = result.adata.uns["ccc_network"]["comparative"]["cci"]
    # A->B is LOST in case (weight_case - weight_ctrl = 0.2 - 0.9 = -0.7 < 0).
    # So B's Listener (incoming degree) should be NEGATIVE (control subtracted).
    b_row = comp[comp["node"] == "B"].iloc[0]
    assert (
        b_row["Listener"] < 0
    ), "B's Listener should be negative (lost incoming edge from control)"
    # A's Influencer (outgoing degree) should also be negative.
    a_row = comp[comp["node"] == "A"].iloc[0]
    assert (
        a_row["Influencer"] < 0
    ), "A's Influencer should be negative (lost outgoing edge from control)"


# ---------------------------------------------------------------------------
# The comparative arms depend on a column NAME resolving, and it did not.
# Every fixture above names the obs column `sample`, which is the topology/ricci
# method default — so nothing here exercised the real-world case. On the LEC arm
# the object's column is `sample_id` (declared as cohort.sample_key, and used by
# the CCC stage that produced this very LR table), the stage bridged
# condition_col/case/control from the design block but NOT sample_col, and so
# resolve_condition_arms found no `sample` column, returned no arms, and the
# comparative Lymphedema-vs-Normal topology and curvature were never computed.
# It was recorded as a note, so the run reported success.
# ---------------------------------------------------------------------------


class _Cohort:
    sample_key = "sample_id"
    donor_key = "donor_id"
    condition_key = "condition"
    batch_key = None


def _adata_with_sample_id() -> ad.AnnData:
    """Same content as `_adata`, with the sample column named as real runs name it."""
    obs = pd.DataFrame({"sample_id": ["s1", "s2"], "condition": ["case", "control"]})
    a = ad.AnnData(X=np.ones((2, 2)), obs=obs)
    a.uns["liana_res"] = pd.DataFrame(
        [
            {
                "sample": "s1",
                "source": "A",
                "target": "B",
                "ligand_complex": "L1",
                "receptor_complex": "R1",
                "magnitude_rank": 0.8,
            },
            {
                "sample": "s2",
                "source": "A",
                "target": "B",
                "ligand_complex": "L1",
                "receptor_complex": "R1",
                "magnitude_rank": 0.1,
            },
        ]
    )
    return a


def test_stage_bridges_sample_col_from_the_cohort_block():
    stage = CCCNetworkStage()
    ctx = _Ctx.__new__(_Ctx)
    ctx.config = _Config()
    ctx.config.cohort = _Cohort()

    aug = stage._augment_config(ctx, {})

    assert aug["sample_col"] == "sample_id"
    # The condition side was already bridged; both must resolve or the arms do not.
    assert aug["condition_col"] == "condition"


def test_an_explicit_stage_sample_col_still_wins_over_the_cohort_block():
    stage = CCCNetworkStage()
    ctx = _Ctx.__new__(_Ctx)
    ctx.config = _Config()
    ctx.config.cohort = _Cohort()

    aug = stage._augment_config(ctx, {"sample_col": "library"})

    assert aug["sample_col"] == "library"


def test_comparative_runs_when_obs_names_the_sample_column_sample_id(tmp_path):
    """The LEC failure, end to end: comparative output exists and nothing warns."""
    a = _adata_with_sample_id()
    ctx = _Ctx(tmp_path, a)
    ctx.config.cohort = _Cohort()

    result = CCCNetworkStage().run(ctx)

    assert (tmp_path / "results" / "ccc_network" / "comparative_cci.csv").exists()
    assert "comparative" in result.adata.uns["ccc_network"]
    joined = " ".join(result.warnings)
    assert "comparative CCC skipped" not in joined
    assert "an arm is empty" not in joined


def test_gci_over_the_cap_warns_and_says_cci_still_ran(tmp_path):
    """`build_gci` defaults True, and the default cap is below every real cohort.

    The LEC arm produced 370,538 LR rows and the BEC arm 703,452 against
    ``gci_max_edges=200000``, so gene-channel topology and curvature have never once
    been computed on real data — and the reason was a note, so both runs reported
    success with no warning. The message must name the cap (so it is actionable) and
    say that the cell-type-level network WAS computed (so it is not read as a total
    failure of the stage).
    """
    a = _adata_with_sample_id()
    ctx = _Ctx(tmp_path, a)
    ctx.config.cohort = _Cohort()
    ctx.config.ccc_network = {"gci_max_edges": 1}  # 2 LR rows > 1

    result = CCCNetworkStage().run(ctx)

    joined = " ".join(result.warnings)
    assert "gci_max_edges=1" in joined
    assert "NOT computed" in joined
    assert "CCI" in joined
    # CCI is unaffected; GCI produced no table.
    assert (tmp_path / "results" / "ccc_network" / "topology_cci.csv").exists()
    assert not (tmp_path / "results" / "ccc_network" / "topology_gci.csv").exists()


def test_gci_under_the_cap_computes_and_does_not_warn(tmp_path):
    """The complement: a cohort inside the cap gets its gene-channel table silently."""
    a = _adata_with_sample_id()
    ctx = _Ctx(tmp_path, a)
    ctx.config.cohort = _Cohort()

    result = CCCNetworkStage().run(ctx)

    assert (tmp_path / "results" / "ccc_network" / "topology_gci.csv").exists()
    assert "gci_max_edges" not in " ".join(result.warnings)


def test_unresolvable_design_warns_instead_of_noting(tmp_path):
    """No cohort block, so sample_col stays 'sample' and obs has only 'sample_id'.

    The comparison the stage exists to make cannot be built, and that has to reach
    the run's warnings — as a note it was invisible in the report, which counts
    warnings and (now) prints them.
    """
    a = _adata_with_sample_id()
    ctx = _Ctx(tmp_path, a)  # _Config().cohort is None

    result = CCCNetworkStage().run(ctx)

    joined = " ".join(result.warnings)
    assert "comparative CCC skipped" in joined
    # The message must name the missing column and the contrast it cost.
    assert "sample" in joined
    assert "case" in joined and "control" in joined
    assert not (tmp_path / "results" / "ccc_network" / "comparative_cci.csv").exists()
