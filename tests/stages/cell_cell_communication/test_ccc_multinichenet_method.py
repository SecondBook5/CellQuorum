from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.cell_cell_communication.multinichenet_method import MultiNicheNetMethod


@pytest.fixture
def mock_context():
    class MockBackend:
        def __init__(self, has_pkg=True):
            self._has_pkg = has_pkg

        def _rscript_available(self) -> bool:
            # Rscript is present; the test varies R-package availability below.
            return True

        def _r_package_available(self, package: str) -> bool:
            return self._has_pkg

    class MockRegistry:
        def __init__(self, backend):
            self._b = backend

        def get(self, name):
            if name == "rscript":
                return self._b
            raise ValueError(name)

    class MockPaths:
        def __init__(self, tmp_path):
            self.root = tmp_path
            self.scratch = tmp_path / "scratch"
            self.results = tmp_path / "results"
            self.scratch.mkdir(parents=True, exist_ok=True)
            self.results.mkdir(parents=True, exist_ok=True)

    class MockContext:
        def __init__(self, tmp_path, has_pkg=True):
            self.paths = MockPaths(tmp_path)
            self.backend_registry = MockRegistry(MockBackend(has_pkg))

    return MockContext


def _toy_adata():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(20, 8)).astype(float))
    obs = pd.DataFrame(
        {
            "cell_type": (["A", "B"] * 10),
            "sample_id": ([f"s{i % 4}" for i in range(20)]),
            "condition": (["case", "ctrl"] * 10),
            "patient_id": ([f"p{i % 4}" for i in range(20)]),
        },
        index=[f"c{i}" for i in range(20)],
    )
    var = pd.DataFrame(index=[f"G{i}" for i in range(8)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_multinichenet_skips_without_contrast(tmp_path, mock_context):
    adata = _toy_adata()
    config = {
        "cell_type_col": "cell_type",
        "sample_col": "sample_id",
        "condition_col": "condition",
    }  # no case/control
    res = MultiNicheNetMethod()._run(adata, config, mock_context(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "contrast" in res.reason.lower()


def test_multinichenet_skips_without_prior_models(tmp_path, mock_context):
    adata = _toy_adata()
    config = {
        "cell_type_col": "cell_type",
        "sample_col": "sample_id",
        "condition_col": "condition",
        "case": "case",
        "control": "ctrl",
    }
    # prior model paths unset
    res = MultiNicheNetMethod()._run(adata, config, mock_context(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "prior" in res.reason.lower() or "model" in res.reason.lower()


class _CapturingContext:
    """A context whose R backend records argv and writes the outputs R would write."""

    def __init__(self, tmp_path, mock_context, *, sample_tbl=None, source_tbl=None):
        self.inner = mock_context(tmp_path)
        self.paths = self.inner.paths
        self.args = None
        self._sample_tbl = sample_tbl
        self._source_tbl = source_tbl
        outer = self

        class Backend:
            def _rscript_available(self):
                return True

            def _r_package_available(self, package):
                return True

            def run_script(self, script, args, timeout=None):
                outer.args = list(args)
                pd.DataFrame(
                    {
                        "contrast": ["case-ctrl"],
                        "group": ["case"],
                        "sender": ["A"],
                        "receiver": ["B"],
                        "ligand": ["L1"],
                        "receptor": ["R1"],
                        "prioritization_score": [0.9],
                    }
                ).to_csv(args[4], index=False)
                if outer._sample_tbl is not None and len(args) >= 22:
                    outer._sample_tbl.to_csv(args[21], index=False)
                if outer._source_tbl is not None and len(args) >= 23:
                    outer._source_tbl.to_csv(args[22], index=False)

                class Proc:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Proc()

        class Registry:
            def get(self, name):
                if name == "rscript":
                    return Backend()
                raise ValueError(name)

        self.backend_registry = Registry()


def _mnn_config(tmp_path):
    lt = tmp_path / "lt.rds"
    lt.write_text("x")
    lr = tmp_path / "lr.rds"
    lr.write_text("x")
    return {
        "cell_type_col": "cell_type",
        "sample_col": "sample_id",
        "condition_col": "condition",
        "case": "case",
        "control": "ctrl",
        "nichenet_ligand_target_matrix": str(lt),
        "nichenet_lr_network": str(lr),
    }


def test_sample_and_source_output_paths_are_passed_and_registered(tmp_path, mock_context):
    sample_tbl = pd.DataFrame(
        {
            "sample": ["s0", "s1"],
            "group": ["case", "ctrl"],
            "sender": ["A", "B"],
            "receiver": ["B", "A"],
            "ligand": ["L1", "L1"],
            "receptor": ["R1", "R1"],
            "scaled_LR_prod": [1.2, -0.4],
            "keep_sender_receiver": ["Sender & Receiver present", "Receiver absent"],
        }
    )
    source_tbl = pd.DataFrame(
        {
            "sender": ["A"],
            "receiver": ["B"],
            "ligand": ["L1"],
            "receptor": ["R1"],
            "scaled_avg_exprs_ligand": [0.8],
            "scaled_avg_frq_ligand": [0.7],
        }
    )
    ctx = _CapturingContext(tmp_path, mock_context, sample_tbl=sample_tbl, source_tbl=source_tbl)
    res = MultiNicheNetMethod()._run(_toy_adata(), _mnn_config(tmp_path), ctx)
    assert not isinstance(res, MethodSkip), getattr(res, "reason", "")
    assert len(ctx.args) == 25
    assert ctx.args[23] == ""
    assert ctx.args[21].endswith("mnn_sample_prioritization.csv")
    assert ctx.args[22].endswith("mnn_group_prioritization_source.csv")
    names = {a.name for a in res.artifacts}
    assert "mnn_sample_prioritization" in names
    assert "mnn_group_prioritization_source" in names


def test_sample_and_source_artifacts_absent_when_r_did_not_write_them(tmp_path, mock_context):
    # Neither table was written by the mock (older-style run) -- must not be
    # claimed as an artifact.
    ctx = _CapturingContext(tmp_path, mock_context)
    res = MultiNicheNetMethod()._run(_toy_adata(), _mnn_config(tmp_path), ctx)
    assert not isinstance(res, MethodSkip), getattr(res, "reason", "")
    names = {a.name for a in res.artifacts}
    assert "mnn_sample_prioritization" not in names
    assert "mnn_group_prioritization_source" not in names


def test_multinichenet_skips_without_r_package(tmp_path, mock_context, monkeypatch):
    monkeypatch.setattr(
        "cellquorum.stages.cell_cell_communication.multinichenet_method.shutil.which",
        lambda name: "/usr/bin/Rscript",
    )
    adata = _toy_adata()
    lt = tmp_path / "lt.rds"
    lt.write_text("x")
    lr = tmp_path / "lr.rds"
    lr.write_text("x")
    config = {
        "cell_type_col": "cell_type",
        "sample_col": "sample_id",
        "condition_col": "condition",
        "case": "case",
        "control": "ctrl",
        "nichenet_ligand_target_matrix": str(lt),
        "nichenet_lr_network": str(lr),
    }
    res = MultiNicheNetMethod()._run(adata, config, mock_context(tmp_path, has_pkg=False))
    assert isinstance(res, MethodSkip)
    assert "package" in res.reason.lower() or "multinichenetr" in res.reason.lower()


def test_failed_canonical_conversion_is_reported_not_swallowed(tmp_path, mock_context, monkeypatch):
    """A conversion failure must reach the user as a warning.

    The canonical table is what ccc_network reads. Swallowing the failure leaves that
    stage with no edges and no explanation, which looks like "nothing was found".
    """
    from cellquorum.stages.cell_cell_communication import multinichenet_method

    def explode(_native):
        raise ValueError("unexpected column layout")

    monkeypatch.setattr(multinichenet_method, "mnn_prioritization_to_canonical", explode)
    ctx = _CapturingContext(tmp_path, mock_context)
    res = MultiNicheNetMethod()._run(_toy_adata(), _mnn_config(tmp_path), ctx)
    assert not isinstance(res, MethodSkip)
    assert any(
        "could not be converted" in w and "unexpected column layout" in w for w in res.warnings
    )
    assert "mnn_canonical_lr" not in {a.name for a in res.artifacts}


def test_paired_mnn_exports_donor_and_excludes_incomplete_pair(tmp_path, mock_context):
    rows = []
    for donor in ["p1", "p2", "p3", "p4"]:
        for condition in ["ctrl", "case"]:
            for cell_type in ["A", "B"]:
                n = 1 if donor == "p4" and condition == "case" and cell_type == "A" else 10
                for _ in range(n):
                    rows.append((donor, condition, cell_type, f"{donor}_{condition}"))
    obs = pd.DataFrame(rows, columns=["donor", "condition", "cell_type", "sample_id"])
    obs.index = obs.index.astype(str)
    a = ad.AnnData(sp.csr_matrix(np.ones((len(obs), 2))), obs=obs)
    a.layers["raw"] = sp.csr_matrix(np.full((len(obs), 2), 3.0))
    cfg = _mnn_config(tmp_path)
    cfg.update(paired=True, donor_col="donor", nichenet_counts_layer="raw")
    ctx = _CapturingContext(tmp_path, mock_context)
    MultiNicheNetMethod()._run(a, cfg, ctx)
    meta = pd.read_csv(ctx.args[3])
    assert set(meta.donor) == {"p1", "p2", "p3"}
    assert meta.groupby("donor").condition.nunique().eq(2).all()
    assert ctx.args[23] == "donor"
    assert ctx.args[24] == "10"
    import scipy.io

    assert np.all(scipy.io.mmread(ctx.args[0]).data == 3)
    audit = pd.read_csv(ctx.paths.scratch / "mnn_pairing_audit.csv")
    assert not audit.loc[audit.donor.eq("p4"), "included_paired_analysis"].any()


def test_paired_mnn_rejects_missing_donor(tmp_path, mock_context):
    cfg = _mnn_config(tmp_path)
    cfg.update(paired=True, donor_col="missing")
    with pytest.raises(ValueError, match="donor_col"):
        MultiNicheNetMethod()._run(_toy_adata(), cfg, mock_context(tmp_path))
