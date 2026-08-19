"""Tests for the pySCENIC GRN method (fake backend; no real pyscenic/DBs)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.gene_regulation.grn.pyscenic_method import PyscenicMethod
from cellquorum.methods.base import MethodSkip


def _adata(n: int = 300, g: int = 40) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    obs = pd.DataFrame(
        {"cell_type": ["A" if i % 2 else "B" for i in range(n)]},
        index=[f"cell_{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(g)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.layers["counts"] = X.copy()
    return a


def _ctx(tmp_path: Path, backend):
    paths = SimpleNamespace(results=tmp_path / "res", scratch=tmp_path / "scr")
    reg = SimpleNamespace(get=lambda name: backend)
    return SimpleNamespace(paths=paths, backend_registry=reg, config=None)


def _db_config(tmp_path: Path) -> dict:
    # point the DB paths at real existing files so the pre-subprocess gate passes
    tfs = tmp_path / "tfs.txt"
    tfs.write_text("STAT1\n")
    motifs = tmp_path / "motifs.tbl"
    motifs.write_text("x\n")
    feather = tmp_path / "rank.feather"
    feather.write_text("x\n")
    return {
        "tfs_path": str(tfs),
        "motifs_path": str(motifs),
        "rankings_glob": str(feather),
    }


def test_skips_when_too_few_cells(tmp_path: Path) -> None:
    m = PyscenicMethod()
    res = m._run(
        _adata(n=10), {"min_cells_total": 200, **_db_config(tmp_path)}, _ctx(tmp_path, object())
    )
    assert isinstance(res, MethodSkip)
    assert "too few cells" in res.reason.lower()


def test_skips_when_db_paths_unset(tmp_path: Path) -> None:
    m = PyscenicMethod()
    res = m._run(_adata(), {}, _ctx(tmp_path, object()))
    assert isinstance(res, MethodSkip)


def test_skips_when_backend_missing(tmp_path: Path) -> None:
    m = PyscenicMethod()
    # launcher present (monkeypatch shutil.which via config launcher that exists) but backend None
    cfg = {"launcher": "python", **_db_config(tmp_path)}  # 'python' resolves on PATH
    res = m._run(_adata(), cfg, _ctx(tmp_path, None))
    assert isinstance(res, MethodSkip)


def test_skips_on_timeout(tmp_path: Path) -> None:
    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _script, _args, *, timeout=None):  # noqa: ANN001, ANN003
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    cfg = {"launcher": "python", **_db_config(tmp_path)}
    m = PyscenicMethod()
    res = m._run(_adata(), cfg, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)


def test_input_contract_does_not_require_group_by() -> None:
    m = PyscenicMethod()
    contract = m.input_contract({})
    assert "cell_type" not in contract.required_obs
    assert m.requires_obs({}) == []


def test_success_path_builds_artifacts_and_metrics(tmp_path: Path) -> None:
    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, script, args, *, timeout=None):  # noqa: ANN001, ANN003
            # Determine which script was called by inspecting args
            if "--out-dir" in args:
                # This is the grn script
                out_dir_idx = args.index("--out-dir") + 1
                out_dir = Path(args[out_dir_idx])
                tag_idx = args.index("--tag") + 1
                tag = args[tag_idx]

                # Write stub regulons CSV (at least one data row so non-empty guard passes)
                out_dir.mkdir(parents=True, exist_ok=True)
                regulons_csv = out_dir / f"scenic_regulons_{tag}.csv"
                regulons_csv.write_text("TF,TargetGenes\nSTAT1,GENE1;GENE2\nIRF1,GENE3;GENE4\n")

                # Write stub adjacencies
                adjacencies_tsv = out_dir / f"scenic_adjacencies_{tag}.tsv"
                adjacencies_tsv.write_text("TF\ttarget\timportance\n")

                # Write stub loom
                loom_path = out_dir / f"scenic_input_{tag}.loom"
                loom_path.write_text("")

            elif "--loom" in args and "--out" in args:
                # This is the aucell script
                out_idx = args.index("--out") + 1
                out_path = Path(args[out_idx])
                loom_idx = args.index("--loom") + 1
                loom_path = Path(args[loom_idx])

                # Extract tag from loom path to determine cell count
                # Read the grn output dir to determine context
                out_dir = loom_path.parent
                tag = loom_path.stem.replace("scenic_input_", "")

                # Build stub AUC matrix: cells x regulons
                # Need to get cell count from _adata() default (300 cells)
                cell_ids = [f"cell_{i}" for i in range(300)]
                auc_data = pd.DataFrame(
                    {
                        "STAT1(+)": np.random.rand(300),
                        "IRF1(+)": np.random.rand(300),
                    },
                    index=cell_ids,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                auc_data.to_parquet(out_path)

            return SimpleNamespace(returncode=0, stderr="")

    cfg = {"launcher": "python", **_db_config(tmp_path)}
    m = PyscenicMethod()
    res = m._run(_adata(), cfg, _ctx(tmp_path, FakeBackend()))

    # Should NOT be a MethodSkip
    assert not isinstance(res, MethodSkip)

    # Check metrics
    assert res.metrics["n_regulons"] == 2  # 2 AUC columns
    assert res.metrics["n_cells_scored"] == 300
    assert res.metrics["group_by"] == "cell_type"
    assert res.metrics["n_obs"] == 300

    # Check artifacts
    artifact_names = {a.name for a in res.artifacts}
    assert "regulons" in artifact_names
    assert "auc_mtx" in artifact_names
