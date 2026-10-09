"""Tests for the pySCENIC GRN method (fake backend; no real pyscenic/DBs)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
from optional_deps import requires

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.gene_regulation.grn.pyscenic_method import PyscenicMethod


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
    # `figures` was missing here, so `getattr(context.paths, "figures", ".")` in
    # the method fell back to "." -- every figure-rendering test wrote real
    # PNG/PDFs into the repo's cwd (a stray grn/ directory) instead of tmp_path.
    paths = SimpleNamespace(
        results=tmp_path / "res", scratch=tmp_path / "scr", figures=tmp_path / "figs"
    )
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


@requires("GraphRicciCurvature")
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

                # Write stub regulons CSV matching pySCENIC's REAL shape: a
                # 2-row column MultiIndex (Enrichment/<metric>) over a 2-level
                # row index (TF, MotifID) -- a flat 2-column CSV does not
                # reproduce what the production reader must parse.
                out_dir.mkdir(parents=True, exist_ok=True)
                regulons_csv = out_dir / f"scenic_regulons_{tag}.csv"
                regulons_csv.write_text(
                    ",,Enrichment,Enrichment\n"
                    ",,AUC,TargetGenes\n"
                    "TF,MotifID,,\n"
                    "STAT1,m1,0.1,\"[('GENE1', 0.9), ('GENE2', 0.5)]\"\n"
                    "IRF1,m1,0.2,\"[('GENE3', 0.8), ('GENE4', 0.4)]\"\n"
                )

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
    assert res.metrics["n_tfs"] == 2  # STAT1, IRF1 -- from the real TF column, not row 0
    assert res.metrics["n_cells_scored"] == 300
    assert res.metrics["group_by"] == "cell_type"
    assert res.metrics["n_obs"] == 300

    # Check artifacts
    artifact_names = {a.name for a in res.artifacts}
    assert "regulons" in artifact_names
    assert "auc_mtx" in artifact_names


@requires("GraphRicciCurvature")
def test_success_path_renders_regulon_curvature_figure(tmp_path: Path) -> None:
    """The regulon curvature network must render from the regulon table alone
    -- it needs no AUCell output, so it must appear even though this test's
    stub AUCell run (same FakeBackend as the success-path test) succeeds."""

    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, script, args, *, timeout=None):  # noqa: ANN001, ANN003
            if "--out-dir" in args:
                out_dir_idx = args.index("--out-dir") + 1
                out_dir = Path(args[out_dir_idx])
                tag_idx = args.index("--tag") + 1
                tag = args[tag_idx]
                out_dir.mkdir(parents=True, exist_ok=True)
                regulons_csv = out_dir / f"scenic_regulons_{tag}.csv"
                regulons_csv.write_text(
                    ",,Enrichment,Enrichment\n"
                    ",,AUC,TargetGenes\n"
                    "TF,MotifID,,\n"
                    "STAT1,m1,0.1,\"[('GENE1', 0.9), ('GENE2', 0.5)]\"\n"
                    "IRF1,m1,0.2,\"[('GENE3', 0.8), ('GENE4', 0.4)]\"\n"
                )
                (out_dir / f"scenic_adjacencies_{tag}.tsv").write_text("TF\ttarget\timportance\n")
                (out_dir / f"scenic_input_{tag}.loom").write_text("")
                return SimpleNamespace(returncode=0, stderr="")
            # AUCell call: fail on purpose -- curvature must still render.
            return SimpleNamespace(returncode=1, stderr="stop before AUCell")

    cfg = {"launcher": "python", **_db_config(tmp_path)}
    m = PyscenicMethod()
    res = m._run(_adata(), cfg, _ctx(tmp_path, FakeBackend()))

    assert not isinstance(res, MethodSkip)
    curvature_figs = [a for a in res.artifacts if "curvature" in a.name]
    assert len(curvature_figs) > 0
    assert all(a.path.exists() for a in curvature_figs)


def test_scratch_h5ad_does_not_carry_unrelated_layers_and_obsm(tmp_path: Path) -> None:
    """Writing the scratch h5ad must not duplicate the WHOLE cohort.

    adata.copy() (the old code) deep-copies every layer and obsm entry to change
    which matrix is X -- the same "12 GB in a minute, VM died" failure mode
    already fixed once in this exact stage category (coexpression/
    hdwgcna_method.py). The GRN script only ever reads /obs and /var axis names
    plus the requested matrix (verified directly against
    pyscenic_scripts/pyscenic_grn.py), so an unrelated large layer/obsm entry
    must not be serialized into the scratch h5ad at all.
    """

    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, script, args, *, timeout=None):  # noqa: ANN001, ANN003
            # Fail fast (a MethodSkip) once the h5ad is written -- this test only
            # cares about what was written to scratch, not the rest of the run.
            return SimpleNamespace(returncode=1, stderr="stop after write")

    a = _adata()
    # An unrelated, sizeable extra layer/obsm the scratch h5ad has no business
    # carrying -- large enough that duplicating it would be the real-world
    # multi-GB waste, small enough to keep the test fast.
    a.layers["huge_unrelated_layer"] = np.zeros((a.n_obs, a.n_vars * 20), dtype="float32")[
        :, : a.n_vars
    ]
    a.obsm["X_huge_unrelated"] = np.zeros((a.n_obs, 50), dtype="float32")

    cfg = {"launcher": "python", "layer": "counts", **_db_config(tmp_path)}
    m = PyscenicMethod()
    res = m._run(a, cfg, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)

    written = ad.read_h5ad(tmp_path / "scr" / "grn_input.h5ad")
    assert "huge_unrelated_layer" not in written.layers
    assert "X_huge_unrelated" not in written.obsm
    assert written.X.shape == (a.n_obs, a.n_vars)
