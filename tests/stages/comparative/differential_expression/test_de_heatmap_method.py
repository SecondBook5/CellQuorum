# tests/stages/comparative/differential_expression/test_de_heatmap_method.py
"""Tests for the ComplexHeatmap-based DE heatmap method (fake backend)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.backends.script_paths import r_script_path
from cellquorum.methods.base import MethodSkip
from cellquorum.stages.comparative.differential_expression.viz.de_heatmap_method import (
    DeHeatmapMethod,
    select_top_genes,
)


def _de_df():
    """6 genes, 2 clusters, all significant -- 2 up + 2 down each, symmetric."""
    return pd.DataFrame(
        {
            "cell_type": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "gene": ["UP0A", "UP0B", "DN0A", "DN0B", "UP1A", "UP1B", "DN1A", "DN1B"],
            "logFC": [3.0, 2.0, -3.0, -2.0, 4.0, 1.0, -4.0, -1.0],
            "FDR": [0.001] * 8,
        }
    )


def test_select_top_genes_takes_top_n_up_and_down_per_group():
    genes = select_top_genes(
        _de_df(),
        group_col="cell_type",
        logfc_col="logFC",
        fdr_col="FDR",
        gene_col="gene",
        fdr_cut=0.05,
        top_n=1,
    )
    # top 1 up + top 1 down per cluster by logFC: UP0A (3.0), DN0A (-3.0),
    # UP1A (4.0), DN1A (-4.0) -- the most extreme value each direction.
    assert genes == sorted(["UP0A", "DN0A", "UP1A", "DN1A"])


def test_select_top_genes_excludes_nonsignificant_rows():
    df = _de_df()
    df.loc[df["gene"] == "UP0A", "FDR"] = 0.9  # no longer significant
    genes = select_top_genes(
        df,
        group_col="cell_type",
        logfc_col="logFC",
        fdr_col="FDR",
        gene_col="gene",
        fdr_cut=0.05,
        top_n=1,
    )
    assert "UP0A" not in genes


def test_select_top_genes_dedupes_across_groups():
    df = _de_df()
    df.loc[df["gene"] == "UP1A", "gene"] = "UP0A"  # same gene, significant in both groups
    genes = select_top_genes(
        df,
        group_col="cell_type",
        logfc_col="logFC",
        fdr_col="FDR",
        gene_col="gene",
        fdr_cut=0.05,
        top_n=1,
    )
    assert genes.count("UP0A") == 1


def _adata(n=40, g=6):
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(n, g)).astype("float32")
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "fibroblast_subcluster": ["0" if i % 2 == 0 else "1" for i in range(n)],
                "condition": ["Lymphedema" if i % 4 < 2 else "Normal" for i in range(n)],
            },
            index=[f"cell_{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=["UP0A", "UP0B", "DN0A", "DN0B", "UP1A", "UP1B"][:g]),
    )
    a.layers["cellquorum_normalized"] = X.copy()
    return a


def _ctx(tmp_path: Path, backend, de_csv: pd.DataFrame | None):
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    if de_csv is not None:
        de_csv.to_csv(results / "de_pseudobulk_edger_by_subcluster.csv", index=False)
    paths = SimpleNamespace(
        results=results, figures=tmp_path / "figs", scratch=tmp_path / "scratch"
    )
    registry = SimpleNamespace(get=lambda name: backend)
    return SimpleNamespace(paths=paths, backend_registry=registry, config=None)


class _FakeBackend:
    def __init__(self, returncode=0, write_output=True):
        self.returncode = returncode
        self.write_output = write_output
        self.calls = []

    def _rscript_available(self):
        return True

    def _r_package_available(self, pkg):
        return True

    def run_script(self, script, args, *, timeout=None):
        self.calls.append((script, args))
        if self.write_output:
            for filename in [args[1], *args[3:]]:
                output = Path(filename)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"fake figure")
        return SimpleNamespace(returncode=self.returncode, stderr="")


def test_skips_when_no_de_csv_present(tmp_path):
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, de_csv=None)
    res = DeHeatmapMethod()._run(_adata(), {}, ctx)
    assert isinstance(res, MethodSkip)
    assert "de_pseudobulk_edger_by_subcluster.csv" in res.reason


def test_skips_when_no_significant_genes(tmp_path):
    backend = _FakeBackend()
    df = _de_df()
    df["FDR"] = 0.9  # nothing significant
    ctx = _ctx(tmp_path, backend, de_csv=df)
    res = DeHeatmapMethod()._run(_adata(), {}, ctx)
    assert isinstance(res, MethodSkip)


def test_cluster_labels_rename_the_matrix_columns_before_rendering(tmp_path):
    """Bare cluster ids ("0", "1"...) are meaningless without the object open --

    cluster_labels lets a project map them to real identities once
    characterized, so the figure itself carries the label instead of asking
    the reader to cross-reference a separate legend every time.
    """
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, de_csv=_de_df())
    DeHeatmapMethod()._run(
        _adata(),
        {
            "cell_type_col": "fibroblast_subcluster",
            "top_n": 2,
            "cluster_labels": {"0": "Papillary", "1": "Reticular"},
        },
        ctx,
    )
    matrix_csv = tmp_path / "scratch" / "de_heatmap_matrix.csv"
    header = matrix_csv.read_text().splitlines()[0]
    assert "Papillary_Lymphedema" in header
    assert "Reticular_Normal" in header
    assert "0_Lymphedema" not in header


def test_success_writes_heatmap_artifact(tmp_path):
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, de_csv=_de_df())
    res = DeHeatmapMethod()._run(
        _adata(), {"cell_type_col": "fibroblast_subcluster", "top_n": 2}, ctx
    )
    assert not isinstance(res, MethodSkip)
    assert len(res.artifacts) == 1
    assert res.artifacts[0].path.exists()
    assert len(backend.calls) == 1


def test_skip_reason_names_the_signal_when_subprocess_is_killed(tmp_path):
    backend = _FakeBackend(returncode=-9, write_output=False)
    ctx = _ctx(tmp_path, backend, de_csv=_de_df())
    res = DeHeatmapMethod()._run(_adata(), {"cell_type_col": "fibroblast_subcluster"}, ctx)
    assert isinstance(res, MethodSkip)
    assert "signal" in res.reason.lower()
    assert "sigkill" in res.reason.lower()


def test_only_upregulated_genes_contribute_to_the_up_quota():
    df = _de_df().query("logFC > 0")
    assert select_top_genes(
        df,
        group_col="cell_type",
        logfc_col="logFC",
        fdr_col="FDR",
        gene_col="gene",
        fdr_cut=0.05,
        top_n=1,
    ) == ["UP0A", "UP1A"]


def test_explicit_arms_preserve_labels_and_reference_order_in_each_cluster(tmp_path):
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, de_csv=_de_df())
    a = _adata()
    a.obs["condition"] = a.obs["condition"].map({"Lymphedema": "a_case", "Normal": "z_ctrl"})
    a.obs.iloc[0, a.obs.columns.get_loc("condition")] = "third_arm"
    result = DeHeatmapMethod()._run(
        a,
        {
            "cell_type_col": "fibroblast_subcluster",
            "cluster_labels": {"0": "cell_A", "1": "cell_B"},
            "case": "a_case",
            "control": "z_ctrl",
            "figure_formats": ["pdf", "png"],
        },
        ctx,
    )
    assert [a.path.suffix for a in result.artifacts] == [".pdf", ".png"]
    groups = pd.read_csv(backend.calls[0][1][2])
    assert groups["cluster"].tolist() == ["cell_A", "cell_A", "cell_B", "cell_B"]
    assert groups["condition"].tolist() == ["z_ctrl", "a_case"] * 2
    matrix = pd.read_csv(backend.calls[0][1][0], index_col=0)
    assert matrix.columns.tolist() == (groups["cluster"] + "_" + groups["condition"]).tolist()


@pytest.mark.r
def test_r_heatmap_renders_one_gene_across_three_populations(tmp_path):
    rscript = shutil.which("Rscript")
    if (
        not rscript
        or subprocess.run(
            [
                rscript,
                "-e",
                'quit(status=as.integer(!requireNamespace("ComplexHeatmap", quietly=TRUE)))',
            ]
        ).returncode
    ):
        pytest.skip("requires Rscript and ComplexHeatmap")
    matrix = tmp_path / "one_gene.csv"
    matrix.write_text("gene,A_0,A_1,B_0,B_1,C_0,C_1\nG1,1,2,3,1,5,8\n")
    output = tmp_path / "one_gene.pdf"
    subprocess.run(
        [rscript, str(r_script_path("de_heatmap.R")), str(matrix), str(output)],
        cwd=tmp_path,
        check=True,
    )
    assert output.read_bytes().startswith(b"%PDF-")
