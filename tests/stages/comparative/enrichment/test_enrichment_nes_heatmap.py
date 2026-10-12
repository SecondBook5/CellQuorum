"""NES-by-cell-type heatmap (ComplexHeatmap): matrix selection, method behaviour, real render."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.comparative.enrichment.viz.nes_heatmap import (
    GseaNesHeatmapMethod,
    nes_matrices,
)


def _table() -> pd.DataFrame:
    """Four gene sets in three cell types; S4 is significant nowhere, S3 is untested in C."""
    rows = []
    spec = {
        "S1": {"A": (2.4, 0.0001), "B": (-1.1, 0.4), "C": (1.9, 0.01)},
        "S2": {"A": (-2.2, 0.02), "B": (-2.5, 0.0004), "C": (0.3, 0.8)},
        "S3": {"A": (1.2, 0.3), "B": (1.6, 0.03)},
        "S4": {"A": (0.4, 0.7), "B": (-0.6, 0.6), "C": (0.1, 0.9)},
    }
    for source, cells in spec.items():
        for cell_type, (nes, padj) in cells.items():
            rows.append({"cell_type": cell_type, "source": source, "score": nes, "padj": padj})
    return pd.DataFrame(rows)


def test_matrices_are_gene_sets_by_cell_types_and_aligned():
    nes, padj = nes_matrices(_table(), alpha=0.05, min_significant=1, max_rows=None)

    assert list(nes.columns) == ["A", "B", "C"]
    assert nes.shape == padj.shape
    assert list(nes.index) == list(padj.index)
    assert nes.loc["S1", "A"] == 2.4 and padj.loc["S1", "A"] == 0.0001


def test_a_gene_set_significant_in_no_cell_type_is_left_out():
    nes, _ = nes_matrices(_table(), alpha=0.05, min_significant=1, max_rows=None)

    assert "S4" not in nes.index
    assert {"S1", "S2", "S3"} <= set(nes.index)


def test_a_gene_set_a_cell_type_could_not_test_is_missing_there_not_zero():
    nes, padj = nes_matrices(_table(), alpha=0.05, min_significant=1, max_rows=None)

    assert np.isnan(nes.loc["S3", "C"]) and np.isnan(padj.loc["S3", "C"])


def test_requiring_more_significant_cell_types_keeps_fewer_gene_sets():
    nes, _ = nes_matrices(_table(), alpha=0.05, min_significant=2, max_rows=None)

    assert set(nes.index) == {"S1", "S2"}


def test_a_row_cap_keeps_the_gene_sets_with_the_strongest_significant_score():
    nes, _ = nes_matrices(_table(), alpha=0.05, min_significant=1, max_rows=1)

    assert list(nes.index) == ["S2"]  # |NES| 2.5 in B beats S1's 2.4 in A


def test_cell_type_labels_can_be_renamed_for_display():
    nes, padj = nes_matrices(
        _table(), alpha=0.05, min_significant=1, max_rows=None, labels={"A": "Alpha cells"}
    )

    assert list(nes.columns) == ["Alpha cells", "B", "C"]
    assert list(padj.columns) == ["Alpha cells", "B", "C"]


# ---- the method ---------------------------------------------------------------------------


class _FakeBackend:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.calls: list[tuple] = []

    def _rscript_available(self):
        return True

    def _r_package_available(self, pkg):
        return True

    def run_script(self, script, args, *, timeout=None):
        self.calls.append((script, list(args)))
        out = Path(args[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-png")
        out.with_suffix(".pdf").write_bytes(b"fake-pdf")
        return SimpleNamespace(returncode=self.returncode, stderr="")


def _ctx(tmp_path, backend, *, enrichment: pd.DataFrame | None, de: pd.DataFrame | None = None):
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    if enrichment is not None:
        enrichment.to_csv(results / "enrichment_gsea_hallmark_by_cell_type.csv", index=False)
    if de is not None:
        de.to_csv(results / "de_pseudobulk_edger_by_subcluster.csv", index=False)
    paths = SimpleNamespace(
        results=results, figures=tmp_path / "figs", scratch=tmp_path / "scratch"
    )
    return SimpleNamespace(
        paths=paths, backend_registry=SimpleNamespace(get=lambda name: backend), config=None
    )


def _adata() -> ad.AnnData:
    return ad.AnnData(X=np.zeros((2, 2)))


def test_skips_and_names_the_method_that_makes_the_table(tmp_path):
    ctx = _ctx(tmp_path, _FakeBackend(), enrichment=None)

    res = GseaNesHeatmapMethod()._run(_adata(), {}, ctx)

    assert isinstance(res, MethodSkip)
    assert "gsea_by_cell_type" in res.reason


def test_skips_when_nothing_is_significant_anywhere(tmp_path):
    table = _table().assign(padj=0.9)
    ctx = _ctx(tmp_path, _FakeBackend(), enrichment=table)

    res = GseaNesHeatmapMethod()._run(_adata(), {}, ctx)

    assert isinstance(res, MethodSkip)
    assert "significant" in res.reason


def test_renders_one_figure_per_collection_and_reports_the_artifacts(tmp_path):
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, enrichment=_table())

    res = GseaNesHeatmapMethod()._run(_adata(), {"figure_formats": ["png"]}, ctx)

    assert not isinstance(res, MethodSkip)
    assert len(backend.calls) == 1
    out_png = Path(backend.calls[0][1][2])
    assert out_png.name == "gsea_nes_heatmap_hallmark.png"
    assert out_png.parent == tmp_path / "figs" / "enrichment"
    assert {a.name for a in res.artifacts} == {"gsea_nes_heatmap"}


def test_a_cell_type_that_had_too_few_genes_is_named_in_the_figure_footnote(tmp_path):
    de = pd.DataFrame({"cell_type": ["A", "B", "C", "Tiny"], "gene": ["g"] * 4})
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, enrichment=_table(), de=de)

    GseaNesHeatmapMethod()._run(_adata(), {}, ctx)

    footnote = backend.calls[0][1][6]
    assert "Tiny" in footnote
    assert "not tested" in footnote


def test_the_scratch_matrices_the_r_script_reads_are_written(tmp_path):
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, enrichment=_table())

    GseaNesHeatmapMethod()._run(_adata(), {}, ctx)

    nes_csv, padj_csv = (Path(a) for a in backend.calls[0][1][:2])
    assert nes_csv.exists() and padj_csv.exists()
    assert list(pd.read_csv(nes_csv, index_col=0).columns) == ["A", "B", "C"]


def _adata_with_cells() -> ad.AnnData:
    a = ad.AnnData(X=np.zeros((9, 2)))
    a.obs["cell_type"] = ["A"] * 5 + ["B"] * 3 + ["C"]
    return a


def test_the_number_of_cells_behind_each_column_is_handed_to_the_figure(tmp_path):
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, enrichment=_table())

    GseaNesHeatmapMethod()._run(_adata_with_cells(), {}, ctx)

    counts = pd.read_csv(Path(backend.calls[0][1][8]), index_col=0)["n_cells"]
    assert counts.to_dict() == {"A": 5, "B": 3, "C": 1}


def test_the_counts_follow_the_display_labels_of_the_columns(tmp_path):
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, enrichment=_table())

    GseaNesHeatmapMethod()._run(_adata_with_cells(), {"cluster_labels": {"A": "Alpha cells"}}, ctx)

    counts = pd.read_csv(Path(backend.calls[0][1][8]), index_col=0)["n_cells"]
    assert counts["Alpha cells"] == 5 and "A" not in counts.index


def test_without_a_cell_type_column_the_figure_is_still_drawn_without_counts(tmp_path):
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, enrichment=_table())

    res = GseaNesHeatmapMethod()._run(_adata(), {}, ctx)

    assert not isinstance(res, MethodSkip)
    assert backend.calls[0][1][8] == ""


def test_the_palette_is_a_named_option_with_a_default(tmp_path):
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, enrichment=_table())

    GseaNesHeatmapMethod()._run(_adata(), {}, ctx)
    GseaNesHeatmapMethod()._run(_adata(), {"palette": "coolwarm"}, ctx)

    assert backend.calls[0][1][7] == "teal_orange"
    assert backend.calls[1][1][7] == "coolwarm"


def test_an_unknown_palette_is_refused_before_anything_is_drawn(tmp_path):
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, enrichment=_table())

    res = GseaNesHeatmapMethod()._run(_adata(), {"palette": "rainbow"}, ctx)

    assert isinstance(res, MethodSkip)
    assert "rainbow" in res.reason
    assert backend.calls == []


# ---- the real render (needs R + ComplexHeatmap) ---------------------------------------------


def _complexheatmap_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    probe = subprocess.run(
        ["Rscript", "-e", "quit(status = !requireNamespace('ComplexHeatmap', quietly = TRUE))"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _complexheatmap_available(), reason="R with ComplexHeatmap not available")
def test_the_r_script_really_draws_png_and_pdf(tmp_path):
    from cellquorum.backends.script_paths import r_script_path

    nes, padj = nes_matrices(_table(), alpha=0.05, min_significant=1, max_rows=None)
    nes.to_csv(tmp_path / "nes.csv")
    padj.to_csv(tmp_path / "padj.csv")
    out = tmp_path / "heat.png"

    proc = subprocess.run(
        [
            "Rscript",
            str(r_script_path("gsea_nes_heatmap.R")),
            str(tmp_path / "nes.csv"),
            str(tmp_path / "padj.csv"),
            str(out),
            "0.05",
            "0.001",
            "GSEA Hallmark: NES by cell type",
            "B cells not tested (too few genes)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr[-800:]
    assert out.stat().st_size > 5000
    assert out.with_suffix(".pdf").stat().st_size > 1000


def _lineage_ctx(tmp_path, backend, group_col):
    from types import SimpleNamespace as NS

    ctx = _ctx(tmp_path, backend, enrichment=_table())
    ctx.config = NS(differential_expression=NS(cell_type_col=group_col))
    return ctx


def _adata_with_states() -> ad.AnnData:
    a = ad.AnnData(X=np.zeros((9, 2)))
    a.obs["cell_type"] = "LEC"  # the same for every cell: not the grouping the table uses
    a.obs["state"] = ["A"] * 5 + ["B"] * 3 + ["C"]
    return a


def test_the_cell_counts_follow_the_grouping_the_differential_expression_used(tmp_path):
    backend = _FakeBackend()
    ctx = _lineage_ctx(tmp_path, backend, "state")

    GseaNesHeatmapMethod()._run(_adata_with_states(), {}, ctx)

    counts = pd.read_csv(Path(backend.calls[0][1][8]), index_col=0)["n_cells"]
    assert counts.to_dict() == {"A": 5, "B": 3, "C": 1}


def test_the_figure_says_what_its_columns_are(tmp_path):
    backend = _FakeBackend()
    ctx = _lineage_ctx(tmp_path, backend, "state")

    GseaNesHeatmapMethod()._run(_adata_with_states(), {"group_label": "LEC state"}, ctx)

    args = backend.calls[0][1]
    assert args[5] == "GSEA Hallmark: NES by LEC state"
    assert args[9] == "LEC states"


def test_the_footnote_names_the_columns_by_what_they_are(tmp_path):
    de = pd.DataFrame({"cell_type": ["A", "B", "C", "Tiny"], "gene": ["g"] * 4})
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend, enrichment=_table(), de=de)

    GseaNesHeatmapMethod()._run(_adata(), {"group_label": "LEC state"}, ctx)

    footnote = backend.calls[0][1][6]
    assert footnote.startswith("LEC states not tested")
    assert "Tiny" in footnote
