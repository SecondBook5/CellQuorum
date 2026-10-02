import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.trajectory.viz._kernel_plots import (
    DriverVizMethod,
    _select_top_drivers_overall,
)


class _Ctx:
    def __init__(self, tmp):
        class P:
            results = tmp / "r"
            figures = tmp / "f"

        self.paths = P()


# Column suffixes cellrank's compute_lineage_drivers emits, per lineage.
_DRIVER_SUFFIXES = ("_corr", "_pval", "_qval", "_ci_low", "_ci_high")


def _adata(n=30, g=12, drivers=True, lineages=("L0", "L1")):
    """AnnData mirroring the cellrank producer's varm layout.

    The producer stores the FULL compute_lineage_drivers frame: for each lineage,
    five columns (<lin>_corr/_pval/_qval/_ci_low/_ci_high), and records the column
    names in uns['trajectory']['cellrank']['driver_columns']. driver_viz must plot
    only the _corr columns, one bar per lineage.
    """
    a = ad.AnnData(np.zeros((n, g), dtype="float32"))
    a.var_names = [f"g{i}" for i in range(g)]
    if drivers:
        cols = [f"{lin}{suf}" for lin in lineages for suf in _DRIVER_SUFFIXES]
        a.varm["cellrank_lineage_drivers"] = (
            np.random.RandomState(0).randn(g, len(cols)).astype("float32")
        )
        a.uns["trajectory"] = {"cellrank": {"fate_names": list(lineages), "driver_columns": cols}}
    return a


def test_select_top_drivers_overall_caps_to_top_k_regardless_of_lineage_count():
    """Real bug: the old selection unioned EACH lineage's own top-K, so total
    rows scaled with n_lineages * top_k -- up to 105 rows for 7 lineages,
    a 43-inch-tall figure. Selecting top-K OVERALL (by max |correlation|
    across any lineage) keeps a fixed cap regardless of lineage count.
    """
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(50, 7))  # 50 genes, 7 lineages
    genes = np.array([f"g{i}" for i in range(50)])
    names, ordered = _select_top_drivers_overall(mat, genes, top_k=10)
    assert len(names) == 10
    assert ordered.shape == (10, 7)


def test_select_top_drivers_overall_picks_the_strongest_genes():
    mat = np.zeros((5, 2))
    mat[0, 0] = 0.9  # strongest overall
    mat[1, 1] = 0.8
    mat[2, 0] = 0.05  # weak, should not be selected
    mat[3, 1] = 0.05
    mat[4, 0] = 0.7
    genes = np.array(["strong_a", "strong_b", "weak_a", "weak_b", "mid_a"])
    names, _ = _select_top_drivers_overall(mat, genes, top_k=3)
    assert set(names) == {"strong_a", "strong_b", "mid_a"}


def test_select_top_drivers_overall_groups_by_peak_lineage_then_magnitude():
    """Rows are ordered by (peak lineage column, then |correlation| within
    that lineage descending) -- genes that peak in the SAME lineage stay
    adjacent, and within a group the strongest driver comes first.
    """
    mat = np.array(
        [
            [0.9, 0.1],  # peaks col 0, strong
            [0.1, 0.9],  # peaks col 1, strong
            [0.5, 0.1],  # peaks col 0, weaker
        ]
    )
    genes = np.array(["a_strong_0", "b_strong_1", "c_weak_0"])
    names, ordered = _select_top_drivers_overall(mat, genes, top_k=3)
    assert names == ["a_strong_0", "c_weak_0", "b_strong_1"]
    assert ordered[0, 0] == 0.9
    assert ordered[1, 0] == 0.5
    assert ordered[2, 1] == 0.9


def test_select_top_drivers_overall_handles_fewer_genes_than_top_k():
    mat = np.random.default_rng(0).normal(size=(3, 2))
    genes = np.array(["a", "b", "c"])
    names, ordered = _select_top_drivers_overall(mat, genes, top_k=10)
    assert len(names) == 3
    assert ordered.shape == (3, 2)


def test_handles_driver_columns_and_fate_names_as_numpy_arrays(tmp_path):
    """Real crash on the fibroblast cohort (same class as fate_viz): both
    driver_columns and fate_names come from adata.uns['trajectory']['cellrank']
    and round-trip through h5ad as numpy arrays, not the plain Python lists
    every existing in-memory test fixture constructs. Two separate
    `if x and len(x) == ...` checks in this method -- one for driver_columns,
    one for fate_names -- each raised "The truth value of an array with more
    than one element is ambiguous" the moment a real array reached them.
    """
    a = _adata()
    cr = a.uns["trajectory"]["cellrank"]
    cr["fate_names"] = np.array(cr["fate_names"], dtype=object)
    cr["driver_columns"] = np.array(cr["driver_columns"], dtype=object)
    res = DriverVizMethod()._run(
        a, {"figure_formats": ["png"], "dpi": 72, "top_k": 5}, _Ctx(tmp_path)
    )
    assert not isinstance(res, MethodSkip)
    heat = list((tmp_path / "f" / "trajectory").glob("drivers_heatmap.png"))
    assert len(heat) == 1


def test_renders_driver_heatmap(tmp_path):
    # Two lineages → 10 varm columns, but only the two _corr columns are plotted.
    res = DriverVizMethod()._run(
        _adata(), {"figure_formats": ["png"], "dpi": 72, "top_k": 5}, _Ctx(tmp_path)
    )
    assert not isinstance(res, MethodSkip)
    heat = list((tmp_path / "f" / "trajectory").glob("drivers_heatmap.png"))
    assert len(heat) == 1


def test_driver_heatmap_columns_carry_the_fate_label_when_configured(tmp_path, monkeypatch):
    """Real complaint: a driver column's label is just the bare macrostate id
    ("1"), and a reader has no way to tell what "1" means without
    cross-referencing a separate table. CellRank names a macrostate after the
    dominant cluster id it was seeded from (optionally "_1"/"_2" suffixed
    when a cluster splits into more than one macrostate), so a
    manifest-declared id->label mapping (fate_labels) should resolve
    "1" -> "1 (Perivascular/FRC-like, Inflammatory)" as the heatmap's column
    label, without needing to guess or invent biology in the engine itself.
    """
    from cellquorum.stages.trajectory.viz import _kernel_plots

    captured = {}
    real_heatmap = _kernel_plots._helpers.matrix_heatmap

    def _spy(matrix, row_labels, col_labels, **kwargs):
        captured["col_labels"] = col_labels
        return real_heatmap(matrix, row_labels, col_labels, **kwargs)

    monkeypatch.setattr(_kernel_plots._helpers, "matrix_heatmap", _spy)

    res = DriverVizMethod()._run(
        _adata(lineages=("1", "8_1")),
        {
            "figure_formats": ["png"],
            "dpi": 72,
            "top_k": 5,
            "fate_labels": {"1": "Perivascular/FRC-like, Inflammatory"},
        },
        _Ctx(tmp_path),
    )
    assert not isinstance(res, MethodSkip)
    assert captured["col_labels"] == ["1 (Perivascular/FRC-like, Inflammatory)", "8_1"]

    # The title-resolution contract itself: given this name and this config,
    # what title text is produced. This is the actual behavior a reader
    # depends on; a saved PNG's pixels aren't inspectable without OCR.
    from cellquorum.stages.trajectory.viz._kernel_plots import _resolve_fate_label

    assert _resolve_fate_label("1", {"1": "Perivascular/FRC-like, Inflammatory"}) == (
        "1 (Perivascular/FRC-like, Inflammatory)"
    )
    assert _resolve_fate_label("8_1", {"8": "Papillary-like, Mechanosensory"}) == (
        "8_1 (Papillary-like, Mechanosensory)"
    )
    # No entry for the base id: unchanged, not a crash or a blank label.
    assert _resolve_fate_label("99", {"1": "x"}) == "99"
    # No mapping at all: unchanged (bare-id behavior preserved by default).
    assert _resolve_fate_label("1", None) == "1"
    assert _resolve_fate_label("1", {}) == "1"


def test_skips_without_drivers(tmp_path):
    res = DriverVizMethod()._run(_adata(drivers=False), {}, _Ctx(tmp_path))
    assert isinstance(res, MethodSkip)


def test_plain_matrix_without_driver_columns_falls_back_per_column(tmp_path):
    """Older runs / plain matrices without driver_columns: each column = one lineage."""
    a = ad.AnnData(np.zeros((30, 12), dtype="float32"))
    a.var_names = [f"g{i}" for i in range(12)]
    a.varm["cellrank_lineage_drivers"] = np.random.RandomState(0).randn(12, 2).astype("float32")
    a.uns["trajectory"] = {"cellrank": {"fate_names": ["L0", "L1"]}}  # no driver_columns
    res = DriverVizMethod()._run(
        a, {"figure_formats": ["png"], "dpi": 72, "top_k": 5}, _Ctx(tmp_path)
    )
    assert not isinstance(res, MethodSkip)
    heat = list((tmp_path / "f" / "trajectory").glob("drivers_heatmap.png"))
    assert len(heat) == 1


def test_skips_with_non_dict_trajectory_uns(tmp_path):
    """Regression: uns['trajectory'] exists but is not a dict → must not raise."""
    a = ad.AnnData(np.zeros((30, 12), dtype="float32"))
    a.var_names = [f"g{i}" for i in range(12)]
    a.varm["cellrank_lineage_drivers"] = np.random.RandomState(0).randn(12, 2).astype("float32")
    a.uns["trajectory"] = "not_a_dict"  # non-dict value
    res = DriverVizMethod()._run(
        a, {"figure_formats": ["png"], "dpi": 72, "top_k": 5}, _Ctx(tmp_path)
    )
    # Must NOT raise; renders with fallback lineage_N names (one per column).
    assert not isinstance(res, MethodSkip)
    heat = list((tmp_path / "f" / "trajectory").glob("drivers_heatmap.png"))
    assert len(heat) == 1
