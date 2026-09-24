import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.trajectory.viz._kernel_plots import DriverVizMethod


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
    bars = sorted(p.name for p in (tmp_path / "f" / "trajectory").glob("drivers_bar_*.png"))
    assert bars == ["drivers_bar_L0.png", "drivers_bar_L1.png"]


def test_renders_driver_bars_and_heatmap(tmp_path):
    # Two lineages → 10 varm columns, but only the two _corr columns are plotted.
    res = DriverVizMethod()._run(
        _adata(), {"figure_formats": ["png"], "dpi": 72, "top_k": 5}, _Ctx(tmp_path)
    )
    assert not isinstance(res, MethodSkip)
    bars = sorted(p.name for p in (tmp_path / "f" / "trajectory").glob("drivers_bar_*.png"))
    heat = list((tmp_path / "f" / "trajectory").glob("drivers_heatmap.png"))
    # Exactly one bar per lineage, labeled by the lineage name (suffix stripped).
    assert bars == ["drivers_bar_L0.png", "drivers_bar_L1.png"]
    assert len(heat) == 1


def test_driver_bar_title_carries_the_fate_label_when_configured(tmp_path):
    """Real complaint: a driver bar's title/filename is just the bare
    macrostate id ("1 drivers"), and a reader has no way to tell what "1"
    means without cross-referencing a separate table. CellRank names a
    macrostate after the dominant cluster id it was seeded from (optionally
    "_1"/"_2" suffixed when a cluster splits into more than one macrostate),
    so a manifest-declared id->label mapping (fate_labels) should resolve
    "1" -> "1 (Perivascular/FRC-like, Inflammatory)" in the figure TITLE,
    without needing to guess or invent biology in the engine itself.
    """
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
    # Filenames stay stable/filesystem-safe on the bare id even when a label
    # is configured -- only the in-figure title changes.
    bars = sorted(p.name for p in (tmp_path / "f" / "trajectory").glob("drivers_bar_*.png"))
    assert bars == ["drivers_bar_1.png", "drivers_bar_8_1.png"]

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
    bars = list((tmp_path / "f" / "trajectory").glob("drivers_bar_*.png"))
    assert len(bars) == 2  # fate_names matches the 2 columns


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
    bars = list((tmp_path / "f" / "trajectory").glob("drivers_bar_*.png"))
    assert len(bars) == 2  # two columns, fallback lineage_N names
