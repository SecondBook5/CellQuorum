import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np

from cellquorum.methods.base import MethodSkip
from cellquorum.trajectory.viz._kernel_plots import DriverVizMethod


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
