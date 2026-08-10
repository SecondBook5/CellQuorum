import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np

from cellquorum.methods.base import MethodSkip
from cellquorum.trajectory_viz.driver_viz import DriverVizMethod


class _Ctx:
    def __init__(self, tmp):
        class P:
            results = tmp / "r"
            figures = tmp / "f"

        self.paths = P()


def _adata(n=30, g=12, drivers=True):
    a = ad.AnnData(np.zeros((n, g), dtype="float32"))
    a.var_names = [f"g{i}" for i in range(g)]
    if drivers:
        a.varm["cellrank_lineage_drivers"] = np.random.RandomState(0).randn(g, 2).astype("float32")
        a.uns["trajectory"] = {"cellrank": {"fate_names": ["L0", "L1"]}}
    return a


def test_renders_driver_bars_and_heatmap(tmp_path):
    res = DriverVizMethod()._run(
        _adata(), {"figure_formats": ["png"], "dpi": 72, "top_k": 5}, _Ctx(tmp_path)
    )
    assert not isinstance(res, MethodSkip)
    bars = list((tmp_path / "f" / "trajectory").glob("drivers_bar_*.png"))
    heat = list((tmp_path / "f" / "trajectory").glob("drivers_heatmap.png"))
    assert len(bars) == 2 and len(heat) == 1


def test_skips_without_drivers(tmp_path):
    res = DriverVizMethod()._run(_adata(drivers=False), {}, _Ctx(tmp_path))
    assert isinstance(res, MethodSkip)


def test_skips_with_non_dict_trajectory_uns(tmp_path):
    """Regression: uns['trajectory'] exists but is not a dict → must not raise."""
    a = _adata(drivers=True)
    a.uns["trajectory"] = "not_a_dict"  # non-dict value
    res = DriverVizMethod()._run(
        a, {"figure_formats": ["png"], "dpi": 72, "top_k": 5}, _Ctx(tmp_path)
    )
    # Must NOT raise; renders with fallback lineage_N names
    assert not isinstance(res, MethodSkip)
    bars = list((tmp_path / "f" / "trajectory").glob("drivers_bar_*.png"))
    assert len(bars) == 2  # two lineages, fallback names
