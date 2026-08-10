import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np

from cellquorum.methods.base import MethodSkip
from cellquorum.trajectory_viz.velocity_viz import VelocityVizMethod


class _Ctx:
    def __init__(self, tmp):
        class P:
            results = tmp
            figures = tmp / "f"

        self.paths = P()


def test_skips_without_velocity_h5ads(tmp_path):
    res = VelocityVizMethod()._run(ad.AnnData(np.zeros((2, 2))), {}, _Ctx(tmp_path))
    assert isinstance(res, MethodSkip)
