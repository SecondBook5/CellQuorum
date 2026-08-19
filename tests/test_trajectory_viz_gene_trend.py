import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np

from cellquorum.methods.base import MethodSkip
from cellquorum.trajectory.viz._pseudotime_plots import GeneTrendVizMethod


class _Ctx:
    def __init__(self, tmp):
        class P:
            results = tmp
            figures = tmp / "f"

        self.paths = P()


def test_skips_when_no_genes_requested(tmp_path):
    # No config["genes"] → no defaulted biology → skip.
    res = GeneTrendVizMethod()._run(ad.AnnData(np.zeros((2, 2))), {}, _Ctx(tmp_path))
    assert isinstance(res, MethodSkip)
