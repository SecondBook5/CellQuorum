import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.trajectory.viz._pseudotime_plots import GeneTrendVizMethod


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


def test_uses_fitted_object_time_and_returned_figure(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import matplotlib.pyplot as plt

    fate = ad.AnnData(np.ones((4, 1)))
    fate.var_names = ["Neurog3"]
    fate.obs["palantir_pseudotime"] = np.linspace(0, 1, 4)
    folder = tmp_path / "trajectory" / "cellrank"
    folder.mkdir(parents=True)
    fate.write_h5ad(folder / "fate_mapping.h5ad")
    calls = []

    def trends(adata, *, model, genes, time_key, return_models, show_progress_bar, data_key):
        assert data_key == "X"
        assert show_progress_bar is False
        calls.append((genes, time_key, return_models))
        fig, axis = plt.subplots()
        axis.plot(adata.obs[time_key], adata.X[:, 0])
        return {
            "Neurog3": {"Beta": SimpleNamespace(x_test=adata.obs[time_key], y_test=adata.X[:, 0])}
        }

    fake = SimpleNamespace(
        models=SimpleNamespace(GAM=lambda adata, **kwargs: object()),
        pl=SimpleNamespace(gene_trends=trends),
    )
    monkeypatch.setitem(sys.modules, "cellrank", fake)
    result = GeneTrendVizMethod()._run(
        fate,
        {
            "genes": ["Neurog3"],
            "figure_formats": ["pdf"],
            "pseudotime_keys": ["missing", "palantir_pseudotime"],
        },
        _Ctx(tmp_path),
    )
    assert not isinstance(result, MethodSkip)
    assert calls == [(["Neurog3"], "palantir_pseudotime", True)]
    assert (tmp_path / "f/trajectory/gene_trends.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "trajectory/gene_trends.csv").is_file()
