import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.contracts.layer_tags import set_layer_tag
from cellquorum.differential_expression.stage import DifferentialExpressionStage


def _adata():
    X = sp.csr_matrix(np.random.default_rng(0).poisson(5, size=(20, 10)).astype(float))
    obs = pd.DataFrame(
        {
            "patient_id": (["d1"] * 5 + ["d2"] * 5) * 2,
            "condition": ["Normal"] * 10 + ["LE"] * 10,
        }
    )
    a = ad.AnnData(X=X, obs=obs)
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"G{i}" for i in range(10)]
    set_layer_tag(a, "counts", kind="counts")
    return a


class _Ctx:
    def __init__(self, config):
        self._adata = _adata()
        self.config = config
        self.backend_registry = None

    def require_adata(self):
        return self._adata


def test_stage_name_and_category():
    stage = DifferentialExpressionStage()
    assert stage.name == "differential_expression"
    assert stage.stage_category == "differential_expression"


def test_stage_disabled_returns_recorded_skip():
    class _Cfg:
        differential_expression = {"enabled": False}

    result = DifferentialExpressionStage().run(_Ctx(_Cfg()))
    assert result.metrics.get("skipped") is True
    assert result.metrics.get("reason") == "disabled by config"


def test_stage_selects_default_method_name():
    stage = DifferentialExpressionStage()
    assert stage._select_method_name({}) == "pseudobulk_edger"
    assert stage._select_method_name({"method": "pseudobulk_edger"}) == "pseudobulk_edger"
