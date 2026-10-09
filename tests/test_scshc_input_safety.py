"""sc-SHC input safety: only original integer counts reach R, and an R failure is fatal."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.exceptions import CellQuorumBackendError
from cellquorum.stages.clustering.subclustering.config import SubclusteringConfig
from cellquorum.stages.clustering.subclustering.partition import run_scshc_test


def setup_case():
    a = ad.AnnData(
        X=np.ones((4, 3)),
        obs=pd.DataFrame({"cluster": ["a", "a", "b", "b"]}, index=["c1", "c2", "c3", "c4"]),
    )
    a.layers["counts"] = np.full((4, 3), 0.5)
    cfg = SubclusteringConfig(counts_layer="counts")
    backend = MagicMock()
    backend._r_package_available.return_value = True
    return a, cfg, backend


def test_fractional_input_fails_before_r(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/Rscript")
    a, cfg, b = setup_case()
    with pytest.raises(CellQuorumBackendError, match="fractional"):
        run_scshc_test(a, "cluster", cfg, b, tmp_path)
    b.run_script.assert_not_called()


def test_original_counts_selected_and_runtime_error_is_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/Rscript")
    a, cfg, b = setup_case()
    a.layers["counts_raw"] = np.full((4, 3), 2)

    def fail(script, args, timeout):
        prepared = ad.read_h5ad(args[0])
        assert np.all(prepared.X == 2)
        assert args[-1] == "1337"
        return SimpleNamespace(returncode=1, stdout="starting", stderr="TridiagEigen failed")

    b.run_script.side_effect = fail
    with pytest.raises(CellQuorumBackendError, match="not validated"):
        run_scshc_test(a, "cluster", cfg, b, tmp_path)
    assert (tmp_path / "scshc_stderr.log").read_text() == "TridiagEigen failed"


@pytest.mark.parametrize("value", [np.nan, np.inf, -1])
def test_invalid_original_counts_rejected(tmp_path, monkeypatch, value):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/Rscript")
    a, cfg, b = setup_case()
    a.layers["counts_raw"] = np.full((4, 3), value)
    with pytest.raises(CellQuorumBackendError, match="finite, non-negative"):
        run_scshc_test(a, "cluster", cfg, b, tmp_path)
    b.run_script.assert_not_called()
