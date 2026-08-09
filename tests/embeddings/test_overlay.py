import anndata as ad
import numpy as np
import pytest

from cellquorum.contracts.exceptions import CellQuorumContractError
from cellquorum.contracts.magic_guard import assert_not_imputed
from cellquorum.embeddings import overlay
from cellquorum.embeddings.config import OverlayConfig


def _adata():
    rng = np.random.default_rng(0)
    X = rng.random((40, 6)).astype("float32")
    a = ad.AnnData(X=X)
    a.var_names = ["GENE_A", "GENE_B", "GENE_C", "S_1", "G2M_1", "OTHER"]
    a.obs["batch"] = (["x"] * 20) + (["y"] * 20)
    return a


def test_resolve_gene_program_obs():
    a = _adata()
    cfg = OverlayConfig(
        genes=["GENE_A", "MISSING"],
        programs={"prog1": ["GENE_A", "GENE_B"]},
        obs_columns=["batch", "absent_col"],
    )
    feats, warnings = overlay.resolve_features(a, cfg, random_state=0)
    labels = {f.label for f in feats}
    assert "GENE_A" in labels  # gene resolved
    assert "prog1" in labels  # program scored -> obs col
    assert "batch" in labels  # obs column resolved
    assert any("MISSING" in w for w in warnings)  # missing gene warned
    assert any("absent_col" in w for w in warnings)  # missing obs warned
    for f in feats:
        assert f.values.shape[0] == a.n_obs


def test_resolve_cell_cycle():
    a = _adata()
    cfg = OverlayConfig(cell_cycle=True, s_genes=["S_1"], g2m_genes=["G2M_1"])
    feats, _ = overlay.resolve_features(a, cfg, random_state=0)
    labels = {f.label for f in feats}
    assert "S_score" in labels and "G2M_score" in labels


def test_magic_scoped_tags_imputed_and_guard_blocks():
    a = _adata()
    imputed = overlay.impute_magic_scoped(
        a, ["GENE_A", "GENE_B"], knn=5, solver="approximate", random_state=0
    )
    assert set(imputed) == {"GENE_A", "GENE_B"}
    assert "magic" in a.layers
    assert a.layers["magic"].shape == a.X.shape
    with pytest.raises(CellQuorumContractError):
        assert_not_imputed(a, "magic")


def test_magic_unavailable_raises(monkeypatch):
    a = _adata()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "magic":
            raise ImportError("no magic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(overlay.MagicUnavailable):
        overlay.impute_magic_scoped(a, ["GENE_A"], knn=5, solver="approximate", random_state=0)
