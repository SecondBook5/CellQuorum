import anndata as ad
import numpy as np
import pytest

from cellquorum.core.contracts.exceptions import CellQuorumContractError
from cellquorum.core.contracts.magic_guard import assert_not_imputed
from cellquorum.stages.integration.embeddings import overlay
from cellquorum.stages.integration.embeddings.config import OverlayConfig


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


def _counts_and_lognorm():
    """An object shaped the way the engine leaves one: X is counts, the layer is lognorm.

    Depth varies fifty-fold across cells, so a score computed on counts and a score
    computed on the normalized layer cannot be confused for each other.
    """
    rng = np.random.default_rng(3)
    depth = np.repeat([1, 50], 20).astype("float32")
    counts = (rng.poisson(4.0, size=(40, 6)) * depth[:, None]).astype("float32")
    a = ad.AnnData(X=counts)
    a.var_names = ["GENE_A", "GENE_B", "GENE_C", "S_1", "G2M_1", "OTHER"]
    lognorm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    a.layers["cellquorum_normalized"] = lognorm.astype("float32")
    return a


def test_a_program_is_scored_on_the_declared_layer_not_on_raw_counts():
    """The default layer is the one every other scoring stage in the engine declares.

    The overlay used to read ``adata.X``, so a program score written to ``obs`` was
    ``score_genes`` over raw counts — on the LEC arm that score spanned -4.4 to 195.3
    in count units and tracked library depth three times as strongly as the same panel
    scored on the normalized layer. The score does not stay in the figure; it lands in
    ``obs``, and nothing downstream can tell which of the two it got.
    """
    a = _counts_and_lognorm()
    cfg = OverlayConfig(genes=["GENE_A"], programs={"prog1": ["GENE_A", "GENE_B"]})

    feats, warnings = overlay.resolve_features(a, cfg, random_state=0)

    assert warnings == []
    lognorm = a.layers["cellquorum_normalized"]
    gene = next(f for f in feats if f.label == "GENE_A")
    np.testing.assert_allclose(gene.values, lognorm[:, 0], rtol=1e-6)
    # The score is on the normalized scale, so it cannot be a count-scale score.
    assert float(np.abs(a.obs["prog1"]).max()) < 10.0


def test_a_missing_layer_falls_back_to_X_and_says_so():
    """Falling back silently is what made the counts score invisible in the first place."""
    a = _counts_and_lognorm()
    del a.layers["cellquorum_normalized"]
    cfg = OverlayConfig(genes=["GENE_A"], programs={"prog1": ["GENE_A", "GENE_B"]})

    feats, warnings = overlay.resolve_features(a, cfg, random_state=0)

    assert any("cellquorum_normalized" in w and "adata.X" in w for w in warnings)
    gene = next(f for f in feats if f.label == "GENE_A")
    np.testing.assert_allclose(gene.values, np.asarray(a.X)[:, 0], rtol=1e-6)


def test_an_explicitly_unset_layer_reads_X_without_complaint():
    """A caller whose X *is* the normalized expression can say so and get no warning."""
    a = _counts_and_lognorm()
    cfg = OverlayConfig(genes=["GENE_A"], layer=None)

    feats, warnings = overlay.resolve_features(a, cfg, random_state=0)

    assert warnings == []
    np.testing.assert_allclose(feats[0].values, np.asarray(a.X)[:, 0], rtol=1e-6)


def test_the_magic_layer_wins_over_the_declared_one():
    """An imputed overlay must paint the imputation, not the layer it was built from."""
    a = _counts_and_lognorm()
    a.layers["magic"] = np.zeros_like(np.asarray(a.X), dtype="float32")
    cfg = OverlayConfig(genes=["GENE_A"])

    feats, warnings = overlay.resolve_features(a, cfg, random_state=0, layer="magic")

    assert warnings == []
    assert not feats[0].values.any()


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
