from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import scanpy as sc

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.integration.embeddings.config import MagicConfig, OverlayConfig
from cellquorum.stages.integration.embeddings.methods import (
    CategoricalEmbeddingMethod,
    ContinuousOverlayMethod,
    PagaMethod,
    PhateMethod,
    UmapMethod,
    _overlay_with_declared_panels,
)


class _Ctx:
    random_seed = 1337


def _adata():
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.random((50, 8)).astype("float32"))
    a.obsm["X_pca_harmony"] = rng.normal(size=(50, 6)).astype("float32")
    a.obs["cell_type"] = ["A"] * 25 + ["B"] * 25
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    sc.pp.neighbors(a, use_rep="X_pca_harmony", random_state=0)
    return a


def _cfg(**kw):
    base = {
        "use_rep": "X_pca_harmony",
        "umap_min_dist": 0.3,
        "phate_knn": 15,
        "phate_decay": 40,
        "random_state": 1337,
        "paga_groupby": None,
        "cell_type_key": "cell_type",
        "cluster_key": "leiden",
    }
    base.update(kw)
    return base


def test_umap_method_writes_obsm():
    a = _adata()
    res = UmapMethod().run(a, _cfg(), _Ctx())
    assert not isinstance(res, MethodSkip)
    assert "X_umap" in res.adata.obsm


def test_umap_method_skips_without_neighbors():
    a = ad.AnnData(X=np.zeros((10, 4), dtype="float32"))
    res = UmapMethod().run(a, _cfg(), _Ctx())
    assert isinstance(res, MethodSkip)


def test_phate_method_writes_obsm():
    a = _adata()
    res = PhateMethod().run(a, _cfg(), _Ctx())
    assert not isinstance(res, MethodSkip)
    assert "X_phate" in res.adata.obsm


def test_paga_method_resolves_cell_type():
    a = _adata()
    res = PagaMethod().run(a, _cfg(), _Ctx())
    assert not isinstance(res, MethodSkip)
    assert "paga" in res.adata.uns
    assert res.metrics["groupby"] == "cell_type"


def test_paga_method_skips_when_no_group():
    a = _adata()
    del a.obs["cell_type"]
    res = PagaMethod().run(a, _cfg(cluster_key="leiden"), _Ctx())
    assert isinstance(res, MethodSkip)


# --- Task 6: Render methods ---


class _CtxPaths:
    random_seed = 1337

    def __init__(self, tmp):
        self.paths = type("P", (), {"figures": str(tmp), "results": str(tmp)})()


def _adata_rendered():
    a = _adata()
    sc.tl.umap(a, random_state=0)
    sc.tl.paga(a, groups="cell_type")
    a.var_names = [f"GENE_{i}" for i in range(a.n_vars)]
    return a


def test_categorical_method_renders_figures(tmp_path):
    a = _adata_rendered()
    cfg = _cfg(embeddings=["umap"], figure_formats=["png"], dpi=80)
    res = CategoricalEmbeddingMethod().run(a, cfg, _CtxPaths(tmp_path))
    assert not isinstance(res, MethodSkip)
    assert res.metrics["n_figures"] >= 1
    assert any(p.suffix == ".png" for art in res.artifacts for p in [Path(art.path)])


def test_overlay_method_renders_gene(tmp_path):
    a = _adata_rendered()
    cfg = _cfg(
        embeddings=["umap"],
        figure_formats=["png"],
        dpi=80,
        overlay=OverlayConfig(genes=["GENE_0"]),
        magic=MagicConfig(enabled=False),
    )
    res = ContinuousOverlayMethod().run(a, cfg, _CtxPaths(tmp_path))
    assert not isinstance(res, MethodSkip)
    assert res.metrics["n_figures"] >= 1


def test_overlay_method_skips_when_nothing_requested(tmp_path):
    a = _adata_rendered()
    cfg = _cfg(embeddings=["umap"], overlay=OverlayConfig(), magic=MagicConfig(enabled=False))
    res = ContinuousOverlayMethod().run(a, cfg, _CtxPaths(tmp_path))
    assert isinstance(res, MethodSkip)


def test_categorical_method_skips_without_embedding(tmp_path):
    a = _adata()  # no X_umap computed
    cfg = _cfg(embeddings=["umap"])
    res = CategoricalEmbeddingMethod().run(a, cfg, _CtxPaths(tmp_path))
    assert isinstance(res, MethodSkip)


def test_overlay_filename_collision_deduplicates(tmp_path):
    """Two features sanitizing to the same stem produce distinct files + warning."""
    a = _adata_rendered()
    # Two genes whose labels sanitize to the same stem: "CD8+" and "CD8/" both → "CD8_"
    a.var_names = [f"GENE_{i}" for i in range(a.n_vars - 2)] + ["CD8+", "CD8/"]
    cfg = _cfg(
        embeddings=["umap"],
        figure_formats=["png"],
        dpi=80,
        overlay=OverlayConfig(genes=["CD8+", "CD8/"]),
        magic=MagicConfig(enabled=False),
    )
    res = ContinuousOverlayMethod().run(a, cfg, _CtxPaths(tmp_path))
    assert not isinstance(res, MethodSkip)
    # Both figures must be saved (2 artifacts for 2 genes).
    assert res.metrics["n_figures"] == 2
    # A warning about the collision must be recorded.
    assert any("CD8" in w for w in res.warnings), "Expected collision warning not found"
    # Both files must exist and be distinct.
    figs_dir = tmp_path / "embeddings"
    saved_files = list(figs_dir.glob("overlay_umap_CD8*.png"))
    assert len(saved_files) == 2, f"Expected 2 distinct files, got {len(saved_files)}"


# --- Declared panels: one gene list per obs column ---
#
# `embeddings.overlay.programs` and `config.markers.panels` both score with
# score_genes and both write obs[name], so a manifest declaring the same panel in
# each leaves the object carrying a score whose gene list is not recoverable from
# it — whichever block ran last won, silently.


def _panel_ctx(panels, tmp=None):
    """A context whose config declares marker panels (and optionally paths)."""
    ctx = _CtxPaths(tmp) if tmp is not None else _Ctx()
    ctx.config = SimpleNamespace(markers=SimpleNamespace(panels=panels))
    return ctx


def test_an_overlay_with_no_programs_inherits_the_declared_panels():
    """Declare-once: the panels are already in the manifest, so restating them is the bug."""
    panels = {"capillary": ["LYVE1", "CCL21"], "collecting": ["FOXC2", "GATA2"]}

    resolved, warnings = _overlay_with_declared_panels(OverlayConfig(), _panel_ctx(panels))

    assert resolved.programs == panels
    assert any("config.markers.panels" in w for w in warnings)


def test_a_panel_declared_twice_with_different_genes_is_reported():
    """The two lists compete for one obs column; say which one won instead of guessing."""
    overlay_cfg = OverlayConfig(programs={"capillary": ["LYVE1", "CCL21", "MRC1"]})
    ctx = _panel_ctx({"capillary": ["LYVE1", "CCL21"], "collecting": ["FOXC2"]})

    resolved, warnings = _overlay_with_declared_panels(overlay_cfg, ctx)

    # The overlay's list is what score_genes uses, so that is the list obs holds.
    assert resolved.programs == {"capillary": ["LYVE1", "CCL21", "MRC1"]}
    assert len(warnings) == 1
    assert "capillary" in warnings[0]
    # A panel declared only in markers.panels is not a conflict, so it is not named.
    assert "collecting" not in warnings[0]


def test_a_panel_declared_twice_identically_is_silent():
    """Duplication that cannot change a score is not worth a warning."""
    overlay_cfg = OverlayConfig(programs={"capillary": ["LYVE1", "CCL21"]})
    ctx = _panel_ctx({"capillary": ["LYVE1", "CCL21"]})

    resolved, warnings = _overlay_with_declared_panels(overlay_cfg, ctx)

    assert resolved.programs == overlay_cfg.programs
    assert warnings == []


def test_a_context_without_declared_panels_leaves_the_overlay_alone():
    """Most callers have no markers block at all; the fallback must not require one."""
    overlay_cfg = OverlayConfig(programs={"capillary": ["LYVE1"]})

    for ctx in (_Ctx(), _panel_ctx({})):
        resolved, warnings = _overlay_with_declared_panels(overlay_cfg, ctx)
        assert resolved is overlay_cfg
        assert warnings == []


def _adata_scoreable():
    """A rendered object with enough genes for score_genes' control bins."""
    rng = np.random.default_rng(1)
    a = ad.AnnData(X=rng.random((50, 60)).astype("float32"))
    a.obsm["X_pca_harmony"] = rng.normal(size=(50, 6)).astype("float32")
    a.obs["cell_type"] = ["A"] * 25 + ["B"] * 25
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    a.var_names = [f"GENE_{i}" for i in range(60)]
    sc.pp.neighbors(a, use_rep="X_pca_harmony", random_state=0)
    sc.tl.umap(a, random_state=0)
    return a


def test_overlay_method_scores_the_declared_panels_when_the_block_is_empty(tmp_path):
    """End to end: a manifest that declares panels once still gets the overlays."""
    a = _adata_scoreable()
    cfg = _cfg(
        embeddings=["umap"],
        figure_formats=["png"],
        dpi=80,
        overlay=OverlayConfig(),
        magic=MagicConfig(enabled=False),
    )
    ctx = _panel_ctx({"capillary": ["GENE_0", "GENE_1"]}, tmp=tmp_path)

    res = ContinuousOverlayMethod().run(a, cfg, ctx)

    assert not isinstance(res, MethodSkip)
    assert res.metrics["n_figures"] == 1
    # The score is on the object under the declared panel's name.
    assert "capillary" in res.adata.obs.columns
