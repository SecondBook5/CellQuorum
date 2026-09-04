import pytest
from pydantic import ValidationError

from cellquorum.config.models import CellQuorumConfig, StageSelectionConfig
from cellquorum.stages.integration.embeddings.config import (
    EmbeddingsConfig,
    MagicConfig,
    OverlayConfig,
)


def test_embeddings_config_defaults():
    cfg = EmbeddingsConfig()
    assert cfg.enabled is True
    assert cfg.use_rep == "X_pca_harmony"
    assert cfg.umap_min_dist == 0.3
    assert cfg.phate_knn == 15
    assert cfg.phate_decay == 40
    assert cfg.paga_groupby is None
    assert cfg.paga_threshold == 0.2
    assert cfg.random_state == 0
    assert cfg.embeddings == ["umap", "phate"]
    assert cfg.figure_formats == ["pdf", "png"]
    assert cfg.dpi == 300
    assert isinstance(cfg.overlay, OverlayConfig)
    assert isinstance(cfg.magic, MagicConfig)


def test_overlay_and_magic_defaults():
    ov = OverlayConfig()
    assert ov.genes == [] and ov.programs == {} and ov.obs_columns == []
    assert ov.cell_cycle is False and ov.s_genes == [] and ov.g2m_genes == []
    mg = MagicConfig()
    assert (
        mg.enabled is False and mg.knn == 15 and mg.solver == "approximate" and mg.random_state == 0
    )


def test_embeddings_config_rejects_unknown_field():
    with pytest.raises(ValidationError):
        EmbeddingsConfig(not_a_field=1)


def test_stage_selection_has_embeddings_toggle():
    assert StageSelectionConfig().embeddings is True


def test_cellquorum_config_has_embeddings_subconfig():
    cfg = CellQuorumConfig(project={"name": "t"}, input={"h5ad": "x.h5ad"})
    assert isinstance(cfg.embeddings, EmbeddingsConfig)
