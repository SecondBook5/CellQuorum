"""Integration: integration + annotation stages register and order correctly."""

from __future__ import annotations


def test_annotation_does_not_skip_when_clustering_runs_first() -> None:
    """
    Verify that annotation actually runs when clustering precedes it.

    This is the critical test for bug C1: when clustering produces leiden labels
    before annotation runs, annotation should NOT skip due to missing requirements.
    """

    import anndata as ad
    import numpy as np
    import pandas as pd

    from cellquorum.core.contracts import set_layer_tag
    from cellquorum.methods.registry import MethodRegistry
    from cellquorum.stages.annotation.marker_vote import MarkerVoteMethod
    from cellquorum.stages.annotation.stage import AnnotationStage
    from cellquorum.stages.clustering.neighbors_leiden import LeidenMethod
    from cellquorum.stages.clustering.stage import ClusteringStage

    # Build a minimal AnnData with an embedding for clustering.
    rng = np.random.default_rng(42)
    n_cells = 100
    n_genes = 10
    x = rng.random((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=x, var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]))
    adata.layers["cellquorum_normalized"] = x.copy()
    set_layer_tag(
        adata, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1"
    )

    # Add a fake PCA embedding for clustering to use.
    adata.obsm["X_pca"] = rng.random((n_cells, 5)).astype(np.float32)

    # Build a minimal context with configs for both stages.
    class _Ctx:
        def __init__(self, adata, config):
            self._adata = adata
            self.config = config

        def require_adata(self):
            return self._adata

    ctx = _Ctx(
        adata,
        {
            "clustering": {
                "method": "leiden",
                "n_neighbors": 15,
                "resolution": 1.0,
                "random_state": 0,
                "key_added": "leiden",
                "use_rep": "X_pca",
            },
            "annotation": {
                "method": "marker_vote",
                "cluster_key": "leiden",
                "score_layer": "cellquorum_normalized",
                "key_added": "cell_type",
                "marker_panels": {"TypeA": ["gene_0", "gene_1"], "TypeB": ["gene_5", "gene_6"]},
                "random_state": 0,
            },
            "stages": {"clustering": True, "annotation": True},
        },
    )

    # Run clustering first to produce leiden labels.
    clustering_reg = MethodRegistry()
    clustering_reg.register(LeidenMethod)
    clustering_stage = ClusteringStage(registry=clustering_reg)
    clustering_result = clustering_stage.run(ctx)

    # Verify clustering succeeded and leiden exists.
    assert not clustering_result.metrics.get("skipped")
    assert "leiden" in clustering_result.adata.obs.columns

    # Run annotation next — it should NOT skip because leiden now exists.
    annotation_reg = MethodRegistry()
    annotation_reg.register(MarkerVoteMethod)
    annotation_stage = AnnotationStage(registry=annotation_reg)
    annotation_result = annotation_stage.run(ctx)

    # Verify annotation succeeded and cell_type exists.
    assert not annotation_result.metrics.get("skipped")
    assert "cell_type" in annotation_result.adata.obs.columns


def test_clustering_auto_couples_use_rep_when_integration_enabled() -> None:
    """
    Verify that clustering auto-sets use_rep to the integration output_rep.

    This is the critical test for bug I1: when integration is enabled and the user
    did NOT explicitly set clustering.use_rep, clustering should read X_pca_harmony
    instead of discarding the corrected embedding and reading X_pca.
    """

    import anndata as ad
    import numpy as np

    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.methods.registry import MethodRegistry
    from cellquorum.stages.clustering.neighbors_leiden import LeidenMethod
    from cellquorum.stages.clustering.stage import ClusteringStage

    # Build a minimal AnnData with both X_pca and X_pca_harmony.
    rng = np.random.default_rng(42)
    n_cells = 100
    adata = ad.AnnData(X=rng.random((n_cells, 10)).astype(np.float32))
    adata.obsm["X_pca"] = rng.random((n_cells, 5)).astype(np.float32)
    adata.obsm["X_pca_harmony"] = rng.random((n_cells, 5)).astype(np.float32)

    # Build a context with integration enabled and use_rep NOT explicitly set.
    class _Ctx:
        def __init__(self, adata, config):
            self._adata = adata
            self.config = config

        def require_adata(self):
            return self._adata

    # Use a pydantic config to enable the model_fields_set detection logic.
    config = CellQuorumConfig()
    config.stages.integration = True
    config.stages.clustering = True
    config.integration.output_rep = "X_pca_harmony"
    # Do NOT set clustering.use_rep explicitly — leave it at default "X_pca"

    ctx = _Ctx(adata, config)

    # Run clustering and capture the result.
    reg = MethodRegistry()
    reg.register(LeidenMethod)
    stage = ClusteringStage(registry=reg)
    result = stage.run(ctx)

    # Verify clustering succeeded and used the corrected embedding.
    assert not result.metrics.get("skipped")
    # The auto-coupling logic should emit a note.
    note_found = any("auto-set to X_pca_harmony" in note for note in result.notes)
    assert note_found, f"Expected auto-coupling note in {result.notes}"
