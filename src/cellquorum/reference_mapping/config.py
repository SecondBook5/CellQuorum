"""Configuration for the reference_mapping stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class ReferenceMappingConfig(StrictBaseModel):
    """Reference mapping settings for atlas-based label transfer."""

    # Whether the reference_mapping stage may run.
    enabled: bool = False

    # Reference mapping method registry key (scarches).
    method: str = "scarches"

    # Path to the reference atlas h5ad file.
    atlas_h5ad: str | None = None

    # obs column in the atlas holding cell-type labels.
    label_key: str = "cell_type"

    # obs column in the atlas holding batch labels.
    atlas_batch_key: str = "batch"

    # Batch value assigned to the query dataset.
    query_batch_value: str = "query"

    # Layer holding raw counts for scVI/scANVI.
    counts_layer: str = "counts"

    # Filters applied to the reference atlas (list of column/keep dicts).
    reference_filters: list[dict] = []

    # Optional compartment filter (dict with column/keep keys).
    compartment_filter: dict | None = None

    # Minimum label probability for high-confidence predictions.
    min_label_prob: float | None = None

    # obs column name to store label probabilities.
    label_prob_col: str | None = None

    # Number of highly-variable genes for scVI training.
    n_top_genes: int = 3000

    # HVG flavor (seurat_v3 | seurat | cell_ranger).
    hvg_flavor: str = "seurat_v3"

    # Genes forced into the HVG set (markers, biology-driven).
    force_genes: list[str] = []

    # scVI latent space dimensionality.
    n_latent: int = 30

    # Number of hidden layers in the scVI encoder.
    n_layers: int = 2

    # Dropout rate in scVI layers.
    dropout_rate: float = 0.2

    # Gene likelihood (zinb | nb | poisson).
    gene_likelihood: str = "zinb"

    # Max epochs for scVI training on the reference atlas.
    max_epochs_scvi: int = 400

    # Max epochs for scANVI training on the reference atlas.
    max_epochs_scanvi: int = 20

    # Max epochs for query surgery (arches).
    max_epochs_query: int = 100

    # Whether early stopping is enabled.
    early_stopping: bool = True

    # Early stopping patience for query surgery.
    query_early_stopping_patience: int = 10

    # Early stopping monitor metric for query surgery.
    query_early_stopping_monitor: str = "reconstruction_loss_train"

    # Label for unlabeled cells in scANVI.
    unlabeled_category: str = "Unknown"

    # Random seeds for multi-seed ensemble.
    seeds: list[int] = [0, 1, 2, 3, 4]

    # k for kNN uncertainty estimation.
    knn_k: int = 30

    # obs column that receives the transferred label.
    key_added: str = "ref_state"

    # Compute backend preference (auto | cpu | gpu).
    compute_backend: str = "auto"

    # Whether to write loss curves to artifacts.
    write_loss_curves: bool = True

    # Whether completed per-seed ScArches checkpoints should be reused on rerun.
    resume: bool = True


__all__ = ["ReferenceMappingConfig"]
