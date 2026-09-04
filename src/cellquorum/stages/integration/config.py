"""Configuration for the integration (batch-correction) stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class IntegrationConfig(StrictBaseModel):
    """Batch-integration settings."""

    # Whether the integration stage may run.
    enabled: bool = True

    # Integration method registry key (harmony | scvi | scanvi).
    method: str = "harmony"

    # obs column identifying the batch to correct over (donor by default).
    batch_key: str = "patient_id"

    # obs column with cell-type labels for semi-supervised scANVI integration.
    # Cells without a label should use `unlabeled_category`; only used by scanvi.
    label_key: str | None = None

    # Label value marking unlabeled cells for scANVI.
    unlabeled_category: str = "Unknown"

    # Input embedding to correct (Harmony) / basis for method.
    input_rep: str = "X_pca"

    # obsm key where the corrected embedding is written (scanpy convention).
    output_rep: str = "X_pca_harmony"

    # scVI latent dimensionality.
    n_latent: int = 30

    # scVI max training epochs (None => scvi-tools default / early stop).
    max_epochs: int | None = None

    # Random seed for deterministic integration.
    random_state: int = 0

    # Harmony iteration cap (harmonypy's own default is 10). Exposed because 10 is
    # not enough on every dataset, and a Harmony that hits the cap returns a
    # PARTIALLY corrected embedding — which the stage now reports as a warning
    # instead of leaving it to an INFO log line nobody sees.
    max_iter_harmony: int = 10

    # Multi-method dispatch: list of per-method sub-configs (each entry is a full
    # method config with its own `method`, `output_rep`, etc.). An empty list (the
    # default) means use the scalar `method:` path; only a non-empty list triggers
    # multi-method dispatch, running each entry in order against the same AnnData.
    methods: list[dict] = []


__all__ = ["IntegrationConfig"]
