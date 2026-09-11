"""Configuration for the integration (batch-correction) stage."""

from __future__ import annotations

from pydantic import Field

from cellquorum.backends.harmonypy_backend import DEFAULT_MAX_ITER_HARMONY
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
    n_latent: int = Field(default=30, ge=1)

    # Whether scVI trains on highly variable genes only. None (the default) follows the
    # feature-selection stage: its flag if present, all genes if not. `true` requires the
    # flag and fails without it rather than training on everything; `false` forces all
    # genes. Declared here because scVI read this key while the schema forbade it, so the
    # override was unreachable from YAML and the value was always the default.
    use_highly_variable: bool | None = None

    # scVI max training epochs (None => scvi-tools default / early stop).
    max_epochs: int | None = Field(default=None, gt=0)

    # Random seed for deterministic integration.
    random_state: int = 0

    # Harmony iteration cap. Exposed because a Harmony that hits the cap returns a PARTIALLY
    # corrected embedding, which the stage reports as a warning rather than leaving to an INFO
    # log line nobody sees.
    #
    # The default is imported, not repeated. It was written as a literal `10` here AND as
    # `DEFAULT_MAX_ITER_HARMONY = 10` in the backend, so raising one would have left the other
    # silently governing every config that does not name the field — the same two-places-one-
    # decision problem that made `stages.feature_selection: true` skip its own stage.
    max_iter_harmony: int = Field(default=DEFAULT_MAX_ITER_HARMONY, ge=1)

    # Multi-method dispatch: list of per-method sub-configs (each entry is a full
    # method config with its own `method`, `output_rep`, etc.). An empty list (the
    # default) means use the scalar `method:` path; only a non-empty list triggers
    # multi-method dispatch, running each entry in order against the same AnnData.
    methods: list[dict] = []


__all__ = ["IntegrationConfig"]
