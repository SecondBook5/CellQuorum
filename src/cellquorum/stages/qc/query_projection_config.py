"""Configuration for the query-projection stage (order 105)."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class QueryProjectionConfig(StrictBaseModel):
    """Settings for projecting borderline cells onto the frozen core manifold.

    No ``enabled`` field, deliberately: whether the stage runs is declared once, in
    ``stages.query_projection``. A second switch here is the double-gate bug that ran a
    stage in the plan and skipped it at execution.
    """

    # obsm key of the frozen representation. When unset, the stage tries X_scvi (fit on
    # core cells with borderline projected through), then X_pca_harmony, then X_pca.
    use_rep: str | None = None

    # obs column supplying the reference labels the neighbourhood votes on. When unset,
    # tries cell_type, then qc_provisional_lineage, then leiden.
    label_column: str | None = None

    # Neighbours per query cell; clamped down when the core reference is smaller.
    k: int = 15


__all__ = ["QueryProjectionConfig"]
