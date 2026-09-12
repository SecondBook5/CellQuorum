"""Configuration for the cell-type composition summary stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class CompositionConfig(StrictBaseModel):
    """
    Cell-type composition summary: overall, per-condition, and per-sample tables
    and figures. Purely descriptive — statistical testing of compositional shifts
    is ``differential_abundance``'s job, not this stage's.

    Attributes:
        cell_type_key: obs column holding cell-type labels.
        condition_key: obs column holding the condition/treatment label. None
            resolves from ``cohort.condition_key``, then the design block's
            ``condition_col``. Absent from obs entirely -> the per-condition table
            and figure are skipped, not an error (not every cohort has one).
        donor_key: obs column identifying a donor/patient. Same resolution and
            skip-if-absent behavior as ``condition_key``.
        sample_key: obs column identifying a library/sample. Same resolution and
            skip-if-absent behavior as ``condition_key``.
        atlas_states: QC states to restrict to (see ``qc_state_column``). Empty
            (the default) keeps every cell.
        qc_state_column: obs column of QC states. None auto-detects
            ``qc_state_final`` when present, else ``qc_state_initial``.
        exclude_multiplets: Whether to drop cells flagged by ``multiplet_column``.
        multiplet_column: obs column of per-cell multiplet calls.
        figure_formats: Export formats for the composition figures.
        dpi: Figure export resolution.
        output_prefix: Prefix used for output artifact filenames.
    """

    cell_type_key: str = "cell_type"
    condition_key: str | None = None
    donor_key: str | None = None
    sample_key: str | None = None

    atlas_states: list[str] = []
    qc_state_column: str | None = None
    exclude_multiplets: bool = True
    multiplet_column: str = "qc_probable_multiplet"

    figure_formats: list[str] = ["pdf", "png"]
    dpi: int = 300
    output_prefix: str = "composition"


__all__ = ["CompositionConfig"]
