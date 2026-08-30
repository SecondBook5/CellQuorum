"""Configuration for the multicellular programs (DIALOGUE) stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class MulticellularProgramsConfig(StrictBaseModel):
    """DIALOGUE-based inference of cross-cell-type coordinated programs.

    DIALOGUE identifies gene programs that coordinate across multiple cell types
    within a multicellular niche, capturing emergent multicellular states beyond
    single-cell-type programs. Requires paired samples (donor/condition design).

    All biological specifics (cell-type/sample/donor/condition columns, contrasts)
    come from config — no study assumptions in code.

    Attributes:
        enabled: Whether the stage runs.
        method: Analysis method (currently only "dialogue" supported).
        cell_type_col: obs column with cell-type labels (grouping key for DIALOGUE).
        sample_col: obs column identifying a sample (DIALOGUE sample context).
        donor_col: obs column identifying a donor (paired design requirement).
        condition_col: obs column identifying experimental condition.
        case: Case condition label (disease/treatment).
        control: Control condition label (healthy/baseline).
        use_rep: Representation to use (default "X_pca").
        n_pcs: Number of principal components to use from use_rep.
        layer: Expression layer to read (None → .X).
        n_programs: Number of multicellular programs to infer.
        n_program_genes: Maximum genes per program.
        min_cells_per_type: Minimum cells per cell type to include.
        min_cell_types: Minimum cell types required to run DIALOGUE.
        min_samples: Minimum samples required to run DIALOGUE.
        quality_col: Optional obs column for quality filtering.
        confounders: List of confounding variables to regress out.
        stability_resamples: Number of bootstrap resamples for stability.
        donor_support_min: Minimum donors supporting a program.
        seed: Random seed for reproducibility.
        timeout_seconds: Maximum runtime in seconds.
    """

    enabled: bool = True
    method: str = "dialogue"
    cell_type_col: str | None = None
    sample_col: str | None = None
    donor_col: str | None = None
    condition_col: str | None = None
    case: str | None = None
    control: str | None = None
    use_rep: str = "X_pca"
    n_pcs: int = 10
    layer: str | None = None
    n_programs: int = 5
    n_program_genes: int = 200
    min_cells_per_type: int = 20
    min_cell_types: int = 2
    min_samples: int = 4
    quality_col: str | None = None
    confounders: list[str] = []
    stability_resamples: int = 5
    donor_support_min: int = 2
    seed: int = 0
    timeout_seconds: int = 7200


__all__ = ["MulticellularProgramsConfig"]
