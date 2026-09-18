"""Configuration for the state-scoring stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class StateScoringConfig(StrictBaseModel):
    """Per-cell cell-state program scoring.

    Scores curated and user-supplied gene programs (stress/HSP, hypoxia/HIF,
    type-I interferon, senescence/SASP, fibrosis/ECM, plus anything the user
    adds) against a log-normalized layer. ``score_genes`` writes a per-cell
    ``obs`` column per program; ``aucell`` writes a per-cell ``obsm`` matrix.
    All biological specifics come from config — curated programs are a
    convenience default, not a study assumption, and are gated by the present
    gene-count so an organism/panel mismatch skips a program instead of scoring
    noise.

    Attributes:
        enabled: Whether the stage runs.
        methods: State-scoring method registry keys (empty → default list injected).
        layer: Log-normalized expression layer the methods read.
        cell_type_col: obs column with cell-type labels (per-type summary tables).
        use_builtin_programs: Whether to include the curated ``STATE_PROGRAMS``.
        builtin_programs: Subset of curated program names to use (empty → all).
        programs: User programs as ``name -> [gene, ...]`` (override/augment curated).
        marker_panels: Names of ``config.markers`` panels to also score as programs.
        gmt_path: Optional ``.gmt`` whose gene sets are scored as programs.
        min_program_genes: Minimum present genes for a program to be eligible.
        random_state: Seed for scanpy ``score_genes`` control-gene sampling.
        key_prefix: Prefix for the per-program obs columns (``score_genes``).
    """

    enabled: bool = True
    methods: list[dict] = []
    layer: str = "cellquorum_normalized"
    cell_type_col: str = "cell_type"
    use_builtin_programs: bool = True
    builtin_programs: list[str] = []
    programs: dict[str, list[str]] = {}
    marker_panels: list[str] = []
    gmt_path: str | None = None
    min_program_genes: int = 3
    random_state: int = 0
    key_prefix: str = "state_"


__all__ = ["StateScoringConfig"]
