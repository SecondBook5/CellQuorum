"""Configuration for the GRN (pySCENIC) stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class GrnConfig(StrictBaseModel):
    """Classic pySCENIC gene-regulatory network inference via an isolated env.

    Infers a directed TF->target network (GRNBoost2 -> cisTarget -> AUCell) in an
    isolated micromamba environment and renders publication-grade regulon figures.
    Skips cleanly when the env or the cisTarget databases are unavailable.

    Attributes:
        enabled: Whether the stage runs (enabled by default).
        method: GRN method registry key (pyscenic).
        layer: Layer holding raw counts for pySCENIC.
        group_by: Optional grouping variable for RSS/figures (falls back to
            cell_type -> leiden -> "all").
        organism: Organism label (documentation/hint; the cisTarget DBs are the
            real organism driver).
        tfs_path: Path to the transcription-factor list (allTFs_*.txt).
        motifs_path: Path to the cisTarget motif annotation table (motifs-*.tbl).
        rankings_glob: cisTarget ranking DB(s): a file, a space-joined list, or a
            glob (*.genes_vs_motifs.rankings.feather).
        num_workers: Worker processes for GRNBoost2 / cisTarget / AUCell. ``None``
            inherits ``compute.n_jobs``, which is the point: GRNBoost2 is the
            longest single step in the engine, and a hard-coded default meant a
            run configured for 2 workers still spawned 8 dask workers — each
            holding its own partition of the expression matrix.
        max_cells: Deterministic downsample cap before loom export.
        min_cells_total: Minimum total cells required to attempt inference.
        top_n: Top-RSS regulons per group selected for figures.
        seed: Random seed for reproducibility.
        env_name: Name of the isolated micromamba environment.
        launcher: Environment launcher (micromamba).
        timeout_seconds: pySCENIC execution timeout in seconds.
    """

    # Whether this stage runs.
    enabled: bool = True

    # Selected GRN method (registry key under stage_category 'grn').
    method: str = "pyscenic"

    # Layer holding raw counts for pySCENIC.
    layer: str = "counts"

    # Optional grouping variable for RSS/figures.
    group_by: str | None = None

    # Organism label (documentation/hint only).
    organism: str = "human"

    # Path to the transcription-factor list.
    tfs_path: str | None = None

    # Path to the cisTarget motif annotation table.
    motifs_path: str | None = None

    # cisTarget ranking DB(s): file, space-joined list, or glob.
    rankings_glob: str | None = None

    # Worker processes for GRNBoost2 / cisTarget / AUCell; None inherits compute.n_jobs.
    num_workers: int | None = None

    # Deterministic downsample cap before loom export.
    max_cells: int = 20000

    # Minimum total cells required to attempt inference.
    min_cells_total: int = 200

    # Top-RSS regulons per group selected for figures.
    top_n: int = 5

    # Random seed for reproducibility.
    seed: int = 0

    # Name of the isolated micromamba environment.
    env_name: str = "pyscenic_env"

    # Environment launcher (micromamba).
    launcher: str = "micromamba"

    # pySCENIC execution timeout (seconds).
    timeout_seconds: int = 7200


__all__ = ["GrnConfig"]
