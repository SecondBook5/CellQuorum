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
        curvature_top_k_targets: Targets kept per TF (by importance) for the
            curvature network figure.
        curvature_ricci_alpha: Ollivier-Ricci alpha for the curvature figure.
        curvature_plot_top_k: Strongest edges actually drawn in the figure.
        cluster_labels: optional cluster id -> display name map for the
            RSS/clustermap/cell-clustermap figures, display only (the
            underlying group_by column is never renamed in place).
        curvature_radial_n_tf: Top TFs shown in the radial curvature figure.
        curvature_radial_per_tf: Targets kept per TF in the radial curvature
            figure.
        curvature_radial_n_sectors: Sectors in the radial curvature figure.
        radial_enabled: Whether the radial DoRothEA/CollecTRI GRN figure
            renders (independent of pyscenic's cisTarget requirement).
        radial_collection: decoupler net collection for the radial figure's
            TF activity.
        radial_de_results_filename: DE results CSV read for the radial
            figure (relative to results/).
        radial_min_size: Minimum overlapping targets to score a TF (ULM
            tmin) in the radial figure.
        radial_n_tf: Top TFs shown in the radial figure.
        radial_per_tf: Targets kept per TF in the radial figure.
        radial_n_sectors: Sectors in the radial figure.
        radial_case_color: Override for the radial figure's case color.
        radial_control_color: Override for the radial figure's control color.
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
    # A list is accepted, most-preferred first, and the first name that EXISTS is used.
    # Environment names are a fact about a machine, not about an analysis: the container
    # built by docker/Dockerfile creates "pyscenic_env", while a workstation may already have
    # the same software under a different name. Pinning the config to one of them made the
    # other silently report the backend missing and skip the stage.
    env_name: str | list[str] = "pyscenic_env"

    # Environment launcher (micromamba).
    launcher: str = "micromamba"

    # pySCENIC execution timeout (seconds).
    timeout_seconds: int = 7200

    # Targets kept per TF (by importance) for the curvature network figure.
    curvature_top_k_targets: int = 15

    # Ollivier-Ricci alpha for the regulon curvature figure.
    curvature_ricci_alpha: float = 0.5

    # Strongest edges actually drawn in the curvature figure (the built graph
    # can have thousands; this keeps the plot legible).
    curvature_plot_top_k: int = 30

    # Number of top TFs (by |mean node curvature|) shown as inner-ring
    # regulators in the radial curvature figure. Separate from
    # curvature_plot_top_k (a global edge cap for the force-directed
    # hairball) because the radial layout selects TF-first, per_tf-capped.
    curvature_radial_n_tf: int = 12

    # Targets kept per TF (by curvature-sorted weight) in the radial
    # curvature figure.
    curvature_radial_per_tf: int = 6

    # Sectors the radial curvature figure's TFs are grouped into.
    curvature_radial_n_sectors: int = 4

    # Optional cluster id -> display name map for RSS/clustermap figures.
    cluster_labels: dict[str, str] = {}

    # --- Radial DoRothEA/CollecTRI GRN figure ("the cool circle plot") ---
    # A different, complementary figure from the pySCENIC ones above: TF
    # activity from decoupler ULM on the LE-vs-Normal DE CONTRAST (not
    # per-cell AUCell), against a curated prior net (not the data-driven
    # pySCENIC regulon graph). Skips cleanly (warning, not error) when the
    # DE stage's CSV or the prior net is unavailable.

    # Whether the radial figure renders (independent of pyscenic's own
    # cisTarget-DB requirement -- this path never touches cisTarget).
    radial_enabled: bool = True

    # decoupler net collection ("dorothea", "collectri", ...).
    radial_collection: str = "dorothea"

    # DE results CSV to read (relative to results/), matching the
    # enrichment stage's own de_results_filename convention.
    radial_de_results_filename: str = "de_pseudobulk_edger.csv"

    # Minimum overlapping targets required to score a TF (decoupler tmin).
    radial_min_size: int = 5

    # Top TFs shown as inner-ring regulators.
    radial_n_tf: int = 12

    # Targets kept per TF, ranked by |logFC| with a significance bonus.
    radial_per_tf: int = 6

    # Sectors TFs are grouped into (by shared-target Jaccard clustering).
    radial_n_sectors: int = 4

    # Node/edge colors for "up in case" / "up in control" (None -> the
    # house defaults, figstyle.LE_RED / figstyle.NORMAL_BLUE, same
    # fallback convention as de_viz's case_color/control_color).
    radial_case_color: str | None = None
    radial_control_color: str | None = None


__all__ = ["GrnConfig"]
