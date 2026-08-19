"""Configuration for the perturbation (in-silico KO) stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class PerturbationConfig(StrictBaseModel):
    """In-silico transcription-factor knockout via CellOracle in an isolated env.

    Infers a simulation-ready GRN from observational counts + a built-in promoter
    base GRN, simulates each TF knockout by zeroing the TF and propagating the
    signal, and ranks knockouts by how strongly they shift disease cells toward the
    healthy state. Skips cleanly when the env or base GRN is unavailable.

    Attributes:
        enabled: Whether the stage runs (enabled by default).
        method: Perturbation method registry key (celloracle).
        layer: Layer holding raw counts for CellOracle.
        organism: Built-in base GRN organism (human/mouse).
        cluster_key: GRN cluster grouping (falls back cell_type -> leiden -> "all").
        embedding_key: Shift-vector embedding (falls back X_umap).
        rep_key: PCA/kNN representation (falls back X_pca -> X_pca_harmony).
        condition_key: Disease/healthy obs column; absent -> direction-agnostic.
        healthy_label: Target condition value; absent -> direction-agnostic.
        tf_list: TFs to knock out; None -> systematic screen of all fitted TFs.
        n_top_targets: Ranked-table / figure cutoff.
        knn_n_neighbors: Neighbors for CellOracle kNN imputation.
        n_propagation: Signal-propagation iterations for the KO simulation.
        min_cells_total: Minimum total cells required to attempt inference.
        seed: Random seed for reproducibility.
        env_name: Name of the isolated micromamba environment.
        launcher: Environment launcher (micromamba).
        timeout_seconds: CellOracle execution timeout in seconds.
    """

    # Whether this stage runs.
    enabled: bool = True

    # Selected perturbation method (registry key under stage_category 'perturbation').
    method: str = "celloracle"

    # Layer holding raw counts for CellOracle.
    layer: str = "counts"

    # Built-in base GRN organism (human/mouse).
    organism: str = "human"

    # GRN cluster grouping (falls back cell_type -> leiden -> "all").
    cluster_key: str | None = None

    # Shift-vector embedding (falls back X_umap).
    embedding_key: str | None = None

    # PCA/kNN representation (falls back X_pca -> X_pca_harmony).
    rep_key: str | None = None

    # Disease/healthy obs column; absent -> direction-agnostic.
    condition_key: str | None = None

    # Target condition value; absent -> direction-agnostic.
    healthy_label: str | None = None

    # TFs to knock out; None -> systematic screen of all fitted TFs.
    tf_list: list[str] | None = None

    # Ranked-table / figure cutoff.
    n_top_targets: int = 20

    # Neighbors for CellOracle kNN imputation.
    knn_n_neighbors: int = 200

    # Signal-propagation iterations for the KO simulation.
    n_propagation: int = 3

    # Minimum total cells required to attempt inference.
    min_cells_total: int = 200

    # Random seed for reproducibility.
    seed: int = 0

    # Name of the isolated micromamba environment.
    env_name: str = "celloracle_env"

    # Environment launcher (micromamba).
    launcher: str = "micromamba"

    # CellOracle execution timeout (seconds).
    timeout_seconds: int = 10800


__all__ = ["PerturbationConfig"]
