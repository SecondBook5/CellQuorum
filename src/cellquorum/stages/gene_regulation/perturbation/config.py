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
        cluster_labels: Optional {raw cluster value -> display name} for figure
            labels; absent -> raw values are shown as-is.
        embedding_key: Shift-vector embedding (falls back X_umap).
        rep_key: PCA/kNN representation (falls back X_pca -> X_pca_harmony).
        condition_key: Disease/healthy obs column; absent -> direction-agnostic.
        healthy_label: Target condition value; absent -> direction-agnostic.
        tf_list: TFs to knock out; None -> systematic screen of all fitted TFs.
        n_top_genes: Highly-variable-gene cap applied before CellOracle ever
            sees the object. CellOracle's own perform_PCA() runs a dense SVD
            over whatever gene panel it is handed; on a real ~33k-gene, ~70k
            -cell object that is over LAPACK's 32-bit dense-indexing ceiling
            ("Indexing a matrix... would incur an in integer overflow in
            LAPACK") -- confirmed live. CellOracle's own tutorials restrict
            to a few thousand HVGs before building the Oracle object for
            exactly this reason. A panel already at or under this cap is
            left untouched.
        n_top_targets: Ranked-table / figure cutoff.
        knn_n_neighbors: Neighbors for CellOracle kNN imputation.
        max_cells: Physically subsample to at most this many cells (proportionally
            stratified by cluster_key, and condition_key when set) before the
            Oracle is built. 0 disables subsampling. estimate_transition_prob's
            call into velocyto's compiled correlation routine
            (colDeltaCorpartial) segfaulted deterministically on a real
            61,608-cell object -- confirmed, by reading velocyto's own source,
            to be an unbounded array write, and confirmed fixed by shrinking the
            object itself (a real ~2,000-cell run completed cleanly with the
            same script; sampled_fraction/knn_n_neighbors alone did not fix it
            even matched exactly to that object's values, since the crashing
            buffer is sized off the actual cell count, not a sampling ratio).
        sampled_fraction: estimate_transition_prob's random-neighbor candidate
            pool as a fraction of cells (CellOracle default 1.0). Independent of
            max_cells; use max_cells for the crash above.
        filter_p: filter_links' significance cutoff (CellOracle default 0.001).
            Usually the actual binding constraint on which TFs get simulated --
            not filter_edge_number's count cap. A TF present in the base GRN can
            still get zero fitted edges at a given cluster's cell count if none
            clear this p-value, regardless of how high the count cap is set.
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

    # {raw cluster_key value -> display name} for figure labels (e.g. the
    # shift-field/shift-grid embedding plots). Purely cosmetic and optional;
    # a cluster with no entry falls back to its raw value. Study-specific, so
    # it lives in config, never hardcoded in the figure-drawing code.
    cluster_labels: dict[str, str] | None = None

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

    # Highly-variable-gene cap applied before CellOracle sees the object.
    n_top_genes: int = 2000

    # Ranked-table / figure cutoff.
    n_top_targets: int = 20

    # Neighbors for CellOracle kNN imputation.
    knn_n_neighbors: int = 200

    # Physically subsample to at most this many cells before the Oracle is
    # built. 0 disables subsampling. See the class docstring for why this
    # exists -- it is the real fix for a real segfault, not a knob to leave at
    # its default on a large object.
    max_cells: int = 0

    # estimate_transition_prob's random-neighbor candidate pool as a fraction
    # of cells (CellOracle default 1.0).
    sampled_fraction: float = 1.0

    # filter_links' significance cutoff (CellOracle default 0.001).
    filter_p: float = 0.001

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
