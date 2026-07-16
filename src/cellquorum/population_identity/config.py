"""Configuration for the population-identity evidence stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class PopulationIdentityConfig(StrictBaseModel):
    """Settings for generic population/state identity evidence outputs."""

    # Whether the population-identity stage may run.
    enabled: bool = True

    # Explicit candidate population column. When None, CellQuorum resolves one
    # from reference labels, annotation labels, then clusters.
    candidate_key: str | None = None

    # Candidate reference/atlas label columns, in priority order.
    reference_keys: list[str] = ["ref_state", "reference_label", "atlas_label"]

    # Candidate annotation label columns, in priority order.
    annotation_keys: list[str] = ["cell_type", "celltypist_cell_type", "predicted_cell_type"]

    # Fallback cluster column.
    cluster_key: str = "leiden"

    # Optional explicit sample/donor/condition columns. Defaults use design config.
    sample_key: str | None = None
    donor_key: str | None = None
    condition_key: str | None = None

    # Candidate confidence/uncertainty columns, in priority order.
    confidence_keys: list[str] = [
        "annotation_confidence",
        "cell_type_conf",
        "celltypist_confidence",
        "ref_state_consensus_frac",
        "ref_state_knn_agreement",
    ]
    entropy_keys: list[str] = ["scdiag_entropy", "ref_state_knn_entropy"]

    # Candidate 2D embeddings for publication plots.
    embedding_keys: list[str] = ["X_umap", "X_pca_harmony", "X_pca"]

    # Output directory under results/.
    output_dir: str = "population_identity"

    # Whether publication-style figures should be written when possible.
    write_figures: bool = True

    # Evidence thresholds.
    min_cells: int = 20
    min_samples: int = 2
    min_donors: int = 2
    max_dominant_donor_fraction: float = 0.80
    max_qc_fail_fraction: float = 0.30
    max_doublet_fraction: float = 0.20
    min_confidence: float = 0.50
    max_entropy: float | None = None


__all__ = ["PopulationIdentityConfig"]
