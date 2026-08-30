"""Configuration for the annotation_diagnostics stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class AnnotationDiagnosticsConfig(StrictBaseModel):
    """Annotation-confidence diagnostics via scDiagnostics (R).

    Opt-in stage (off by default). Measures annotation confidence via
    anomaly detection, kNN probabilities, and categorization entropy.
    READ-ONLY: adds diagnostic obs columns but never reassigns cell_type.

    Attributes:
        enabled: Whether the stage runs (opt-in; off by default).
        method: Diagnostic method registry key (scdiagnostics).
        backend: Backend to use (rscript).
        r_package: R package name for status checks (scDiagnostics).
        cell_type_col: obs column containing cell type annotations.
        expression_layer: Log-normalized layer to pass to scDiagnostics.
        reference_h5ad: Optional reference h5ad for query-vs-reference metrics.
        soft_scores_obsm: Optional obsm key for soft probabilities (entropy).
        pc_subset: PC indices to use (1-indexed per R convention).
        n_tree: Number of trees for isolation forest anomaly detection.
        n_neighbor: Number of neighbors for kNN probability calculation.
        timeout_seconds: R script timeout in seconds.
    """

    # Whether the annotation-diagnostics stage runs (opt-in; off by default).
    enabled: bool = False

    # Diagnostic method registry key (scdiagnostics).
    method: str = "scdiagnostics"

    # Backend to use (rscript for R-based scDiagnostics).
    backend: str = "rscript"

    # R package name for backend status checks.
    r_package: str = "scDiagnostics"

    # obs column containing cell type annotations.
    cell_type_col: str = "cell_type"

    # Log-normalized expression layer used to build the scDiagnostics query.
    expression_layer: str = "lognorm"

    # Optional reference h5ad for query-vs-reference diagnostics.
    reference_h5ad: str | None = None

    # Optional obsm key for soft probability matrix (for entropy calculation).
    soft_scores_obsm: str | None = None

    # PC indices to use (1-indexed per R convention).
    pc_subset: list[int] = [1, 2, 3, 4, 5]

    # Number of trees for isolation forest anomaly detection.
    n_tree: int = 500

    # Number of neighbors for kNN probability calculation.
    n_neighbor: int = 15

    # R script timeout in seconds.
    timeout_seconds: int = 1800


__all__ = ["AnnotationDiagnosticsConfig"]
