"""Configuration for the discovery stage (de-novo consensus-NMF programs)."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class DiscoveryConfig(StrictBaseModel):
    """Unbiased, data-driven program discovery by consensus NMF.

    Every other program signal in the engine is curated (marker panels, state
    programs, pathway gene sets). Discovery is the unsupervised counterpart: it
    factorizes the expression matrix into ``n_components`` non-negative gene
    programs with no prior, replicating the factorization ``n_runs`` times and
    taking a consensus so the programs are stable rather than seed-dependent
    (the cNMF idea, run in-process on scikit-learn).

    NMF requires non-negative input; the log-normalized layer's small negative
    values (the shifted-CLR recipe) are clipped to zero before factorization,
    which is recorded in the stage metrics.

    Attributes:
        enabled: Whether the stage runs.
        method: Discovery method registry key.
        layer: Log-normalized expression layer to factorize.
        n_components: Number of programs (NMF rank ``k``) to discover.
        n_runs: Replicate factorizations whose spectra are consensus-clustered.
        use_hvg: Restrict to ``var['highly_variable']`` genes when present.
        n_top_genes: Top genes per program written to the loadings table.
        cell_type_col: obs column for the per-cell-type mean-usage table.
        max_iter: Maximum NMF iterations per fit.
        random_state: Base seed (each replicate offsets it).
        key: obsm key for the per-cell program-usage matrix.
    """

    enabled: bool = True
    method: str = "nmf"
    layer: str = "cellquorum_normalized"
    n_components: int = 10
    n_runs: int = 20
    use_hvg: bool = True
    n_top_genes: int = 50
    cell_type_col: str = "cell_type"
    max_iter: int = 200
    random_state: int = 0
    key: str = "X_cnmf"


__all__ = ["DiscoveryConfig"]
