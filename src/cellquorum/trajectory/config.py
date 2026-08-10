"""Configuration for the trajectory stage (spec #1: loom I/O + RNA velocity)."""

from __future__ import annotations

from pathlib import Path

from cellquorum.config.base import StrictBaseModel


class VelocityGenerationConfig(StrictBaseModel):
    """Config-gated, idempotent velocyto loom generation.

    Generation only runs when ``generate_missing`` is true AND a sample's loom
    is absent AND the BAM+GTF resolve. It is heavy (BAM-level) and never a forced
    per-run step. All paths are structural — no dataset assumptions in code.

    Attributes:
        generate_missing: Master gate; when False, missing looms simply skip.
        bam_dir: Root holding per-sample CellRanger ``outs`` directories.
        gtf_path: ``genes.gtf`` passed to ``velocyto run10x``.
        repeat_mask: Optional repeat-mask GTF (``-m``).
        threads: ``-@`` thread count for samtools + velocyto.
        samtools_memory: Per-thread sort memory in MB.
    """

    generate_missing: bool = False
    bam_dir: Path | None = None
    gtf_path: Path | None = None
    repeat_mask: Path | None = None
    threads: int = 8
    samtools_memory: int = 2000


class VelocityConfig(StrictBaseModel):
    """scVelo RNA-velocity method configuration.

    ``grouping_col``/``sample_col``/``loom_path_col`` are structural keys, not
    biology. ``use_rep`` and ``use_rep_fallback`` name integration-output obsm
    keys. Velocity is computed once per group in ``use_rep`` space and
    re-projected onto every 2D embedding present.

    Attributes:
        enabled: Whether the velocity method runs.
        grouping_col: obs column to subset cell-lineage groups by.
        sample_col: obs column identifying a sample (barcode namespace).
        loom_path_col: manifest column holding each sample's ``.loom`` path.
        groups: Optional subset of ``grouping_col`` levels; None = every level.
        use_rep: Named representation for moments; None → fallback chain.
        use_rep_fallback: Ordered obsm keys tried when ``use_rep`` is unset/absent.
        mode: scVelo velocity mode (``dynamical`` | ``stochastic`` | ``deterministic``).
        min_shared_counts: ``scv.pp.filter_genes`` threshold.
        n_top_genes: HVG count.
        n_pcs: ``scv.pp.moments`` PCs.
        n_neighbors: ``scv.pp.moments`` neighbors.
        min_cells: Minimum cells per group to attempt velocity.
        n_jobs: Worker count (1 = reproducible).
        seed: Random seed threaded into ``recover_dynamics``.
        generation: Nested loom-generation gate.
    """

    enabled: bool = True
    grouping_col: str = "cell_type"
    sample_col: str = "sample_id"
    loom_path_col: str = "loom_path"
    groups: list[str] | None = None
    use_rep: str | None = None
    use_rep_fallback: list[str] = ["X_scANVI", "X_scVI", "X_pca"]
    mode: str = "dynamical"
    min_shared_counts: int = 20
    n_top_genes: int = 2000
    n_pcs: int = 30
    n_neighbors: int = 30
    min_cells: int = 30
    n_jobs: int = 1
    seed: int = 1337
    generation: VelocityGenerationConfig = VelocityGenerationConfig()


class TrajectoryConfig(StrictBaseModel):
    """Trajectory stage config. Spec #1 exposes one method: velocity.

    Attributes:
        enabled: Whether the stage runs.
        methods: Method sub-configs (empty → default [velocity] injected).
        velocity: The RNA-velocity method config.
    """

    enabled: bool = True
    methods: list[dict] = []
    velocity: VelocityConfig = VelocityConfig()


__all__ = ["TrajectoryConfig", "VelocityConfig", "VelocityGenerationConfig"]
