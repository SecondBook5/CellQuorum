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
        threads: ``-@`` thread count for samtools + velocyto. Deliberately NOT
            inherited from ``compute.n_jobs`` like the other worker counts are:
            these threads sort a BAM, so they are bound by disk rather than by
            CPU, and each one claims ``samtools_memory`` MB. Inheriting a CPU
            worker count would silently change how much memory a sort takes,
            which is the paired knob below, not this one.
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
        n_jobs: Worker count for ``recover_dynamics`` / ``velocity_graph``.
            ``None`` inherits ``compute.n_jobs``. Worker count does not affect the
            result: at 1, 4 and 8 workers the ``velocity`` layer, every ``fit_*``
            parameter and ``velocity_graph`` come out bit-identical, and the one
            output that does move — ``velocity_pseudotime``, by ~1e-5 — moves just
            as much between two runs at the SAME worker count and is not fixed by
            re-seeding numpy or pinning root/end cells (the wobble is inside the
            eigensolver, out of reach of any seed we control). So pinning this to
            1 buys no reproducibility; it only costs time, and this is the most
            expensive step in the pipeline: 151s serial vs 20s on 8 workers for
            1200 cells x 2000 genes.
        seed: Random seed threaded into ``recover_dynamics``.
        whole_object: Also run velocity once on the WHOLE object (writes
            ``whole_object.h5ad`` with Ms + velocity layers) so CellRank's
            VelocityKernel can consume it. Per-group behaviour is unchanged.
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
    n_jobs: int | None = None
    seed: int = 1337
    whole_object: bool = False
    generation: VelocityGenerationConfig = VelocityGenerationConfig()


class CellRankConfig(StrictBaseModel):
    """CellRank 2.x GPCCA fate-mapping method configuration.

    All keys are structural, not biology. ``cluster_key`` names the obs column
    whose levels seed macrostate→terminal assignment; ``pseudotime_key`` /
    ``cytotrace_key`` name optional directionality inputs. CellRank runs on the
    WHOLE object (never per-group) so cross-lineage fate probabilities are valid.

    Attributes:
        enabled: Whether the method runs.
        cluster_key: obs column seeding macrostate→terminal assignment.
        pseudotime_key: Whole-object pseudotime obs col; None → connectivity-only.
        cytotrace_key: Optional CytoTRACE obs col for a CytoTRACEKernel.
        use_velocity: Consume the whole-object velocity h5ad (written when
            ``VelocityConfig.whole_object`` is set) to build a VelocityKernel.
        velocity_model: VelocityKernel transition model
            (``deterministic`` | ``stochastic`` | ``monte_carlo``).
        time_key: obs column naming an experimental time/stage axis for a moscot
            RealTimeKernel; None → the RealTimeKernel is skipped.
        realtime_epsilon: moscot ``TemporalProblem.solve`` regularization.
        use_rep: Rep to build neighbors when connectivities absent; None → fallback.
        use_rep_fallback: Ordered obsm keys tried when ``use_rep`` unset/absent.
        n_neighbors: Neighbor count when a graph must be built.
        weight_connectivities: w in (1-w)*directionality + w*connectivity.
        n_components: Schur vectors; clamped to [n_states+1, n_obs-1].
        n_states: Macrostate REQUEST (CellRank may return more).
        n_terminal_states: Terminal-state count; None → method-driven auto.
        terminal_method: predict_terminal_states method.
        predict_initial_states: Best-effort initial-state prediction.
        n_initial_states: Initial states to predict when enabled.
        max_cells: None = no cap; else seeded subsample for GPCCA.
        seed: Random seed for subsample + lineage drivers.
    """

    enabled: bool = True
    cluster_key: str = "cell_type"
    pseudotime_key: str | None = None
    cytotrace_key: str | None = None
    use_velocity: bool = False
    velocity_model: str = "deterministic"
    time_key: str | None = None
    realtime_epsilon: float = 0.1
    use_rep: str | None = None
    use_rep_fallback: list[str] = ["X_scANVI", "X_scVI", "X_pca"]
    n_neighbors: int = 30
    weight_connectivities: float = 0.2
    n_components: int = 20
    n_states: int = 8
    n_terminal_states: int | None = None
    terminal_method: str = "stability"
    predict_initial_states: bool = False
    n_initial_states: int = 1
    max_cells: int | None = None
    seed: int = 1337


class DptConfig(StrictBaseModel):
    """Diffusion pseudotime (scanpy) producer configuration.

    All keys are structural, not biology. Writes a whole-object
    ``dpt_pseudotime`` obs column that CellRank's PseudotimeKernel can consume.
    A root MUST be resolvable (marker-score argmax or a root group); scanpy 1.12
    silently emits meaningless pseudotime when ``iroot`` is unset, so the method
    skips rather than emit an unrooted result.

    Attributes:
        enabled: Whether the method runs.
        use_rep: Rep to build neighbors/diffmap; None → fallback chain.
        use_rep_fallback: Ordered obsm keys tried when ``use_rep`` unset/absent.
        n_neighbors: Neighbor count when a graph must be built.
        n_comps: Diffusion-map components (``sc.tl.diffmap``).
        n_dcs: Diffusion components used by ``sc.tl.dpt``.
        n_branchings: DPT branch detections (0 = pseudotime only).
        root_key: obs column naming the root group (used with ``root_group``).
        root_group: level of ``root_key`` whose centroid seeds the root.
        root_marker_score_key: obs column with a precomputed score; root = argmax.
        exclude_outliers: Opt-in 5-MAD robust-z outlier flagging before diffmap.
        outlier_mad: Robust-z threshold for outlier flagging.
        orient_by_score_key: obs column; sign-check corr(dpt, score), re-root once
            if orientation is reversed.
        seed: Random seed (determinism; diffmap/dpt are otherwise deterministic).
    """

    enabled: bool = True
    use_rep: str | None = None
    use_rep_fallback: list[str] = ["X_scANVI", "X_scVI", "X_pca"]
    n_neighbors: int = 15
    n_comps: int = 15
    n_dcs: int = 10
    n_branchings: int = 0
    root_key: str | None = None
    root_group: str | None = None
    root_marker_score_key: str | None = None
    exclude_outliers: bool = False
    outlier_mad: float = 5.0
    orient_by_score_key: str | None = None
    seed: int = 1337


class PalantirConfig(StrictBaseModel):
    """Palantir pseudotime producer configuration.

    All keys are structural, not biology. Runs the mandatory 3-step Palantir
    pipeline (diffusion maps → multiscale space → run_palantir) on the whole
    object and writes ``palantir_pseudotime`` + ``palantir_entropy`` obs columns
    (plus fate probabilities in obsm) that downstream methods can consume.

    Attributes:
        enabled: Whether the method runs.
        use_rep: Rep for diffusion maps; None → fallback chain.
        use_rep_fallback: Ordered obsm keys tried when ``use_rep`` unset/absent.
        n_components: Diffusion-map components.
        knn: Nearest neighbors for diffusion maps / palantir.
        n_eigs: Eigenvectors for ``determine_multiscale_space``.
        num_waypoints: Waypoint count (clamped to n_obs at runtime).
        root_key: obs column naming the root group (used with ``root_group``).
        root_group: level of ``root_key`` whose centroid seeds ``early_cell``.
        root_marker_score_key: obs column with a precomputed score; root = argmax.
        max_cells: None = no cap; else seeded subsample, results reindexed (NaN
            outside sample); the root cell is always retained in the subsample.
        seed: Random seed threaded into diffusion maps + run_palantir + subsample.
    """

    enabled: bool = True
    use_rep: str | None = None
    use_rep_fallback: list[str] = ["X_scANVI", "X_scVI", "X_pca"]
    n_components: int = 10
    knn: int = 30
    n_eigs: int = 10
    num_waypoints: int = 1200
    root_key: str | None = None
    root_group: str | None = None
    root_marker_score_key: str | None = None
    max_cells: int | None = None
    seed: int = 1337


class CytoTraceConfig(StrictBaseModel):
    """CytoTRACE 2 developmental-potency producer configuration.

    All keys are structural, not biology. Runs the pretrained CytoTRACE 2 model
    on the whole object and writes a ``cytotrace2_score`` obs column (plus a
    categorical ``cytotrace2_potency``) that CellRank's CytoTRACEKernel can
    consume via ``CellRankConfig.cytotrace_key``. The ``cytotrace2-py`` package
    is optional; the method skips when it is not importable.

    Attributes:
        enabled: Whether the method runs.
        species: ``human`` | ``mouse`` (selects the pretrained model).
        counts_layer: Layer holding raw counts; None → use ``.X``.
        batch_size: Model prediction batch size.
        smooth_batch_size: Diffusion-smoothing batch size.
        disable_parallelization: Force single-process execution (reproducible).
        seed: Random seed threaded into the CytoTRACE 2 run.
    """

    enabled: bool = True
    species: str = "human"
    counts_layer: str | None = None
    batch_size: int = 20000
    smooth_batch_size: int = 1000
    disable_parallelization: bool = False
    seed: int = 14


class TrajectoryConfig(StrictBaseModel):
    """Trajectory stage config. Spec #1 exposes one method: velocity.

    Attributes:
        enabled: Whether the stage runs.
        methods: Method sub-configs (empty → default [velocity] injected).
        velocity: The RNA-velocity method config.
        cellrank: The CellRank fate-mapping method config.
        dpt: The DPT pseudotime method config.
        palantir: The Palantir pseudotime method config.
        cytotrace: The CytoTRACE 2 potency method config.
    """

    enabled: bool = True
    methods: list[dict] = []
    velocity: VelocityConfig = VelocityConfig()
    cellrank: CellRankConfig = CellRankConfig()
    dpt: DptConfig = DptConfig()
    palantir: PalantirConfig = PalantirConfig()
    cytotrace: CytoTraceConfig = CytoTraceConfig()


__all__ = [
    "TrajectoryConfig",
    "VelocityConfig",
    "VelocityGenerationConfig",
    "CellRankConfig",
    "DptConfig",
    "PalantirConfig",
    "CytoTraceConfig",
]
