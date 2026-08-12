"""Configuration for the cell-cell communication (LR co-expression) stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class CellCellCommunicationConfig(StrictBaseModel):
    """LIANA consensus LR scoring + Tensor-cell2cell decomposition.

    All biological specifics (cell-type/sample columns, resource) come from
    config — no study assumptions in code. ``cell_type_col``/``sample_col`` are
    structural keys.

    Attributes:
        enabled: Whether the stage runs.
        methods: Method sub-configs (empty → default [liana, tensor_c2c] injected).
        cell_type_col: obs column with cell-type labels (LIANA groupby).
        sample_col: obs column identifying a sample (per-sample LIANA + tensor context).
        layer: Log-normalized expression layer read by LIANA.
        seed: Random seed for stochastic steps.
        resource_name: LIANA ligand-receptor resource.
        expr_prop: Minimum expression proportion for LIANA.
        min_cells: Minimum cells per group for LIANA.
        n_perms: LIANA permutation count.
        rank: Tensor decomposition rank (None → elbow auto-select).
        tf_optimization: 'robust' (runs=100) or 'regular' (runs=1).
        min_samples: Minimum distinct samples to attempt tensor decomposition.
        tensor_how: How to handle missing indices when building the tensor.
        outer_fraction: Fraction threshold for the 'outer' join.
        timeout_seconds: Reserved for parity with other stages.
    """

    enabled: bool = True
    methods: list[dict] = []
    cell_type_col: str = "cell_type"
    sample_col: str = "sample_id"
    layer: str = "cellquorum_normalized"
    seed: int = 42
    # LIANA
    resource_name: str = "consensus"
    expr_prop: float = 0.1
    min_cells: int = 5
    n_perms: int = 100
    # Tensor-cell2cell
    rank: int | None = None
    tf_optimization: str = "robust"
    min_samples: int = 3
    tensor_how: str = "outer"
    outer_fraction: float = 1.0 / 3.0
    timeout_seconds: int = 1800
    # Device for the tensor decomposition: 'cpu', 'cuda'/'gpu', or None to
    # auto-resolve from compute.prefer_gpu (falls back to CPU if CUDA is
    # unavailable). GPU offload keeps robust factorization tractable.
    device: str | None = None
    # --- NicheNet / MultiNicheNet (spec #2) — optional; MethodSkip when unset ---
    # Prior-model RDS paths (organism-specific biology → config, never bundled).
    nichenet_ligand_target_matrix: str | None = None
    nichenet_lr_network: str | None = None
    nichenet_weighted_networks: str | None = None
    # Shared knobs
    nichenet_min_cells: int = 10
    nichenet_expr_prop: float = 0.10
    nichenet_n_cores: int = 4
    nichenet_timeout_seconds: int = 7200
    # MultiNicheNet-specific (AJ's validated defaults)
    mnn_fraction_cutoff: float = 0.05
    mnn_min_sample_prop: float = 0.5
    mnn_logfc_threshold: float = 0.5
    mnn_p_val_threshold: float = 0.05
    mnn_p_val_adj: bool = False
    mnn_top_n_target: int = 250
    mnn_scenario: str = "regular"
    # NicheNet-specific
    nichenet_sender: str | None = None
    nichenet_receiver: str | None = None
    nichenet_de_csv: str | None = None
    nichenet_top_ligands: int = 10
    nichenet_top_targets: int = 50
    nichenet_de_fdr: float = 0.05
    nichenet_de_top_n: int = 200


__all__ = ["CellCellCommunicationConfig"]
