"""Configuration for principled subclustering subsystem."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class FocusConfig(StrictBaseModel):
    """
    Focus extraction configuration (which lineage to subcluster).

    Args:
        label_key: obs column containing cell-type labels.
        labels: list of labels to extract (empty = keep all cells).
    """

    label_key: str = "cell_type"
    labels: list[str] = []


class GroupFilterConfig(StrictBaseModel):
    """
    Group-level filter configuration (drop groups with < N focus cells).

    The generic "drop groups with < N cells of the focus type" rule.
    This is the KC<100-per-patient rule made atlas/lineage-agnostic.

    Args:
        group_key: obs column for grouping (e.g., patient_id, sample).
            None = no group filter applied.
        min_cells: minimum cells per group to keep.
            None = no group filter applied.
    """

    group_key: str | None = None
    min_cells: int | None = None


class ReembedConfig(StrictBaseModel):
    """
    Re-embedding configuration for focus subset (placeholder for Task 2).

    Args:
        representations: list of representations to compute.
        hvg: HVG selection parameters.
        integration: integration parameters.
    """

    representations: list[str] = ["batch_aware"]
    hvg: dict = {}
    integration: dict = {}


class PartitionConfig(StrictBaseModel):
    """
    Partition configuration (CHOIR + fallback grid, Task 2).

    Args:
        method: partition method (choir or leiden_grid).
        seeds: random seeds for ensemble partitioning.
        choir: CHOIR-specific parameters.
        leiden_grid: Leiden grid-search parameters.
    """

    method: str = "choir"
    seeds: list[int] = [0]
    choir: dict = {}
    leiden_grid: dict = {}


class FormalTestConfig(StrictBaseModel):
    """
    Formal significance test configuration (sc-SHC, Task 2).

    Args:
        method: test method (scshc).
        alpha: significance level.
    """

    method: str = "scshc"
    alpha: float = 0.05


class DonorGateConfig(StrictBaseModel):
    """
    Donor-reproducibility gatekeeper configuration (Task 3).

    Args:
        group_key: obs column for donor/group (e.g., donor_id).
            None = skip donor gate.
        min_groups: minimum number of groups required.
        min_cells_per_group: minimum cells a group must contribute to count as a
            supporting group toward min_groups. 0 disables the per-group floor.
        max_group_frac: max fraction of cluster cells allowed from a single group
            (one-donor-dominated detector). None = skip the check.
        leave_one_donor_out: whether to run LODO cross-validation.
        classifier_separability: whether to run RF classifier test.
    """

    group_key: str | None = None
    min_groups: int = 3
    min_cells_per_group: int = 10
    max_group_frac: float | None = 0.8
    leave_one_donor_out: bool = True
    classifier_separability: bool = True


class DiagnosticsConfig(StrictBaseModel):
    """
    Subclustering diagnostics configuration.

    Args:
        clustree: whether to generate clustree plot.
        stability_curve: whether to generate stability curve.
    """

    clustree: bool = True
    stability_curve: bool = True


class SubclusteringConfig(StrictBaseModel):
    """
    Principled subclustering configuration.

    Subclustering extracts a lineage (focus), applies group-level filters,
    re-embeds, partitions with CHOIR, tests significance with sc-SHC, and
    gates on donor reproducibility. All steps are atlas/lineage-agnostic.

    Args:
        enabled: whether subclustering stage may run.
        focus: focus extraction configuration (which lineage to extract).
        group_filter: group-level filter (drop groups with < N focus cells).
        counts_layer: layer containing raw counts.
        reembed: re-embedding configuration (placeholder for Task 2).
        partition: partition configuration (CHOIR + fallback, Task 2).
        formal_test: formal significance test configuration (sc-SHC, Task 2).
        donor_gate: donor-reproducibility gatekeeper (Task 3).
        diagnostics: diagnostic plot configuration.
        action: action when clusters fail gate (flag or drop).
        key_added: obs column for subcluster labels.
    """

    enabled: bool = False
    focus: FocusConfig = FocusConfig()
    group_filter: GroupFilterConfig = GroupFilterConfig()
    counts_layer: str = "counts"
    reembed: ReembedConfig = ReembedConfig()
    partition: PartitionConfig = PartitionConfig()
    formal_test: FormalTestConfig = FormalTestConfig()
    donor_gate: DonorGateConfig = DonorGateConfig()
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
    action: str = "flag"
    key_added: str = "subcluster"


__all__ = [
    "SubclusteringConfig",
    "FocusConfig",
    "GroupFilterConfig",
    "ReembedConfig",
    "PartitionConfig",
    "FormalTestConfig",
    "DonorGateConfig",
    "DiagnosticsConfig",
]
