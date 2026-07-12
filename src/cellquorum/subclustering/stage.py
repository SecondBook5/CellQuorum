"""Subclustering stage implementation."""

from __future__ import annotations

from pathlib import Path

from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.subclustering.diagnostics import plot_group_recovery
from cellquorum.subclustering.extract import apply_group_filter, extract_focus


class SubclusteringStage:
    """
    Principled subclustering stage.

    This stage orchestrates:
    1. Focus extraction (subset to lineage)
    2. Group filter (drop groups with < N focus cells)
    3. Re-embedding (Task 2)
    4. Partition (CHOIR + sc-SHC, Task 2)
    5. Donor gate (Task 3)
    6. Diagnostics (clustree, stability curve)

    Task 1 scope: steps 1-2 only (extract + group_filter).
    """

    # Stable stage name (satisfies the PipelineStage Protocol).
    name = "subclustering"
    stage_category = "subclustering"

    def run(self, context: object) -> StageResult:
        """
        Run principled subclustering pipeline.

        Args:
            context: Pipeline context exposing config, adata, paths.

        Returns:
            StageResult with focused/filtered subset as adata.
        """
        # Resolve config and adata.
        config = getattr(context, "config", None)
        sc_config = getattr(config, "subclustering", None)
        adata = getattr(context, "adata", None)

        # Skip (non-silent) when disabled.
        if sc_config is None or not getattr(sc_config, "enabled", False):
            return StageResult(
                adata=adata,
                warnings=["subclustering disabled by config"],
                metrics={"skipped": True, "reason": "disabled by config"},
            )

        # Require adata.
        if adata is None:
            return StageResult(
                adata=None,
                warnings=["subclustering skipped: no adata available"],
                metrics={"skipped": True, "reason": "no adata"},
            )

        # Extract focus lineage.
        focused = extract_focus(
            adata,
            sc_config.focus,
            sc_config.counts_layer,
        )

        notes = []
        warnings = []
        artifacts = []
        metrics = {}

        # Record focus extraction provenance.
        if "subcluster_extraction" in focused.uns:
            prov = focused.uns["subcluster_extraction"]
            notes.append(
                f"Focus extracted: {prov['n_cells_kept']} / "
                f"{prov['n_cells_total']} cells "
                f"({prov['label_key']} in {prov['labels']})"
            )
            metrics["n_cells_before_focus"] = prov["n_cells_total"]
            metrics["n_cells_after_focus"] = prov["n_cells_kept"]

        # Apply group filter (if configured).
        group_key = sc_config.group_filter.group_key
        min_cells = sc_config.group_filter.min_cells
        if group_key is not None and min_cells is not None:
            filtered, filter_prov = apply_group_filter(
                focused,
                group_key,
                min_cells,
            )

            # Record group filter provenance.
            if filter_prov["applied"]:
                n_kept = len(filter_prov["kept"])
                n_dropped = len(filter_prov["dropped"])
                notes.append(
                    f"Group filter: kept {n_kept} groups, "
                    f"dropped {n_dropped} groups "
                    f"(< {min_cells} cells in {group_key})"
                )
                if filter_prov["dropped"]:
                    warnings.append(f"Dropped groups: {', '.join(filter_prov['dropped'])}")
                metrics["n_groups_kept"] = n_kept
                metrics["n_groups_dropped"] = n_dropped
                metrics["n_cells_after_group_filter"] = filtered.n_obs

                # Generate recovery plot.
                paths = getattr(context, "paths", None)
                if paths is not None:
                    figures_dir = Path(getattr(paths, "figures", "."))
                    recovery_path = figures_dir / "subclustering_group_recovery.png"
                    plot_group_recovery(
                        filter_prov["counts"],
                        min_cells,
                        group_key,
                        recovery_path,
                    )
                    artifacts.append(
                        StageArtifact(
                            name="subclustering_group_recovery",
                            path=recovery_path,
                            kind="png",
                            description=(
                                f"Group recovery: {group_key} cell counts "
                                f"vs. threshold ({min_cells} cells)"
                            ),
                        )
                    )
            else:
                filtered = focused
                notes.append("Group filter: skipped (not configured)")
        else:
            filtered = focused
            notes.append("Group filter: skipped (not configured)")

        # Task 2-3: partition + formal_test + donor_gate (hooks for later).
        # For Task 1, return the filtered focus subset.
        notes.append("Partition, formal_test, donor_gate: deferred to Tasks 2-3")

        return StageResult(
            adata=filtered,
            artifacts=artifacts,
            notes=notes,
            warnings=warnings,
            metrics=metrics,
        )


__all__ = ["SubclusteringStage"]
