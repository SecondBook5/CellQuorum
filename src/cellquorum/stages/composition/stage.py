# Pipeline step (order=190): composition — descriptive cell-type composition summary.
"""Cell-type composition summary stage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cellquorum.config.cohort import resolve_cohort_key
from cellquorum.core.artifacts import ArtifactManager
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.context_access import resolve_stage_config
from cellquorum.stages.composition.compute import grouped_composition, overall_composition
from cellquorum.stages.composition.config import CompositionConfig
from cellquorum.visualization import figstyle
from cellquorum.visualization.qc.summarise import natural_sort_key

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


@register_stage(
    name="composition",
    order=190,
    config_flag="composition",
    config_field="composition",
    category="composition",
)
class CompositionStage:
    """Compute and render overall, per-condition, and per-sample cell-type composition.

    Purely descriptive: statistical testing of compositional shifts belongs to
    ``differential_abundance``, not here. A missing grouping column (no condition,
    no sample) skips only the table/figure that needs it, never the whole stage —
    not every cohort has both.
    """

    def run(self, context: PipelineContext) -> StageResult:
        """
        Execute the composition stage.

        Args:
            context: Pipeline context with AnnData, config, and output paths.

        Returns:
            StageResult with unchanged AnnData plus composition artifacts.
        """

        adata = context.require_adata()
        config = CompositionConfig.model_validate(resolve_stage_config(context, "composition"))

        if config.cell_type_key not in adata.obs.columns:
            return StageResult.skipped(
                adata=adata,
                reason=f"missing obs column '{config.cell_type_key}'",
            )

        mask = _qc_mask(adata, config)
        obs = adata.obs.loc[mask] if mask is not None else adata.obs
        n_cells_used = int(len(obs))
        if n_cells_used == 0:
            return StageResult.skipped(adata=adata, reason="no cells survived atlas_states filter")

        top_config = getattr(context, "config", None)
        condition_key = _resolve_key(config.condition_key, top_config, "condition_key", "condition")
        donor_key = _resolve_key(config.donor_key, top_config, "donor_key", "donor_id")
        sample_key = _resolve_key(config.sample_key, top_config, "sample_key", "sample_id")

        cell_type = obs[config.cell_type_key]
        overall = overall_composition(cell_type)

        manager = ArtifactManager.from_root(context.paths.root)
        artifacts = [
            manager.write_dataframe(
                overall,
                name="composition_overall",
                relative_path=f"results/{config.output_prefix}_overall.csv",
                description="Overall cell-type composition (counts and fractions).",
            )
        ]

        figures_dir = Path(context.paths.figures) / "composition"
        formats = tuple(config.figure_formats)
        figstyle.set_style()

        condition_present = bool(condition_key) and condition_key in obs.columns
        if condition_present:
            assert condition_key is not None  # narrows for the type checker
            by_condition = grouped_composition(cell_type, obs[condition_key])
            artifacts.append(
                manager.write_dataframe(
                    by_condition.reset_index(names=condition_key),
                    name="composition_by_condition",
                    relative_path=f"results/{config.output_prefix}_by_condition.csv",
                    description="Cell-type composition per condition (percent).",
                )
            )
            fig = _stacked_bar(
                by_condition, x_label=condition_key, cluster_labels=config.cluster_labels
            )
            paths = figstyle.save_figure(
                fig, figures_dir, "composition_by_condition", formats=formats, dpi=config.dpi
            )
            artifacts.extend(
                StageArtifact(
                    name="composition_figure_condition",
                    path=p,
                    kind="figure",
                    description="Cell-type composition stacked bar, by condition.",
                )
                for p in paths
            )

        sample_present = bool(sample_key) and sample_key in obs.columns
        if sample_present:
            assert sample_key is not None  # narrows for the type checker
            by_sample = grouped_composition(cell_type, obs[sample_key])
            if donor_key and donor_key in obs.columns:
                sample_donor = obs.drop_duplicates(sample_key).set_index(sample_key)[donor_key]
                # donor_id is commonly a pandas Categorical on a real object --
                # .fillna("") on one raises unless "" is a declared category,
                # which it never is. astype(object) first fully decategorizes
                # (keeping real NaN as NaN) so fillna has an unconstrained
                # dtype to fill into.
                donor_keys = natural_sort_key(
                    sample_donor.reindex(by_sample.index).astype(object).fillna("").astype(str)
                )
                sample_keys = natural_sort_key(pd.Series(by_sample.index, index=by_sample.index))
                sort_order = (donor_keys + "|" + sample_keys).sort_values().index
                by_sample = by_sample.loc[sort_order]
            artifacts.append(
                manager.write_dataframe(
                    by_sample.reset_index(names=sample_key),
                    name="composition_by_sample",
                    relative_path=f"results/{config.output_prefix}_by_sample.csv",
                    description="Cell-type composition per sample (percent).",
                )
            )
            fig = _stacked_bar(by_sample, x_label=sample_key, cluster_labels=config.cluster_labels)
            paths = figstyle.save_figure(
                fig, figures_dir, "composition_by_sample", formats=formats, dpi=config.dpi
            )
            artifacts.extend(
                StageArtifact(
                    name="composition_figure_sample",
                    path=p,
                    kind="figure",
                    description="Cell-type composition stacked bar, by sample.",
                )
                for p in paths
            )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[
                f"composition summarized {n_cells_used} cells across "
                f"{overall.shape[0]} cell types."
            ],
            warnings=[],
            metrics={
                "n_cells_used": n_cells_used,
                "n_cell_types": int(overall.shape[0]),
                "condition_key": condition_key if condition_present else None,
                "sample_key": sample_key if sample_present else None,
                "donor_key": donor_key if donor_key in obs.columns else None,
            },
        )


def _resolve_key(stage_value: str | None, config: object, attr: str, default: str) -> str | None:
    """Resolve a structural key: stage config -> cohort -> the given default."""

    if stage_value:
        return stage_value
    return resolve_cohort_key(config, attr=attr, stage_value=default)


def _qc_mask(adata: object, config: CompositionConfig) -> np.ndarray | None:
    """Boolean mask restricting to configured QC states, or None for "keep all".

    Mirrors the embeddings stage's qc_state_column auto-detect: qc_state_final
    (qc_finalization's own final verdict) when present, else qc_state_initial. A
    hardcoded default naming the initial column would go silently stale the
    moment qc_finalization is enabled without separately updating this column too.
    """

    if not config.atlas_states:
        return None
    if config.qc_state_column:
        column = config.qc_state_column
    elif "qc_state_final" in adata.obs.columns:
        column = "qc_state_final"
    else:
        column = "qc_state_initial"
    if column not in adata.obs.columns:
        return None
    mask = adata.obs[column].astype(str).isin(config.atlas_states).to_numpy()

    if config.exclude_multiplets and config.multiplet_column in adata.obs.columns:
        is_multiplet = adata.obs[config.multiplet_column].to_numpy(dtype=bool)
        mask = mask & ~is_multiplet

    return mask


def _stacked_bar(
    table: pd.DataFrame, *, x_label: str, cluster_labels: dict[str, str] | None = None
) -> Figure:
    """Percent-stacked bar of cell-type composition, one bar per row of `table`.

    ``cluster_labels`` renames the legend only -- ``table``'s columns (the
    actual grouping the bars are computed from) are never touched, same
    display-only convention as de_heatmap's cluster_labels.
    """

    import matplotlib.pyplot as plt

    cluster_labels = cluster_labels or {}
    groups = list(table.index)
    cell_types = list(table.columns)
    palette = {
        ct: figstyle.CATEGORICAL_PALETTE[i % len(figstyle.CATEGORICAL_PALETTE)]
        for i, ct in enumerate(cell_types)
    }
    fig, ax = plt.subplots(figsize=(max(3.0, 0.6 * len(groups) + 2.0), 4.4))
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for ct in cell_types:
        values = table[ct].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.7,
            color=palette[ct],
            edgecolor="white",
            linewidth=0.4,
            label=cluster_labels.get(str(ct), ct),
        )
        bottom += values
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Composition (% of cells)")
    ax.set_xlabel(x_label)
    ax.set_title("Cell-type composition")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="Cell type", fontsize=8)
    return fig


__all__ = ["CompositionStage"]
