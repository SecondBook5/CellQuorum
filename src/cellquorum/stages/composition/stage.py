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
    import anndata as ad
    from matplotlib.axes import Axes

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
        figstyle.apply_theme("compact")

        condition_present = bool(condition_key) and condition_key in obs.columns
        if condition_present:
            assert condition_key is not None  # narrows for the type checker
            by_condition = grouped_composition(
                cell_type, obs[condition_key], group_order=_condition_order(top_config)
            )
            artifacts.append(
                manager.write_dataframe(
                    by_condition.reset_index(names=condition_key),
                    name="composition_by_condition",
                    relative_path=f"results/{config.output_prefix}_by_condition.csv",
                    description="Cell-type composition per condition (percent).",
                )
            )
            fig = _stacked_bar(by_condition, x_label=None, cluster_labels=config.cluster_labels)
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

        donor_present = bool(donor_key) and donor_key in obs.columns
        if condition_present and donor_present:
            assert condition_key is not None and donor_key is not None
            by_patient = _composition_by_patient(
                cell_type,
                obs[condition_key],
                obs[donor_key],
                condition_order=_condition_order(top_config),
            )
            table = pd.concat(by_patient, names=[condition_key, donor_key]).reset_index()
            artifacts.append(
                manager.write_dataframe(
                    table,
                    name="composition_by_patient",
                    relative_path=f"results/{config.output_prefix}_by_patient.csv",
                    description="Per-patient cell-type composition in each condition (percent).",
                )
            )
            fig = _stacked_bar_by_patient(
                by_patient,
                cluster_labels=config.cluster_labels,
                title="Cell type composition by patient",
            )
            paths = figstyle.save_figure(
                fig, figures_dir, "composition_by_patient", formats=formats, dpi=config.dpi
            )
            artifacts.extend(
                StageArtifact(
                    name="composition_figure_patient",
                    path=p,
                    kind="figure",
                    description="Composition stacked bars per patient, a panel per condition.",
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


def _composition_by_patient(
    cell_type: pd.Series,
    condition: pd.Series,
    donor: pd.Series,
    *,
    condition_order: list[str],
) -> dict[str, pd.DataFrame]:
    """Per-patient composition within each condition, conditions in the given order.

    One table per condition, patients as rows, every table with the same cell-type columns
    (ordered by overall abundance) so the panels stack identically.
    """
    joint = condition.astype(str) + "\t" + donor.astype(str)
    table = grouped_composition(cell_type, joint)
    index = pd.MultiIndex.from_tuples([tuple(i.split("\t")) for i in table.index])
    table.index = index
    conditions = [c for c in condition_order if c in set(index.get_level_values(0))]
    conditions += sorted(set(index.get_level_values(0)) - set(conditions))
    out: dict[str, pd.DataFrame] = {}
    for c in conditions:
        sub = pd.DataFrame(table.loc[c])
        sub.index = sub.index.astype(str)
        out[c] = sub
    return out


def _condition_order(config: object) -> list[str]:
    """Control first, then case, from the design block; empty when the design names neither.

    The control is always the left bar and the case the right, whatever the alphabet says.
    """
    design = getattr(config, "design", None)
    named = [getattr(design, "control", None), getattr(design, "case", None)]
    return [str(name) for name in named if name]


def _qc_mask(adata: ad.AnnData, config: CompositionConfig) -> np.ndarray | None:
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


def _stack_palette(n: int) -> list[str]:
    """Dark red at the bottom of the stack to dark blue at the top, skipping the near-white
    middle of the ramp so no segment disappears against the page."""
    import matplotlib as mpl

    ramp = mpl.colormaps["RdBu"]
    positions = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
    mapped = np.where(positions <= 0.5, positions * 0.8, 0.6 + (positions - 0.5) * 0.8)
    return [mpl.colors.to_hex(ramp(float(m))) for m in mapped]


def _stack_bars(
    ax: Axes,
    table: pd.DataFrame,
    x: np.ndarray,
    colours: dict[str, str],
    cluster_labels: dict[str, str],
    *,
    width: float,
) -> None:
    """Draw ``table`` (rows in percent) as stacked bars at ``x``, first column at the bottom."""
    bottom = np.zeros(len(table))
    for ct in table.columns:
        values = table[ct].to_numpy() / 100.0
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=width,
            color=colours[ct],
            edgecolor="white",
            linewidth=0.4,
            label=cluster_labels.get(str(ct), ct),
        )
        bottom += values


def _stacked_bar(
    table: pd.DataFrame, *, x_label: str | None, cluster_labels: dict[str, str] | None = None
) -> Figure:
    """Stacked bars of cell-type proportion, one bar per row of `table` (rows in percent).

    The first column is the bottom of the stack and the legend lists the cell types in the
    order they are stacked, top to bottom. ``cluster_labels`` renames the legend only --
    ``table``'s columns (the actual grouping the bars are computed from) are never touched,
    same display-only convention as de_heatmap's cluster_labels.
    """

    import matplotlib.pyplot as plt

    cluster_labels = cluster_labels or {}
    groups = list(table.index)
    cell_types = list(table.columns)
    colours = dict(zip(cell_types, _stack_palette(len(cell_types)), strict=True))
    fig, ax = plt.subplots(figsize=(max(3.0, 0.6 * len(groups) + 2.0), 4.4))
    x = np.arange(len(groups))
    _stack_bars(ax, table, x, colours, cluster_labels, width=0.7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xticks(x)
    ax.set_xticklabels(
        groups,
        rotation=45 if len(groups) > 4 else 0,
        ha="right" if len(groups) > 4 else "center",
        fontsize=9,
    )
    ax.set_ylabel("Proportion of cells")
    if x_label:
        ax.set_xlabel(x_label)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Cell type",
        fontsize=8,
        frameon=False,
    )
    return fig


def _stacked_bar_by_patient(
    tables: dict[str, pd.DataFrame],
    *,
    cluster_labels: dict[str, str] | None = None,
    title: str | None = None,
) -> Figure:
    """One panel per condition, in the order given, each with a stacked bar per patient.

    ``tables`` maps condition -> patients (rows, in percent) by cell type (columns). Patients
    sit in natural order (P1, P2, P10) at the same place in every panel, so a patient reads
    straight across; one with no cells in a condition has no bar there. The panels share the
    proportion axis and one legend, in stack order.
    """
    import matplotlib.pyplot as plt

    cluster_labels = cluster_labels or {}
    conditions = list(tables)
    cell_types = list(next(iter(tables.values())).columns)
    donors = sorted(
        {d for t in tables.values() for d in t.index},
        key=lambda d: natural_sort_key(pd.Series([str(d)])).iloc[0],
    )
    position = {d: i for i, d in enumerate(donors)}
    colours = dict(zip(cell_types, _stack_palette(len(cell_types)), strict=True))
    fig, axes = plt.subplots(
        1,
        len(conditions),
        sharey=True,
        figsize=(1.0 + len(conditions) * (0.45 * len(donors) + 0.5), 4.6),
        squeeze=False,
    )
    for ax, condition in zip(axes[0], conditions, strict=True):
        table = tables[condition]
        x = np.array([position[d] for d in table.index], dtype=float)
        _stack_bars(ax, table, x, colours, cluster_labels, width=0.85)
        ax.set_xlim(-0.6, len(donors) - 0.4)
        ax.set_xticks(range(len(donors)))
        ax.set_xticklabels(donors, rotation=45, ha="right", fontsize=8.5)
        ax.set_title(
            condition,
            fontsize=9.5,
            fontweight="bold",
            pad=14,
            bbox={"boxstyle": "square,pad=0.4", "fc": "white", "ec": "black", "lw": 0.8},
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylim(0, 1)
    axes[0][0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axes[0][0].set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    axes[0][0].set_ylabel("Proportion of cells")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        title="Cell type",
        fontsize=8,
        frameon=False,
    )
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1))
    return fig


__all__ = ["CompositionStage"]
