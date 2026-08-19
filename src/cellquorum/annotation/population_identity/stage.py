"""Generic population/state identity evidence stage.

This stage is deliberately not atlas-specific. If reference mapping exists, the
reference label becomes external identity evidence. If no atlas exists, the
stage falls back to annotation labels or native clusters and reports them as
dataset-native candidates with explicit support and limitations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.annotation.population_identity.config import PopulationIdentityConfig
from cellquorum.core.artifacts import ArtifactManager
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.visualization.figstyle import (
    categorical_embedding,
    clean_axis,
    condition_palette,
    save_publication_figure,
    set_publication_style,
)


@dataclass(frozen=True)
class IdentityResolution:
    """Resolved candidate identity column and optional evidence columns."""

    candidate_key: str
    candidate_source: str
    reference_key: str | None
    annotation_key: str | None
    cluster_key: str | None
    sample_key: str | None
    donor_key: str | None
    condition_key: str | None
    confidence_key: str | None
    entropy_key: str | None
    embedding_key: str | None


class PopulationIdentityStage:
    """Write generic population/state identity tables, plots, and evidence."""

    name = "population_identity"

    def run(self, context: object) -> StageResult:
        """Execute the population-identity stage."""

        adata = context.require_adata()
        config = resolve_population_identity_config(context)
        if not config.enabled:
            return StageResult.skipped(
                adata=adata,
                reason="disabled by config",
                warnings=["population_identity disabled by config"],
            )

        resolution = resolve_identity_columns(adata, context, config)
        if resolution is None:
            return StageResult.skipped(
                adata=adata,
                reason="no population identity column available",
                warnings=[
                    "population_identity requires at least one reference, annotation, or "
                    "cluster column."
                ],
                metrics={
                    "candidate_key": config.candidate_key,
                    "reference_keys": config.reference_keys,
                    "annotation_keys": config.annotation_keys,
                    "cluster_key": config.cluster_key,
                },
            )

        tables = build_population_identity_tables(adata, resolution, config)

        # A candidate column can still yield no populations (0 cells, or all-NA
        # labels). An empty summary has no 'evidence_status' column, so skip
        # cleanly rather than crashing the metrics block that indexes it.
        if tables["population_summary"].empty:
            return StageResult.skipped(
                adata=adata,
                reason="no populations to characterize (empty candidate grouping)",
                warnings=[
                    "population_identity: candidate column "
                    f"'{resolution.candidate_key}' produced no populations "
                    "(0 cells or all-NA labels)."
                ],
                metrics={
                    "candidate_key": resolution.candidate_key,
                    "candidate_source": resolution.candidate_source,
                    "n_populations": 0,
                },
            )

        audit = build_population_identity_audit(adata, resolution, tables, config)

        cq = adata.uns.setdefault("cellquorum", {})
        cq["population_identity"] = {
            "resolution": resolution.__dict__,
            "audit": audit,
            "summary": tables["population_summary"].to_dict(orient="records"),
        }

        case, control = resolve_case_control(context)
        manager = ArtifactManager.from_root(context.paths.root)
        artifacts = write_population_identity_artifacts(
            manager=manager,
            adata=adata,
            resolution=resolution,
            tables=tables,
            audit=audit,
            config=config,
            case=case,
            control=control,
        )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[
                "population_identity used "
                f"obs['{resolution.candidate_key}'] as {resolution.candidate_source} evidence.",
                f"wrote {len(tables['population_summary'])} population candidate(s).",
            ],
            warnings=list(audit["warnings"]),
            metrics={
                "candidate_key": resolution.candidate_key,
                "candidate_source": resolution.candidate_source,
                "n_populations": int(len(tables["population_summary"])),
                "reference_key": resolution.reference_key,
                "annotation_key": resolution.annotation_key,
                "cluster_key": resolution.cluster_key,
                "embedding_key": resolution.embedding_key,
                "status_counts": tables["population_summary"]["evidence_status"]
                .value_counts()
                .to_dict(),
            },
            method_version="population_identity_v1",
            backend="python",
            device="cpu",
        )


def resolve_population_identity_config(context: object) -> PopulationIdentityConfig:
    """Resolve population-identity config from a PipelineContext-like object."""

    context_config = getattr(context, "config", None)
    if isinstance(context_config, PopulationIdentityConfig):
        return context_config
    if isinstance(context_config, Mapping):
        return PopulationIdentityConfig(**dict(context_config.get("population_identity", {})))
    subconfig = getattr(context_config, "population_identity", None)
    if isinstance(subconfig, PopulationIdentityConfig):
        return subconfig
    if hasattr(subconfig, "model_dump"):
        return PopulationIdentityConfig(**subconfig.model_dump())
    if isinstance(subconfig, Mapping):
        return PopulationIdentityConfig(**dict(subconfig))
    return PopulationIdentityConfig()


def resolve_identity_columns(
    adata: ad.AnnData,
    context: object,
    config: PopulationIdentityConfig,
) -> IdentityResolution | None:
    """Resolve candidate population and supporting evidence columns."""

    obs = adata.obs
    reference_key = first_present_column(obs, config.reference_keys)
    annotation_key = first_present_column(obs, config.annotation_keys)
    cluster_key = config.cluster_key if config.cluster_key in obs.columns else None

    candidate_key: str | None = None
    candidate_source = "unknown"
    if config.candidate_key is not None:
        if config.candidate_key in obs.columns:
            candidate_key = config.candidate_key
            candidate_source = "configured"
        else:
            return None
    elif reference_key is not None:
        candidate_key = reference_key
        candidate_source = "reference"
    elif annotation_key is not None:
        candidate_key = annotation_key
        candidate_source = "annotation"
    elif cluster_key is not None:
        candidate_key = cluster_key
        candidate_source = "cluster"

    if candidate_key is None:
        return None

    donor_key, condition_key = resolve_design_keys(context, config)

    # Cohort sample_key wins when set; else the stage value; else common names.
    from cellquorum.config.cohort import resolve_cohort_key

    resolved_sample_key = resolve_cohort_key(
        getattr(context, "config", None), attr="sample_key", stage_value=config.sample_key
    )
    sample_key = (
        resolved_sample_key
        if resolved_sample_key is not None and resolved_sample_key in obs.columns
        else first_present_column(obs, ["sample_id", "sample", "library_id"])
    )
    confidence_key = first_present_numeric_column(obs, config.confidence_keys)
    entropy_key = first_present_numeric_column(obs, config.entropy_keys)
    embedding_key = first_present_embedding(adata, config.embedding_keys)

    return IdentityResolution(
        candidate_key=candidate_key,
        candidate_source=candidate_source,
        reference_key=reference_key,
        annotation_key=annotation_key,
        cluster_key=cluster_key,
        sample_key=sample_key,
        donor_key=donor_key if donor_key in obs.columns else None,
        condition_key=condition_key if condition_key in obs.columns else None,
        confidence_key=confidence_key,
        entropy_key=entropy_key,
        embedding_key=embedding_key,
    )


def resolve_design_keys(
    context: object,
    config: PopulationIdentityConfig,
) -> tuple[str, str]:
    """Resolve donor and condition columns from config or design settings."""

    context_config = getattr(context, "config", None)

    # Cohort schema wins when it declares the structural keys (declare-once).
    from cellquorum.config.cohort import resolve_cohort_key

    donor_key = resolve_cohort_key(context_config, attr="donor_key", stage_value=config.donor_key)
    condition_key = resolve_cohort_key(
        context_config, attr="condition_key", stage_value=config.condition_key
    )

    design = getattr(context_config, "design", None)
    if isinstance(context_config, Mapping):
        design = context_config.get("design", {})

    if donor_key is None:
        donor_key = (
            design.get("donor_col", "patient_id")
            if isinstance(design, Mapping)
            else getattr(design, "donor_col", "patient_id")
        )
    if condition_key is None:
        condition_key = (
            design.get("condition_col", "condition")
            if isinstance(design, Mapping)
            else getattr(design, "condition_col", "condition")
        )
    return donor_key, condition_key


def resolve_case_control(context: object) -> tuple[str | None, str | None]:
    """Resolve primary case/control condition tokens from the design config."""

    context_config = getattr(context, "config", None)
    design = getattr(context_config, "design", None)
    if isinstance(context_config, Mapping):
        design = context_config.get("design", {})
    if isinstance(design, Mapping):
        return design.get("case"), design.get("control")
    return getattr(design, "case", None), getattr(design, "control", None)


def build_population_identity_tables(
    adata: ad.AnnData,
    resolution: IdentityResolution,
    config: PopulationIdentityConfig,
) -> dict[str, pd.DataFrame]:
    """Build all population-identity evidence tables."""

    obs = adata.obs.copy()
    obs["_population_id"] = obs[resolution.candidate_key].astype(str).fillna("NA")
    summary = build_population_summary(obs, resolution, config)
    tables = {
        "population_summary": summary,
        "population_by_sample": build_crosstab(obs, "_population_id", resolution.sample_key),
        "population_by_donor": build_crosstab(obs, "_population_id", resolution.donor_key),
        "population_by_condition": build_crosstab(obs, "_population_id", resolution.condition_key),
        "population_by_cluster": build_crosstab(obs, "_population_id", resolution.cluster_key),
    }
    return {key: value for key, value in tables.items() if value is not None}


def build_population_summary(
    obs: pd.DataFrame,
    resolution: IdentityResolution,
    config: PopulationIdentityConfig,
) -> pd.DataFrame:
    """Build one row per candidate population/state."""

    rows: list[dict[str, object]] = []
    total_cells = max(1, len(obs))
    for population_id, group in obs.groupby("_population_id", observed=False):
        row: dict[str, object] = {
            "population_id": str(population_id),
            "population_key": resolution.candidate_key,
            "population_source": resolution.candidate_source,
            "n_cells": int(len(group)),
            "fraction_cells": float(len(group) / total_cells),
        }
        add_diversity_columns(row, group, resolution.sample_key, "sample")
        add_diversity_columns(row, group, resolution.donor_key, "donor")
        add_diversity_columns(row, group, resolution.condition_key, "condition")
        add_dominant_label_columns(row, group, resolution.cluster_key, "cluster")
        add_dominant_label_columns(row, group, resolution.reference_key, "reference")
        add_dominant_label_columns(row, group, resolution.annotation_key, "annotation")
        add_qc_columns(row, group)
        add_numeric_mean_column(row, group, resolution.confidence_key, "mean_confidence")
        add_numeric_mean_column(row, group, resolution.entropy_key, "mean_entropy")
        status, reasons = classify_population(row, resolution, config)
        row["evidence_status"] = status
        row["evidence_reasons"] = " | ".join(reasons)
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    status_order = {
        "atlas_supported_state": 0,
        "annotation_supported_state": 1,
        "cluster_native_candidate": 2,
        "underpowered_rare_population": 3,
        "donor_restricted_candidate": 4,
        "technical_or_qc_suspect": 5,
        "low_confidence_candidate": 6,
    }
    summary["_status_order"] = summary["evidence_status"].map(status_order).fillna(99)
    summary = summary.sort_values(["_status_order", "n_cells"], ascending=[True, False])
    return summary.drop(columns=["_status_order"]).reset_index(drop=True)


def classify_population(
    row: dict[str, object],
    resolution: IdentityResolution,
    config: PopulationIdentityConfig,
) -> tuple[str, list[str]]:
    """Classify a candidate population based on available evidence."""

    reasons: list[str] = []
    n_cells = int(row.get("n_cells", 0))
    n_samples = int(row.get("n_samples", 0) or 0)
    n_donors = int(row.get("n_donors", 0) or 0)
    dominant_donor_fraction = float(row.get("dominant_donor_fraction", 0.0) or 0.0)
    qc_fail_fraction = float(row.get("qc_fail_fraction", 0.0) or 0.0)
    doublet_fraction = float(row.get("doublet_fraction", 0.0) or 0.0)
    mean_confidence = row.get("mean_confidence")
    mean_entropy = row.get("mean_entropy")

    if (
        qc_fail_fraction > config.max_qc_fail_fraction
        or doublet_fraction > config.max_doublet_fraction
    ):
        reasons.append("QC/doublet enrichment exceeds configured threshold.")
        return "technical_or_qc_suspect", reasons
    if n_cells < config.min_cells or (n_samples > 0 and n_samples < config.min_samples):
        reasons.append("Population is rare or observed in too few samples.")
        return "underpowered_rare_population", reasons
    if n_donors and (
        n_donors < config.min_donors or dominant_donor_fraction > config.max_dominant_donor_fraction
    ):
        reasons.append("Population is donor-restricted or poorly replicated.")
        return "donor_restricted_candidate", reasons
    if mean_confidence is not None and not pd.isna(mean_confidence):
        if float(mean_confidence) < config.min_confidence:
            reasons.append("Mean annotation/reference confidence is low.")
            return "low_confidence_candidate", reasons
    if (
        config.max_entropy is not None
        and mean_entropy is not None
        and not pd.isna(mean_entropy)
        and float(mean_entropy) > config.max_entropy
    ):
        reasons.append("Mean annotation/reference entropy exceeds configured threshold.")
        return "low_confidence_candidate", reasons

    if resolution.candidate_source == "reference":
        reasons.append("Candidate is supported by transferred atlas/reference labels.")
        return "atlas_supported_state", reasons
    if resolution.candidate_source == "annotation":
        reasons.append("Candidate is supported by dataset annotation labels.")
        return "annotation_supported_state", reasons
    reasons.append("Candidate is derived from dataset-native clustering.")
    return "cluster_native_candidate", reasons


def add_diversity_columns(
    row: dict[str, object],
    group: pd.DataFrame,
    column: str | None,
    prefix: str,
) -> None:
    """Add cardinality and dominant-label columns for sample/donor/condition."""

    if column is None or column not in group.columns:
        row[f"n_{prefix}s"] = 0
        row[f"dominant_{prefix}"] = None
        row[f"dominant_{prefix}_fraction"] = None
        return
    counts = group[column].astype(str).value_counts(dropna=False)
    row[f"n_{prefix}s"] = int(len(counts))
    row[f"dominant_{prefix}"] = str(counts.index[0]) if len(counts) else None
    row[f"dominant_{prefix}_fraction"] = float(counts.iloc[0] / len(group)) if len(counts) else None


def add_dominant_label_columns(
    row: dict[str, object],
    group: pd.DataFrame,
    column: str | None,
    prefix: str,
) -> None:
    """Add dominant label and purity columns for cluster/reference/annotation."""

    if column is None or column not in group.columns:
        row[f"dominant_{prefix}"] = None
        row[f"dominant_{prefix}_fraction"] = None
        return
    counts = group[column].astype(str).value_counts(dropna=False)
    row[f"dominant_{prefix}"] = str(counts.index[0]) if len(counts) else None
    row[f"dominant_{prefix}_fraction"] = float(counts.iloc[0] / len(group)) if len(counts) else None


def add_qc_columns(row: dict[str, object], group: pd.DataFrame) -> None:
    """Add QC/doublet enrichment evidence columns when available."""

    if "cellquorum_qc_keep" in group.columns:
        keep = group["cellquorum_qc_keep"].fillna(True).astype(bool)
        row["qc_fail_fraction"] = float((~keep).mean())
    else:
        row["qc_fail_fraction"] = 0.0
    if "predicted_doublet" in group.columns:
        row["doublet_fraction"] = float(
            group["predicted_doublet"].fillna(False).astype(bool).mean()
        )
    else:
        row["doublet_fraction"] = 0.0


def add_numeric_mean_column(
    row: dict[str, object],
    group: pd.DataFrame,
    column: str | None,
    output_key: str,
) -> None:
    """Add a numeric mean column if present."""

    if column is None or column not in group.columns:
        row[output_key] = None
        return
    values = pd.to_numeric(group[column], errors="coerce")
    row[output_key] = None if values.dropna().empty else float(values.mean())


def build_crosstab(obs: pd.DataFrame, row_key: str, column_key: str | None) -> pd.DataFrame | None:
    """Build a long-form population-by-metadata count/fraction table."""

    if column_key is None or column_key not in obs.columns:
        return None
    table = pd.crosstab(obs[row_key].astype(str), obs[column_key].astype(str))
    long = table.reset_index().melt(
        id_vars=row_key,
        var_name=column_key,
        value_name="n_cells",
    )
    long = long.rename(columns={row_key: "population_id"})
    totals = long.groupby("population_id")["n_cells"].transform("sum").replace(0, np.nan)
    long["fraction_within_population"] = (long["n_cells"] / totals).fillna(0.0)
    return long.sort_values(["population_id", "n_cells"], ascending=[True, False]).reset_index(
        drop=True
    )


def build_population_identity_audit(
    adata: ad.AnnData,
    resolution: IdentityResolution,
    tables: dict[str, pd.DataFrame],
    config: PopulationIdentityConfig,
) -> dict[str, Any]:
    """Build structured audit metadata for the stage."""

    skipped_panels: list[dict[str, str]] = []
    if resolution.reference_key is None:
        skipped_panels.append(
            {
                "panel": "reference_label",
                "reason": "No atlas/reference label column was present.",
            }
        )
    if resolution.annotation_key is None:
        skipped_panels.append(
            {
                "panel": "annotation_label",
                "reason": "No annotation label column was present.",
            }
        )
    if resolution.embedding_key is None:
        skipped_panels.append(
            {
                "panel": "embedding_plots",
                "reason": "No configured 2D embedding was present in adata.obsm.",
            }
        )

    warnings: list[str] = []
    if resolution.candidate_source == "cluster":
        warnings.append(
            "No atlas/reference or annotation labels were available; populations are "
            "dataset-native cluster candidates, not named cell types."
        )

    return {
        "schema_version": 1,
        "stage": "population_identity",
        "n_cells": int(adata.n_obs),
        "n_populations": int(len(tables["population_summary"])),
        "resolution": resolution.__dict__,
        "thresholds": {
            "min_cells": config.min_cells,
            "min_samples": config.min_samples,
            "min_donors": config.min_donors,
            "max_dominant_donor_fraction": config.max_dominant_donor_fraction,
            "max_qc_fail_fraction": config.max_qc_fail_fraction,
            "max_doublet_fraction": config.max_doublet_fraction,
            "min_confidence": config.min_confidence,
            "max_entropy": config.max_entropy,
        },
        "skipped_panels": skipped_panels,
        "warnings": warnings,
    }


def write_population_identity_artifacts(
    *,
    manager: ArtifactManager,
    adata: ad.AnnData,
    resolution: IdentityResolution,
    tables: dict[str, pd.DataFrame],
    audit: dict[str, Any],
    config: PopulationIdentityConfig,
    case: str | None = None,
    control: str | None = None,
) -> list[StageArtifact]:
    """Write population-identity tables, metadata, evidence, and plots."""

    artifacts: list[StageArtifact] = []
    base = Path("results") / config.output_dir
    table_dir = base / "tables"
    plot_dir = base / "plots"

    table_descriptions = {
        "population_summary": "Population/state candidate identity evidence summary.",
        "population_by_sample": "Population composition by sample.",
        "population_by_donor": "Population composition by donor.",
        "population_by_condition": "Population composition by condition.",
        "population_by_cluster": "Population composition by cluster.",
    }
    for table_name, table in tables.items():
        artifacts.append(
            manager.write_dataframe(
                table,
                name=f"population_identity_{table_name}",
                relative_path=table_dir / f"{table_name}.csv",
                description=table_descriptions.get(table_name, table_name),
            )
        )

    artifacts.append(
        manager.write_json(
            audit,
            name="population_identity_audit",
            relative_path=base / "audit.json",
            description="Structured audit metadata for population identity evidence.",
        )
    )
    artifacts.append(
        manager.write_markdown(
            build_population_identity_markdown(tables["population_summary"], audit),
            name="population_identity_evidence",
            relative_path=base / "evidence.md",
            description="Human-readable population identity evidence summary.",
        )
    )

    if config.write_figures:
        artifacts.extend(
            write_population_identity_figures(
                manager=manager,
                adata=adata,
                resolution=resolution,
                tables=tables,
                plot_dir=plot_dir,
                case=case,
                control=control,
            )
        )
    return artifacts


def write_population_identity_figures(
    *,
    manager: ArtifactManager,
    adata: ad.AnnData,
    resolution: IdentityResolution,
    tables: dict[str, pd.DataFrame],
    plot_dir: Path,
    case: str | None = None,
    control: str | None = None,
) -> list[StageArtifact]:
    """Write optional publication-style plots when required data exist."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    set_publication_style(dpi=400, small=True)
    artifacts: list[StageArtifact] = []

    summary = tables["population_summary"]
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(4.2, max(2.4, 0.18 * len(summary) + 1.2)))
        plot_summary = summary.sort_values("n_cells", ascending=True)
        colors = plot_summary["evidence_status"].map(status_color).fillna("#8C8C8C")
        ax.barh(plot_summary["population_id"], plot_summary["n_cells"], color=colors)
        ax.set_xlabel("Cells")
        ax.set_ylabel("")
        ax.set_title("Population/state candidate sizes")
        clean_axis(ax, grid=True)
        artifacts.extend(register_saved_figure(manager, fig, plot_dir, "population_sizes"))
        plt.close(fig)

    if resolution.embedding_key is not None:
        artifacts.extend(
            write_embedding_panel(
                manager,
                adata,
                plot_dir,
                group_key=resolution.candidate_key,
                stem="embedding_by_population",
                title="Population identity",
                basis=resolution.embedding_key,
                case=case,
                control=control,
            )
        )
        for key, stem, title in [
            (resolution.cluster_key, "embedding_by_cluster", "Native clusters"),
            (resolution.condition_key, "embedding_by_condition", "Condition"),
            (resolution.reference_key, "embedding_by_reference", "Reference labels"),
            (resolution.annotation_key, "embedding_by_annotation", "Annotation labels"),
        ]:
            if key is None or key == resolution.candidate_key:
                continue
            artifacts.extend(
                write_embedding_panel(
                    manager,
                    adata,
                    plot_dir,
                    group_key=key,
                    stem=stem,
                    title=title,
                    basis=resolution.embedding_key,
                    case=case,
                    control=control,
                )
            )

    condition_table = tables.get("population_by_condition")
    if condition_table is not None and not condition_table.empty:
        pivot = condition_table.pivot_table(
            index="population_id",
            columns=resolution.condition_key,
            values="fraction_within_population",
            fill_value=0.0,
        )
        fig, ax = plt.subplots(figsize=(4.8, max(2.4, 0.18 * len(pivot) + 1.1)))
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Fraction within population"},
        )
        ax.set_xlabel("Condition")
        ax.set_ylabel("")
        ax.set_title("Population composition by condition")
        artifacts.extend(register_saved_figure(manager, fig, plot_dir, "condition_composition"))
        plt.close(fig)

    if resolution.entropy_key is not None:
        values = pd.to_numeric(adata.obs[resolution.entropy_key], errors="coerce").dropna()
        if not values.empty:
            fig, ax = plt.subplots(figsize=(3.4, 2.8))
            ax.hist(values, bins=40, color="#AEB7BE", edgecolor="white", linewidth=0.2)
            ax.set_xlabel(resolution.entropy_key)
            ax.set_ylabel("Cells")
            ax.set_title("Annotation/reference uncertainty")
            clean_axis(ax, grid=True)
            artifacts.extend(register_saved_figure(manager, fig, plot_dir, "uncertainty_entropy"))
            plt.close(fig)

    return artifacts


def write_embedding_panel(
    manager: ArtifactManager,
    adata: ad.AnnData,
    plot_dir: Path,
    *,
    group_key: str,
    stem: str,
    title: str,
    basis: str,
    case: str | None = None,
    control: str | None = None,
) -> list[StageArtifact]:
    """Write one categorical embedding panel as PNG and PDF."""

    if group_key not in adata.obs.columns:
        return []
    values = adata.obs[group_key].astype(str)
    categories = values.value_counts().index.tolist()
    if group_key == "condition":
        others = [c for c in categories if c not in {case, control}]
        palette = condition_palette(case, control, others=others)
    else:
        palette = None
    fig = categorical_embedding(
        adata,
        group_key,
        basis=basis,
        title=title,
        order=categories,
        palette=palette,
        point_size=1.2,
        label_on_plot=len(categories) <= 18,
        legend=len(categories) > 18,
        axis_labels=embedding_axis_labels(basis),
        clip_pct=0.5 if adata.n_obs > 1000 else None,
    )
    artifacts = register_saved_figure(manager, fig, plot_dir, stem)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return artifacts


def register_saved_figure(
    manager: ArtifactManager,
    fig: Any,
    plot_dir: Path,
    stem: str,
) -> list[StageArtifact]:
    """Save PNG/PDF copies and register them as stage artifacts."""

    artifacts: list[StageArtifact] = []
    for suffix in ("png", "pdf"):
        relative_path = plot_dir / f"{stem}.{suffix}"
        save_publication_figure(
            fig,
            manager.resolve_path(relative_path),
            dpi=400,
            tight=True,
        )
        artifacts.append(
            manager.register(
                name=f"population_identity_{stem}_{suffix}",
                relative_path=relative_path,
                kind="figure",
                description=f"Population identity figure: {stem}.",
            )
        )
    return artifacts


def build_population_identity_markdown(summary: pd.DataFrame, audit: dict[str, Any]) -> str:
    """Build human-readable evidence summary."""

    lines = ["# Population Identity Evidence", ""]
    resolution = audit["resolution"]
    lines.extend(
        [
            "## Identity resolution",
            "",
            f"- Candidate key: `{resolution['candidate_key']}`",
            f"- Candidate source: `{resolution['candidate_source']}`",
            f"- Reference key: `{resolution['reference_key']}`",
            f"- Annotation key: `{resolution['annotation_key']}`",
            f"- Cluster key: `{resolution['cluster_key']}`",
            "",
        ]
    )

    if audit["warnings"]:
        lines.extend(["## Warnings", ""])
        for warning in audit["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    if audit["skipped_panels"]:
        lines.extend(["## Skipped optional evidence", ""])
        for item in audit["skipped_panels"]:
            lines.append(f"- {item['panel']}: {item['reason']}")
        lines.append("")

    if summary.empty:
        lines.append("No population candidates were summarized.")
        return "\n".join(lines) + "\n"

    lines.extend(["## Evidence status counts", ""])
    for status, count in summary["evidence_status"].value_counts().items():
        lines.append(f"- {status}: {int(count)}")
    lines.extend(["", "## Population candidates", ""])
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.population_id}: {row.evidence_status}; "
            f"n={int(row.n_cells)} cells; {row.evidence_reasons}"
        )
    return "\n".join(lines) + "\n"


def status_color(status: str) -> str:
    """Return colors for evidence status classes."""

    return {
        "atlas_supported_state": "#4C72B0",
        "annotation_supported_state": "#55A868",
        "cluster_native_candidate": "#8172B3",
        "underpowered_rare_population": "#CCB974",
        "donor_restricted_candidate": "#DD8452",
        "technical_or_qc_suspect": "#C44E52",
        "low_confidence_candidate": "#8C8C8C",
    }.get(str(status), "#8C8C8C")


def embedding_axis_labels(basis: str) -> tuple[str, str]:
    """Return readable axis labels for a 2D embedding key."""

    lower = basis.lower()
    if "umap" in lower:
        return "UMAP1", "UMAP2"
    if "phate" in lower:
        return "PHATE1", "PHATE2"
    if "pca" in lower:
        return "PC1", "PC2"
    return "Dim1", "Dim2"


def first_present_column(obs: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first candidate column that exists and has at least one value."""

    for candidate in candidates:
        if candidate in obs.columns and obs[candidate].notna().any():
            return candidate
    return None


def first_present_numeric_column(obs: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first candidate numeric-ish column with at least one finite value."""

    for candidate in candidates:
        if candidate not in obs.columns:
            continue
        values = pd.to_numeric(obs[candidate], errors="coerce")
        if np.isfinite(values).any():
            return candidate
    return None


def first_present_embedding(adata: ad.AnnData, candidates: list[str]) -> str | None:
    """Return first configured embedding with at least two dimensions."""

    for candidate in candidates:
        if candidate not in adata.obsm:
            continue
        values = np.asarray(adata.obsm[candidate])
        if values.ndim == 2 and values.shape[1] >= 2:
            return candidate
    return None


__all__ = [
    "IdentityResolution",
    "PopulationIdentityStage",
    "build_population_identity_tables",
    "resolve_identity_columns",
    "resolve_population_identity_config",
]
