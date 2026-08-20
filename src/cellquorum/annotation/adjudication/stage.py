"""Pipeline stage for adjudicating cluster/state claims."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from cellquorum.annotation.adjudication.adjudicate import adjudicate_cluster
from cellquorum.annotation.adjudication.config import AdjudicationConfig
from cellquorum.annotation.adjudication.evidence import (
    build_cluster_evidence_table,
    cluster_evidence_to_dataframe,
)
from cellquorum.core.artifacts import ArtifactManager
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage


@register_stage(
    name="adjudication", order=110, config_flag="adjudication", config_field="adjudication"
)
class AdjudicationStage:
    """Build cluster evidence, adjudicate claims, and write audit artifacts."""

    def run(self, context: object) -> StageResult:
        """
        Execute the adjudication stage.

        Args:
            context: Pipeline context with AnnData, config, and output paths.

        Returns:
            StageResult with unchanged AnnData plus adjudication artifacts.
        """

        adata = context.require_adata()
        config = resolve_adjudication_config(context)
        if not config.enabled:
            return StageResult.skipped(
                adata=adata,
                reason="disabled by config",
                warnings=["adjudication disabled by config"],
            )

        donor_key, condition_key = resolve_design_keys(context, config)
        missing = [
            column
            for column in (config.cluster_key, donor_key, condition_key)
            if column not in adata.obs.columns
        ]
        if missing:
            return StageResult.skipped(
                adata=adata,
                reason=f"missing required obs column(s): {missing}",
                warnings=[f"adjudication skipped: missing required obs column(s): {missing}"],
                metrics={
                    "cluster_key": config.cluster_key,
                    "donor_key": donor_key,
                    "condition_key": condition_key,
                    "missing_obs": missing,
                },
            )

        evidence_rows = build_cluster_evidence_table(
            adata,
            config=config,
            donor_key=donor_key,
            condition_key=condition_key,
        )
        results = [adjudicate_cluster(evidence) for evidence in evidence_rows]

        evidence_df = cluster_evidence_to_dataframe(evidence_rows)
        results_df = adjudication_results_to_dataframe(results)

        payload = {
            "stage": self.name,
            "cluster_key": config.cluster_key,
            "donor_key": donor_key,
            "condition_key": condition_key,
            "n_clusters": len(results),
            "results": [result.to_dict() for result in results],
            "evidence": [evidence.to_dict() for evidence in evidence_rows],
        }

        cq = adata.uns.setdefault("cellquorum", {})
        cq["adjudication"] = payload

        manager = ArtifactManager.from_root(context.paths.root)
        artifacts = [
            manager.write_dataframe(
                results_df,
                name="adjudication_results",
                relative_path=f"results/{config.output_prefix}_results.csv",
                description="Cluster/state adjudication results.",
            ),
            manager.write_dataframe(
                evidence_df,
                name="adjudication_evidence",
                relative_path=f"results/{config.output_prefix}_evidence.csv",
                description="Cluster-level evidence used for adjudication.",
            ),
            manager.write_json(
                payload,
                name="adjudication_results_json",
                relative_path=f"results/{config.output_prefix}_results.json",
                description="Structured adjudication results and evidence trail.",
            ),
            manager.write_markdown(
                build_adjudication_markdown(results_df),
                name="adjudication_summary",
                relative_path=f"reports/{config.output_prefix}_summary.md",
                description="Human-readable adjudication summary.",
            ),
        ]

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"adjudicated {len(results)} cluster/state candidate(s)."],
            warnings=[],
            metrics={
                "n_clusters": len(results),
                "cluster_key": config.cluster_key,
                "donor_key": donor_key,
                "condition_key": condition_key,
                "taxonomy_counts": results_df["taxonomy_class"].value_counts().to_dict()
                if not results_df.empty
                else {},
            },
        )


def adjudication_results_to_dataframe(results: list) -> pd.DataFrame:
    """
    Convert adjudication results to a flat table.

    Args:
        results: AdjudicationResult objects.

    Returns:
        DataFrame suitable for CSV output.
    """

    rows = []
    for result in results:
        rows.append(
            {
                "cluster_id": result.cluster_id,
                "taxonomy_class": result.taxonomy_class,
                "confidence": result.confidence,
                "reasons": " | ".join(result.reasons),
                "vetoes": " | ".join(item.name for item in result.vetoes),
            }
        )
    return pd.DataFrame(rows)


def build_adjudication_markdown(results_df: pd.DataFrame) -> str:
    """
    Build a concise Markdown summary of adjudication results.

    Args:
        results_df: Flat adjudication result table.

    Returns:
        Markdown report fragment.
    """

    lines = ["# Adjudication Summary", ""]
    if results_df.empty:
        lines.append("No cluster/state candidates were adjudicated.")
        return "\n".join(lines) + "\n"

    counts = results_df["taxonomy_class"].value_counts()
    lines.extend(["## Taxonomy Counts", ""])
    for taxonomy_class, count in counts.items():
        lines.append(f"- {taxonomy_class}: {int(count)}")

    lines.extend(["", "## Cluster Results", ""])
    for row in results_df.sort_values("cluster_id").itertuples(index=False):
        lines.append(
            f"- {row.cluster_id}: {row.taxonomy_class} " f"(confidence={row.confidence:.3f})"
        )
    return "\n".join(lines) + "\n"


def resolve_adjudication_config(context: object) -> AdjudicationConfig:
    """
    Resolve adjudication config from a PipelineContext-like object.

    Args:
        context: Context with config.

    Returns:
        Validated AdjudicationConfig.
    """

    context_config = getattr(context, "config", None)
    if isinstance(context_config, AdjudicationConfig):
        return context_config
    if isinstance(context_config, Mapping):
        return AdjudicationConfig(**dict(context_config.get("adjudication", {})))
    subconfig = getattr(context_config, "adjudication", None)
    if isinstance(subconfig, AdjudicationConfig):
        return subconfig
    if hasattr(subconfig, "model_dump"):
        return AdjudicationConfig(**subconfig.model_dump())
    if isinstance(subconfig, Mapping):
        return AdjudicationConfig(**dict(subconfig))
    return AdjudicationConfig()


def resolve_design_keys(
    context: object,
    config: AdjudicationConfig,
) -> tuple[str, str]:
    """
    Resolve donor and condition columns for adjudication.

    Args:
        context: Context with top-level design config.
        config: Adjudication config.

    Returns:
        Tuple of donor key and condition key.
    """

    if config.donor_key and config.condition_key:
        return config.donor_key, config.condition_key

    context_config = getattr(context, "config", None)
    design = getattr(context_config, "design", None)
    if isinstance(context_config, Mapping):
        design = context_config.get("design", {})

    donor_key = config.donor_key
    condition_key = config.condition_key
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


__all__ = [
    "AdjudicationStage",
    "adjudication_results_to_dataframe",
    "build_adjudication_markdown",
    "resolve_adjudication_config",
    "resolve_design_keys",
]
