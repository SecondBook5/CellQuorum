"""Pipeline stage that reconciles per-method label columns into one label."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from cellquorum.annotation.consensus.config import AnnotationConsensusConfig
from cellquorum.annotation.consensus.consensus import normalize_label, reconcile_votes
from cellquorum.core.artifacts import ArtifactManager
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage


@register_stage(
    name="annotation_consensus",
    order=130,
    config_flag="annotation_consensus",
    config_field="annotation_consensus",
)
class AnnotationConsensusStage:
    """Reconcile several obs label columns into a consensus label + confidence."""

    def run(self, context: object) -> StageResult:
        """
        Execute the annotation-consensus stage.

        Args:
            context: Pipeline context with AnnData, config, and output paths.

        Returns:
            StageResult with consensus obs columns added, or a recorded skip.
        """

        adata = context.require_adata()
        config = _resolve_config(context)

        # Honor the enabled flag.
        if not config.enabled:
            return StageResult.skipped(
                adata=adata,
                reason="disabled by config",
                warnings=["annotation_consensus disabled by config"],
            )

        # Keep only configured label columns that actually exist (a method may
        # have skipped, leaving no column). Skip loudly if none are present.
        present_keys = [k for k in config.method_label_keys if k in adata.obs.columns]
        if not present_keys:
            return StageResult.skipped(
                adata=adata,
                reason=f"no configured label columns present: {config.method_label_keys}",
                warnings=[
                    "annotation_consensus skipped: none of "
                    f"{config.method_label_keys} found in obs"
                ],
                metrics={"method_label_keys": config.method_label_keys},
            )

        # Build the normalized vote matrix (n_cells x n_methods).
        vote_frame = pd.DataFrame(index=adata.obs_names)
        for key in present_keys:
            vote_frame[key] = [
                normalize_label(v, config.backbone_aliases) for v in adata.obs[key].tolist()
            ]

        # Reconcile each cell.
        labels: list[str | None] = []
        tiers: list[str] = []
        review: list[bool] = []
        for _, row in vote_frame.iterrows():
            label, tier, needs = reconcile_votes(
                list(row.values),
                min_agree_fraction=config.min_agree_fraction,
                high_confidence_all=config.high_confidence_all,
            )
            labels.append(label)
            tiers.append(tier)
            review.append(needs)

        # Write consensus columns onto obs.
        adata.obs[config.key_added] = pd.Categorical(labels)
        adata.obs[config.confidence_key] = pd.Categorical(
            tiers, categories=["high", "medium", "low"]
        )
        adata.obs[config.needs_review_key] = review

        # Optionally copy a granular label for high-confidence cells.
        if config.granular_source_key and config.granular_source_key in adata.obs.columns:
            granular_src = adata.obs[config.granular_source_key].astype("object")
            granular = [
                granular_src.iloc[i] if tiers[i] == "high" else None for i in range(adata.n_obs)
            ]
            adata.obs["cell_type_granular"] = pd.Categorical(granular)

        # Build an agreement summary (tier counts + label counts).
        tier_counts = pd.Series(tiers, name="count").value_counts()
        agreement_df = (
            pd.Series(labels, name="cell_type")
            .value_counts(dropna=False)
            .rename_axis("cell_type")
            .reset_index(name="n_cells")
        )

        payload = {
            "stage": self.name,
            "method_label_keys": present_keys,
            "n_cells": int(adata.n_obs),
            "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
            "n_needs_review": int(sum(review)),
            "label_counts": {
                str(r.cell_type): int(r.n_cells) for r in agreement_df.itertuples(index=False)
            },
        }
        adata.uns.setdefault("cellquorum", {})["annotation_consensus"] = payload

        # Write artifacts.
        manager = ArtifactManager.from_root(context.paths.root)
        artifacts = [
            manager.write_dataframe(
                agreement_df,
                name="annotation_consensus_agreement",
                relative_path="results/annotation_consensus_agreement.csv",
                description="Consensus label counts across cells.",
                index=False,
            ),
            manager.write_markdown(
                _build_summary_markdown(payload),
                name="annotation_consensus_summary",
                relative_path="reports/annotation_consensus_summary.md",
                description="Human-readable annotation-consensus summary.",
            ),
        ]

        # Fail-loud: the consensus label column must exist after a real run.
        DataContract(required_obs=[config.key_added]).validate(adata)

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[
                f"annotation_consensus reconciled {len(present_keys)} method(s) -> "
                f"{config.key_added}; {payload['n_needs_review']} cell(s) need review."
            ],
            warnings=[],
            metrics={
                "n_methods": len(present_keys),
                "tier_counts": payload["tier_counts"],
                "n_needs_review": payload["n_needs_review"],
                "key_added": config.key_added,
            },
        )


def _resolve_config(context: object) -> AnnotationConsensusConfig:
    """
    Resolve the annotation_consensus config from a PipelineContext-like object.

    Args:
        context: Context carrying config as an object or mapping.

    Returns:
        Validated AnnotationConsensusConfig.
    """

    context_config = getattr(context, "config", None)
    if isinstance(context_config, AnnotationConsensusConfig):
        return context_config
    if isinstance(context_config, Mapping):
        return AnnotationConsensusConfig(**dict(context_config.get("annotation_consensus", {})))
    subconfig = getattr(context_config, "annotation_consensus", None)
    if isinstance(subconfig, AnnotationConsensusConfig):
        return subconfig
    if hasattr(subconfig, "model_dump"):
        return AnnotationConsensusConfig(**subconfig.model_dump())
    if isinstance(subconfig, Mapping):
        return AnnotationConsensusConfig(**dict(subconfig))
    return AnnotationConsensusConfig()


def _build_summary_markdown(payload: dict) -> str:
    """
    Build a concise Markdown summary of the consensus result.

    Args:
        payload: The stage payload dict.

    Returns:
        Markdown report fragment.
    """

    lines = ["# Annotation Consensus Summary", ""]
    lines.append(f"- Methods reconciled: {', '.join(payload['method_label_keys'])}")
    lines.append(f"- Cells: {payload['n_cells']}")
    lines.append(f"- Needs review: {payload['n_needs_review']}")
    lines.extend(["", "## Confidence tiers", ""])
    for tier, count in payload["tier_counts"].items():
        lines.append(f"- {tier}: {count}")
    lines.extend(["", "## Label counts", ""])
    for label, count in payload["label_counts"].items():
        lines.append(f"- {label}: {count}")
    return "\n".join(lines) + "\n"


__all__ = ["AnnotationConsensusStage"]
