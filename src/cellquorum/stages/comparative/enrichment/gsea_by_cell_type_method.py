"""Preranked GSEA run separately within each cell type (the NES-by-cell-type table)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import pandas as pd

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.comparative.enrichment.gsea_method import filter_net_by_size, gsea_table
from cellquorum.stages.comparative.enrichment.priors import PriorFetchError, get_net
from cellquorum.stages.comparative.enrichment.ranking import de_table_to_ranking

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


class GseaByCellTypeMethod(AnalysisMethod):
    """GSEA of each cell type's own case-vs-control DE, one row per gene set per cell type.

    The ranking is the signed -log10 p of the donor-aware pseudobulk edgeR test run within
    that cell type, the same ranking the pooled GSEA uses, so a cell type is never scored on
    a contrast pooled over cells of other types. Significance is adjusted across gene sets
    within a cell type, because each cell type is its own family of tests; a reader scanning
    one column of the heatmap is asking about one cell type.
    """

    name = "gsea_by_cell_type"
    stage_category = "enrichment"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """Reads the per-cell-type DE CSV, not the matrix."""
        return DataContract()

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        de_name = config.get("de_by_group_filename", "de_pseudobulk_edger_by_subcluster.csv")
        group_col = config.get("de_group_col", "cell_type")
        collections = config.get("gene_set_collections", ["hallmark", "reactome"])
        min_size = int(config.get("min_size", 10))
        max_size = int(config.get("max_size", 500))
        min_genes = int(config.get("min_genes", 1000))
        permutations = int(config.get("gsea_permutations", 1000))
        seed = int(config.get("seed", 42))
        fdr_method = config.get("fdr_method", "fdr_bh")
        fdr = float(config.get("fdr", 0.05))

        results_dir = Path(context.paths.results)
        de_path = results_dir / de_name
        if not de_path.exists():
            return self._skip(
                f"no per-cell-type DE table at {de_path}; set differential_expression."
                "cell_type_col so the DE stage writes one",
                de_path=str(de_path),
            )
        de = pd.read_csv(de_path)
        if group_col not in de.columns:
            return self._skip(f"per-cell-type DE table has no '{group_col}' column")

        rankings: dict[str, pd.DataFrame] = {}
        too_small: list[str] = []
        for group, frame in de.groupby(group_col, sort=False):
            ranking = de_table_to_ranking(frame)
            if ranking.shape[1] < min_genes:
                too_small.append(f"{group} ({ranking.shape[1]} ranked genes)")
                continue
            rankings[str(group)] = ranking
        if not rankings:
            return self._skip("no cell type has enough ranked genes", too_small=too_small)

        writer = StageArtifactWriter.from_context(context)
        artifacts, done, skipped = [], [], []
        for collection in collections:
            try:
                net = get_net(
                    collection,
                    organism=config.get("organism", "human"),
                    gmt_path=config.get("gmt_path"),
                    license=config.get("license", "academic"),
                )
            except PriorFetchError as exc:
                skipped.append({"collection": collection, "reason": str(exc)[:300]})
                continue

            frames = []
            for group, ranking in rankings.items():
                group_net = filter_net_by_size(net, set(ranking.columns), max_size)
                try:
                    table = gsea_table(
                        ranking.iloc[0],
                        group_net,
                        min_size=min_size,
                        permutations=permutations,
                        seed=seed,
                        fdr_method=fdr_method,
                        fdr=fdr,
                        skipped=skipped,
                        collection=collection,
                    )
                except Exception as exc:  # noqa: BLE001 - one cell type must not sink the rest
                    skipped.append(
                        {"collection": collection, "reason": f"{group}: {str(exc)[:200]}"}
                    )
                    continue
                table.insert(0, "cell_type", group)
                frames.append(table)
            if not frames:
                continue
            artifacts.append(
                writer.table(
                    pd.concat(frames, ignore_index=True),
                    f"enrichment_gsea_{collection}_by_cell_type.csv",
                    name="enrichment_results",
                    description=(
                        f"GSEA ({collection}) within each cell type. `score` is the "
                        "normalized ES (NES); padj is across gene sets within a cell type."
                    ),
                    index=False,
                )
            )
            done.append(collection)

        if not done:
            return self._skip("no collection produced results", skipped=skipped)
        notes = [f"GSEA by cell type over {done}: {len(rankings)} cell type(s)."]
        if too_small:
            notes.append("Skipped cell types with too few ranked genes: " + "; ".join(too_small))
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics={
                "method": self.name,
                "collections": done,
                "cell_types": sorted(rankings),
                "skipped_cell_types": too_small,
                "skipped_collections": skipped,
                "seed": seed,
            },
            backend="python",
        )


__all__ = ["GseaByCellTypeMethod"]
