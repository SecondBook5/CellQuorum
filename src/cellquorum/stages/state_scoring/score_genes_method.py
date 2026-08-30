"""State scoring via scanpy ``score_genes`` (per-cell program scores → obs)."""

from __future__ import annotations

import anndata as ad
import scanpy as sc

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.state_scoring.programs import resolve_programs


class ScoreGenesMethod(AnalysisMethod):
    """Score each program with ``sc.tl.score_genes``, writing one obs column each.

    A program is eligible only when at least ``min_program_genes`` of its genes
    are present in the data; ineligible programs are recorded as skipped rather
    than scored on a handful of genes. Scores land in
    ``adata.obs[f"{key_prefix}{program}"]``, and a per-cell-type mean-score table
    is written to the results directory.
    """

    name = "score_genes"
    stage_category = "state_scoring"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        layer = config.get("layer", "cellquorum_normalized")
        return DataContract(
            required_layers=[layer] if layer != "X" else [],
            expression_layer=layer,
            expected_kind="lognorm",
        )

    def requires_layers(self) -> list[str]:
        # Config-less base hook; guard the default lognorm layer. The contract
        # re-checks the configured layer (and its lognorm tag) before _run.
        return ["cellquorum_normalized"]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        layer = config.get("layer", "cellquorum_normalized")
        cell_type_col = config.get("cell_type_col", "cell_type")
        key_prefix = config.get("key_prefix", "state_")
        min_genes = int(config.get("min_program_genes", 3))
        random_state = int(config.get("random_state", 0))

        programs = resolve_programs(config, context)
        if not programs:
            return self._skip("no programs configured", n_programs=0)

        # Score on a working copy whose .X is the lognorm layer so score_genes
        # reads it; write the resulting scores back onto the real object's obs.
        scored = adata.copy()
        scored.X = scored.layers[layer] if layer != "X" and layer in scored.layers else scored.X

        done: list[dict] = []
        skipped: list[dict] = []
        score_cols: dict[str, str] = {}
        for program, genes in programs.items():
            present = [g for g in genes if g in scored.var_names]
            if len(present) < min_genes:
                skipped.append(
                    {"program": program, "n_present": len(present), "n_genes": len(genes)}
                )
                continue
            col = f"{key_prefix}{program}"
            sc.tl.score_genes(scored, present, score_name=col, random_state=random_state)
            adata.obs[col] = scored.obs[col].to_numpy()
            score_cols[program] = col
            done.append({"program": program, "n_present": len(present), "n_genes": len(genes)})

        if not done:
            return self._skip(
                "no program met the present-gene gate",
                min_program_genes=min_genes,
                skipped=skipped,
            )

        # Per-cell-type mean of each program score (when a label column exists).
        artifacts = []
        writer = StageArtifactWriter.from_context(context)
        if cell_type_col in adata.obs.columns:
            frame = adata.obs[[cell_type_col, *score_cols.values()]].copy()
            frame[cell_type_col] = frame[cell_type_col].astype(str)
            per_ct = frame.groupby(cell_type_col).mean()
            per_ct.columns = list(score_cols.keys())
            long = per_ct.reset_index().melt(
                id_vars=cell_type_col, var_name="program", value_name="mean_score"
            )
            long = long.rename(columns={cell_type_col: "cell_type"})
            long = long[["cell_type", "program", "mean_score"]]
            artifacts.append(
                writer.table(
                    long,
                    "state_scoring_score_genes_by_celltype.csv",
                    name="state_scoring_results",
                    description="Per-cell-type mean program score (scanpy score_genes).",
                    index=False,
                )
            )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"score_genes scored {len(done)} program(s) → obs[{key_prefix}*]."],
            metrics={
                "method": self.name,
                "n_programs": len(done),
                "programs": [d["program"] for d in done],
                "scored": done,
                "skipped": skipped,
                "key_prefix": key_prefix,
            },
            backend="python",
        )


__all__ = ["ScoreGenesMethod"]
