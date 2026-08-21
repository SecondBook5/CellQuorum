"""State scoring via decoupler AUCell (per-cell program AUC → obsm)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.state_scoring.programs import resolve_programs


class AucellMethod(AnalysisMethod):
    """Per-cell program enrichment via decoupler AUCell.

    Programs with at least ``min_program_genes`` present genes are assembled into
    one net and scored in a single AUCell call. Per-cell AUCs are written to
    ``adata.obsm["X_state_aucell"]`` (cells × programs, column order recorded in
    ``adata.uns["state_aucell"]``), and a per-cell-type mean-AUC table is written
    to the results directory. Cells decoupler drops (all-zero over the net) are
    backfilled with 0.0 so the obsm matrix stays cell-aligned.
    """

    name = "aucell"
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
        min_genes = int(config.get("min_program_genes", 3))

        programs = resolve_programs(config, context)
        if not programs:
            return self._skip("no programs configured", n_programs=0)

        # Eligibility gate: keep only programs with enough present genes, then
        # build one long-format net (source=program, target=gene, weight=1).
        rows: list[tuple[str, str]] = []
        eligible: list[dict] = []
        skipped: list[dict] = []
        for program, genes in programs.items():
            present = [g for g in genes if g in adata.var_names]
            if len(present) < min_genes:
                skipped.append(
                    {"program": program, "n_present": len(present), "n_genes": len(genes)}
                )
                continue
            rows.extend((program, gene) for gene in present)
            eligible.append({"program": program, "n_present": len(present), "n_genes": len(genes)})

        if not eligible:
            return self._skip(
                "no program met the present-gene gate",
                min_program_genes=min_genes,
                skipped=skipped,
            )

        try:
            import decoupler as dc
        except Exception as exc:
            return self._skip("decoupler unavailable", error=str(exc)[:300])
        if dc is None:
            return self._skip("decoupler unavailable")

        net = pd.DataFrame(rows, columns=["source", "target"])
        net["weight"] = 1.0

        matrix = adata.layers[layer] if layer != "X" and layer in adata.layers else adata.X
        dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
        data = pd.DataFrame(dense, index=adata.obs_names, columns=adata.var_names)

        # AUCell can drop all-zero cells, so realign to the full obs index and
        # backfill dropped cells with 0.0 (no expression → no enrichment).
        try:
            es, _ = dc.mt.aucell(data, net, tmin=min_genes)
            es = es.copy()
            aligned = es.reindex(adata.obs_names)
        except Exception as exc:
            return self._skip("AUCell failed", error=str(exc)[:300])

        if aligned.shape[1] == 0:
            return self._skip("AUCell produced no program columns")

        n_dropped = int(aligned.isna().any(axis=1).sum())
        aligned = aligned.fillna(0.0)

        adata.obsm["X_state_aucell"] = aligned.to_numpy(dtype=float)
        adata.uns["state_aucell"] = {"programs": list(aligned.columns)}

        # Per-cell-type mean AUC (when a label column exists).
        artifacts = []
        writer = StageArtifactWriter.from_context(context)
        if cell_type_col in adata.obs.columns:
            summary = aligned.copy()
            summary[cell_type_col] = adata.obs[cell_type_col].astype(str).to_numpy()
            per_ct = summary.groupby(cell_type_col).mean()
            long = per_ct.reset_index().melt(
                id_vars=cell_type_col, var_name="program", value_name="mean_auc"
            )
            long = long.rename(columns={cell_type_col: "cell_type"})
            long = long[["cell_type", "program", "mean_auc"]]
            artifacts.append(
                writer.table(
                    long,
                    "state_scoring_aucell_by_celltype.csv",
                    name="state_scoring_results",
                    description="Per-cell-type mean program AUC (decoupler AUCell).",
                    index=False,
                )
            )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"AUCell scored {len(aligned.columns)} program(s) → obsm['X_state_aucell']."],
            metrics={
                "method": self.name,
                "n_programs": int(aligned.shape[1]),
                "programs": list(aligned.columns),
                "eligible": eligible,
                "skipped": skipped,
                "n_cells_backfilled": n_dropped,
            },
            backend="python",
        )


__all__ = ["AucellMethod"]
