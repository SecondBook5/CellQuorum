"""State scoring via decoupler AUCell (per-cell program AUC → obsm)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import pandas as pd

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.state_scoring.programs import resolve_programs

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


class AucellMethod(AnalysisMethod):
    """Per-cell program enrichment via decoupler AUCell.

    Programs with at least ``min_program_genes`` present genes are assembled into
    one net and scored in a single AUCell call. Per-cell AUCs are written to
    ``adata.obsm["X_state_aucell"]`` (cells × programs, column order recorded in
    ``adata.uns["state_aucell"]``), and a per-cell-type mean-AUC table is written
    to the results directory. Cells decoupler drops (all-zero over the net) are
    backfilled with 0.0 so the obsm matrix stays cell-aligned.

    ``uns["state_aucell"]["genes"]`` records the genes each program was *actually*
    scored on — the manifest's list intersected with ``var_names``. Two downstream
    questions cannot be answered without it. Whether two program scores are
    independent readouts or the same genes read twice is a property of the gene
    lists, and every stage downstream of here sees only the score matrix; and a
    module that lost half its genes to the detection filter is not the module the
    manifest names, which is invisible from the AUC alone.
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

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
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
        scored_genes: dict[str, list[str]] = {}
        for program, genes in programs.items():
            present = [g for g in genes if g in adata.var_names]
            if len(present) < min_genes:
                skipped.append(
                    {"program": program, "n_present": len(present), "n_genes": len(genes)}
                )
                continue
            rows.extend((program, gene) for gene in present)
            eligible.append({"program": program, "n_present": len(present), "n_genes": len(genes)})
            scored_genes[str(program)] = [str(gene) for gene in present]

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

        # Pass the AnnData + layer name straight through rather than building a
        # dense pandas DataFrame from the FULL expression matrix ourselves.
        # decoupler's AnnData-input path handles a sparse layer natively (with
        # internal batching via `bsize`); manually densifying here is the same
        # "12 GB in a minute, VM died" failure mode already fixed once elsewhere
        # in this stage (score_genes_method.py's adata.copy() bug), just much
        # bigger: a real cohort's full matrix (201,871 x 33,417) densified is
        # ~54 GB, dwarfing the handful of program genes actually being scored.
        #
        # decoupler mutates adata.obsm["score_aucell"] in place and returns None
        # when no observations were dropped; when it drops all-zero observations
        # it instead returns a NEW, smaller AnnData carrying the scores and
        # leaves the original `adata` untouched (verified empirically -- this is
        # not documented behavior). So the source of the per-cell AUCs depends
        # on which happened, and the result is realigned to the full obs index
        # with dropped cells backfilled to 0.0 (no expression -> no enrichment)
        # exactly as the dense-DataFrame path did before this fix.
        try:
            result = dc.mt.aucell(
                adata, net, tmin=min_genes, layer=(layer if layer != "X" else None)
            )
        except Exception as exc:
            return self._skip("AUCell failed", error=str(exc)[:300])

        source = adata if result is None else result
        try:
            es = source.obsm.pop("score_aucell")
        except KeyError:
            return self._skip("AUCell produced no program columns")
        aligned = es.reindex(adata.obs_names)

        if aligned.shape[1] == 0:
            return self._skip("AUCell produced no program columns")

        n_dropped = int(aligned.isna().any(axis=1).sum())
        aligned = aligned.fillna(0.0)

        adata.obsm["X_state_aucell"] = aligned.to_numpy(dtype=float)
        adata.uns["state_aucell"] = {
            "programs": list(aligned.columns),
            # Keyed on the surviving columns, not on ``scored_genes`` wholesale: AUCell
            # can drop a program of its own accord, and a gene list for a column that
            # is not in the matrix would let a downstream overlap check describe a
            # program nothing was scored for.
            "genes": {
                str(name): scored_genes[str(name)]
                for name in aligned.columns
                if str(name) in scored_genes
            },
        }

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
