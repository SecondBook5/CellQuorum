"""Pseudobulk edgeR differential-expression method (R)."""

from __future__ import annotations

import shutil
from pathlib import Path

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.differential_expression.pseudobulk import aggregate_pseudobulk
from cellquorum.methods.base import AnalysisMethod, MethodSkip

# Path to the bundled edgeR script.
_EDGER_R = Path(__file__).parent.parent / "backends" / "r_scripts" / "edger.R"


class PseudobulkEdgeRMethod(AnalysisMethod):
    """Donor-blocked pseudobulk DE via edgeR quasi-likelihood.

    Pseudobulk is the primary DE test (spec §6). Aggregates cells to donor x
    condition counts, fits ``~ [covariates +] donor + condition`` (paired) or
    ``~ [covariates +] condition`` (unpaired) in edgeR, and returns the DE table.
    """

    name = "pseudobulk_edger"
    stage_category = "differential_expression"
    backend = "rscript"

    def input_contract(self, config: dict) -> DataContract:
        """Require the raw-counts layer plus the design obs columns."""
        layer = config.get("layer", "counts")
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        covariates = list(config.get("covariates", []))
        return DataContract(
            required_layers=[layer],
            required_obs=[condition_col, donor_col, *covariates],
            expression_layer=layer,
            expected_kind="counts",
        )

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        """Aggregate pseudobulk, fit edgeR, and return the DE table."""

        # Resolve config fields (all schema-driven; no hardcoded study assumptions).
        layer = config.get("layer", "counts")
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        case = config.get("case")
        control = config.get("control")
        covariates = list(config.get("covariates", []))
        paired = bool(config.get("paired", False))
        min_count = int(config.get("min_count", 10))
        min_total_count = int(config.get("min_total_count", 15))
        timeout = int(config.get("timeout_seconds", 1800))
        r_package = config.get("r_package", "edgeR")

        # A comparison needs both case and control labels.
        if not case or not control:
            return MethodSkip(
                reason="pseudobulk_edger skipped: case/control labels not set in config",
                details={"method": self.name},
            )

        # Rscript availability guard (mirrors SoupX/scDiagnostics).
        if shutil.which("Rscript") is None:
            return MethodSkip(
                reason="pseudobulk_edger skipped: Rscript unavailable",
                details={"method": self.name},
            )

        # Resolve the Rscript backend from the context registry.
        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("rscript")
            except Exception:
                backend = None
        if backend is None:
            return MethodSkip(
                reason="pseudobulk_edger skipped: rscript backend unavailable",
                details={"method": self.name},
            )

        # edgeR package guard.
        if not backend._r_package_available(r_package):
            return MethodSkip(
                reason=f"pseudobulk_edger skipped: {r_package} R package unavailable",
                details={"method": self.name, "r_package": r_package},
            )

        # Aggregate to donor x condition pseudobulk counts.
        pb = aggregate_pseudobulk(
            adata,
            layer=layer,
            donor_col=donor_col,
            condition_col=condition_col,
            extra_obs=covariates,
        )

        # Write pseudobulk inputs to scratch.
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        counts_csv = scratch / "pb_counts.csv"
        meta_csv = scratch / "pb_meta.csv"
        pb.counts.reset_index(names="sample").to_csv(counts_csv, index=False)
        # Rename design cols so the R script's fixed names (condition/donor) apply.
        meta = pb.sample_meta.rename(columns={condition_col: "condition", donor_col: "donor"})
        meta.to_csv(meta_csv)

        # Build the design right-hand side: covariates + [donor +] condition.
        rhs_terms = [*covariates]
        if paired:
            rhs_terms.append("donor")
        rhs_terms.append("condition")
        design_rhs = " + ".join(rhs_terms)

        # Prepare the output path in the run results directory.
        results_dir = Path(context.paths.results)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_csv = results_dir / "de_pseudobulk_edger.csv"

        # Invoke the edgeR script; non-zero exit -> recorded skip (never crash).
        args = [
            str(counts_csv),
            str(meta_csv),
            str(out_csv),
            "condition",
            case,
            control,
            design_rhs,
            str(min_count),
            str(min_total_count),
        ]
        try:
            proc = backend.run_script(_EDGER_R, args, timeout=timeout)
        except FileNotFoundError as exc:
            return MethodSkip(
                reason="pseudobulk_edger skipped: R execution failed",
                details={"method": self.name, "error": str(exc)[:500]},
            )
        if proc.returncode != 0:
            return MethodSkip(
                reason="pseudobulk_edger skipped: edgeR script failed",
                details={"method": self.name, "stderr": proc.stderr.strip()[:500]},
            )

        # Return the DE table as an artifact plus provenance metrics.
        return StageResult(
            adata=adata,
            artifacts=[
                StageArtifact(
                    name="de_results",
                    path=out_csv,
                    kind="csv",
                    description=f"Pseudobulk edgeR DE ({case} vs {control}), {design_rhs}.",
                )
            ],
            notes=[f"Pseudobulk edgeR DE: {case} vs {control}, design ~ {design_rhs}."],
            metrics={
                "case": case,
                "control": control,
                "paired": paired,
                "design_rhs": design_rhs,
                "covariates": covariates,
                "n_pseudosamples": int(pb.counts.shape[0]),
                "n_genes_input": int(pb.counts.shape[1]),
            },
            backend="rscript",
        )


__all__ = ["PseudobulkEdgeRMethod"]
