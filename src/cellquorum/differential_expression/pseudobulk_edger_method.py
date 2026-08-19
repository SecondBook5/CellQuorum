"""Pseudobulk edgeR differential-expression method (R)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellquorum.config.design import DesignConfig, validate_design_against_obs
from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.differential_expression.pseudobulk import aggregate_pseudobulk
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod

# Path to the bundled edgeR script.
_EDGER_R = Path(__file__).parent.parent / "backends" / "r_scripts" / "edger.R"


class PseudobulkEdgeRMethod(RAnalysisMethod):
    """Donor-blocked pseudobulk DE via edgeR quasi-likelihood.

    Pseudobulk is the primary DE test (spec §6). Aggregates cells to donor x
    condition counts, fits ``~ [covariates +] donor + condition`` (paired) or
    ``~ [covariates +] condition`` (unpaired) in edgeR, and returns the DE table.
    """

    name = "pseudobulk_edger"
    stage_category = "differential_expression"
    r_package = "edgeR"

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

    def requires_obs(self, config: dict) -> list[str]:
        """Return the design obs columns that must exist for DE to run."""

        # Read the design columns from config.
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        covariates = list(config.get("covariates", []))

        # Require all design columns to exist.
        return [condition_col, donor_col, *covariates]

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

        # A comparison needs both case and control labels.
        if not case or not control:
            return MethodSkip(
                reason="pseudobulk_edger skipped: case/control labels not set in config",
                details={"method": self.name},
            )

        # Pre-flight estimability gate. This HALTS LOUDLY on a comparison that
        # cannot yield valid statistics — a missing condition arm, or an arm
        # without independent donor replication — the class of silent-wrong DE
        # that the inline paired-handling below cannot catch (an unreplicated
        # arm produces confident-but-meaningless p-values). Paired completeness
        # and donor-confounding are owned by the graceful auto-promote/restrict
        # logic further down, so validate with paired=False here: this runs only
        # the existence + per-arm replication checks and never pre-empts that
        # recovery. The per-arm floor is config-overridable so a deliberate,
        # documented pilot design can lower it rather than tripping a silent cap.
        min_donors_per_arm = int(config.get("min_donors_per_arm", 2))
        validate_design_against_obs(
            adata.obs,
            design=DesignConfig(
                donor_col=donor_col,
                condition_col=condition_col,
                case=case,
                control=control,
                paired=False,
            ),
            min_donors_per_arm=min_donors_per_arm,
        )

        # Rscript + backend + package guards (hoisted to RAnalysisMethod).
        backend, skip = self._resolve_rscript_backend(context, config)
        if skip is not None:
            return skip

        # Aggregate to donor x condition pseudobulk counts.
        pb = aggregate_pseudobulk(
            adata,
            layer=layer,
            donor_col=donor_col,
            condition_col=condition_col,
            extra_obs=covariates,
        )

        # Inspect donor pairing in the pseudobulk meta so a matched design is
        # never analysed unpaired by accident. This is the safety net for the
        # class of error where `design.paired` is left False on data where every
        # donor contributes both a case and a control sample: blocking on donor
        # removes inter-patient baseline variance and is strictly more powerful.
        sample_meta = pb.sample_meta
        counts_df = pb.counts
        case_donors = set(sample_meta.loc[sample_meta[condition_col] == case, donor_col])
        control_donors = set(sample_meta.loc[sample_meta[condition_col] == control, donor_col])
        all_donors = case_donors | control_donors
        complete_pairs = case_donors & control_donors
        design_notes: list[str] = []

        # Auto-promote to paired when the design is fully matched but was not
        # declared paired — the corrected, higher-powered default for such data.
        if not paired and complete_pairs and complete_pairs == all_donors and len(all_donors) >= 2:
            paired = True
            design_notes.append(
                "Auto-promoted to PAIRED: every donor contributes both a "
                f"{case} and a {control} pseudobulk sample "
                f"({len(complete_pairs)} complete donor pairs). Blocking on donor."
            )

        # For a paired fit, keep only donors with a complete pair so the
        # donor-blocked design stays estimable (mixed/rare cell types can lose a
        # donor from one arm); log what was dropped rather than failing silently.
        if paired:
            incomplete = all_donors - complete_pairs
            if incomplete:
                keep = sample_meta.index[sample_meta[donor_col].isin(complete_pairs)]
                counts_df = counts_df.loc[keep]
                sample_meta = sample_meta.loc[keep]
                design_notes.append(
                    f"Paired fit restricted to {len(complete_pairs)} complete donor "
                    f"pairs; dropped {len(incomplete)} donor(s) present in only one "
                    f"arm: {sorted(str(d) for d in incomplete)}."
                )

        # Write pseudobulk inputs to scratch.
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        counts_csv = scratch / "pb_counts.csv"
        meta_csv = scratch / "pb_meta.csv"
        counts_df.reset_index(names="sample").to_csv(counts_csv, index=False)
        # Rename design cols so the R script's fixed names (condition/donor) apply.
        meta = sample_meta.rename(columns={condition_col: "condition", donor_col: "donor"})
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
            notes=[
                f"Pseudobulk edgeR DE: {case} vs {control}, design ~ {design_rhs}.",
                *design_notes,
            ],
            metrics={
                "case": case,
                "control": control,
                "paired": paired,
                "design_rhs": design_rhs,
                "covariates": covariates,
                "n_complete_donor_pairs": len(complete_pairs),
                "design_notes": design_notes,
                "n_pseudosamples": int(counts_df.shape[0]),
                "n_genes_input": int(counts_df.shape[1]),
            },
            backend="rscript",
        )


__all__ = ["PseudobulkEdgeRMethod"]
