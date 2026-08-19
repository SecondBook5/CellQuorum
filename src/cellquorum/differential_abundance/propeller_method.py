"""Propeller differential-abundance method (R speckle)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import anndata as ad

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.differential_abundance.aggregation import aggregate_celltype_counts
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod

# Path to the bundled propeller script.
_PROPELLER_R = Path(__file__).parent.parent / "backends" / "r_scripts" / "propeller.R"


class PropellerMethod(RAnalysisMethod):
    """Speckle propeller moderated-t proportion test for differential abundance.

    Propeller tests for cell-type proportion differences between conditions using
    transformed proportions and a moderated t-statistic (spec §DA). Aggregates cells
    to sample × cell-type counts, transforms proportions (asin or logit), and fits
    a linear model to detect abundance changes.
    """

    name = "propeller"
    stage_category = "differential_abundance"
    r_package = "speckle"

    def input_contract(self, config: dict) -> DataContract:
        """Require the design obs columns (no layer needed for DA)."""
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col", "cell_type")
        return DataContract(
            required_obs=[condition_col, donor_col, cell_type_col],
        )

    def requires_obs(self, config: dict) -> list[str]:
        """Return the design obs columns that must exist for DA to run."""

        # Read the design columns from config.
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col", "cell_type")

        # Require all design columns to exist.
        return [condition_col, donor_col, cell_type_col]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        """Aggregate cell-type counts, fit propeller, and return the DA table."""

        # Resolve config fields (all schema-driven; no hardcoded study assumptions).
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col", "cell_type")
        case = config.get("case")
        control = config.get("control")
        transform = config.get("transform", "asin")
        timeout = int(config.get("timeout_seconds", 1800))

        # A comparison needs both case and control labels.
        if not case or not control:
            return self._skip("case/control labels not set in config")

        # Rscript + backend + package guards (hoisted to RAnalysisMethod).
        backend, skip = self._resolve_rscript_backend(context)
        if skip is not None:
            return skip

        # Aggregate to sample × cell-type counts.
        cc = aggregate_celltype_counts(
            adata,
            donor_col=donor_col,
            condition_col=condition_col,
            cell_type_col=cell_type_col,
        )

        # Write aggregated inputs to scratch.
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        counts_csv = scratch / "da_counts.csv"
        meta_csv = scratch / "da_meta.csv"

        # counts.csv: first col 'sample', remaining cols = cell types, integer counts.
        cc.counts.reset_index(names="sample").to_csv(counts_csv, index=False)

        # meta.csv: first col = sample id (row index), must contain condition_col column.
        # cc.sample_meta already has columns named by condition_col and donor_col.
        cc.sample_meta.to_csv(meta_csv, index=True)

        # Prepare the output path in the run results directory.
        results_dir = Path(context.paths.results)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_csv = results_dir / "da_propeller.csv"

        # Invoke the propeller script; non-zero exit -> recorded skip (never crash).
        # propeller.R CLI: <counts.csv> <meta.csv> <out.csv> <condition_col> <case>
        # <control> <transform>
        args = [
            str(counts_csv),
            str(meta_csv),
            str(out_csv),
            condition_col,
            case,
            control,
            transform,
        ]
        try:
            proc = backend.run_script(_PROPELLER_R, args, timeout=timeout)
        except FileNotFoundError as exc:
            return self._skip("R execution failed", error=str(exc)[:500])
        except subprocess.TimeoutExpired as exc:
            # A configured timeout must skip this method, not crash the stage
            # and abort the sibling methods still queued after it.
            return self._skip(f"R execution timed out after {timeout}s", error=str(exc)[:500])
        if proc.returncode != 0:
            return self._skip("propeller script failed", stderr=proc.stderr.strip()[:500])

        # Return the DA table as an artifact plus provenance metrics.
        return StageResult(
            adata=adata,
            artifacts=[
                StageArtifact(
                    name="da_results",
                    path=out_csv,
                    kind="csv",
                    description=f"Propeller DA ({case} vs {control}), {transform} transform.",
                )
            ],
            notes=[f"Propeller DA: {case} vs {control}, transform={transform}."],
            metrics={
                "case": case,
                "control": control,
                "transform": transform,
                "n_samples": int(cc.counts.shape[0]),
                "n_celltypes": int(cc.counts.shape[1]),
            },
            backend="rscript",
        )


__all__ = ["PropellerMethod"]
