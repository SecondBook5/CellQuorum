"""Milo neighborhood-level differential-abundance method (R miloR)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod

# Path to the bundled milo script.
_MILO_R = Path(__file__).parent.parent.parent / "backends" / "r_scripts" / "milo.R"


class MiloMethod(RAnalysisMethod):
    """Milo neighborhood-level differential-abundance test via miloR.

    Milo tests for local abundance differences at the neighborhood level using
    k-nearest-neighbor graphs over a reduced-dimensionality representation (spec §DA).
    Builds a Milo object, constructs neighborhood graphs, and tests for DA using
    a quasi-likelihood negative-binomial model (via edgeR) on neighborhood counts.
    """

    name = "milo"
    stage_category = "differential_abundance"
    r_package = "miloR"

    def input_contract(self, config: dict) -> DataContract:
        """Require the design obs columns (no layer needed for Milo DA)."""
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        # cell_type_col is optional, so we don't require it in input_contract
        return DataContract(
            required_obs=[condition_col, donor_col],
        )

    def requires_obs(self, config: dict) -> list[str]:
        """Return the design obs columns that must exist for DA to run."""

        # Read the design columns from config.
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        # cell_type_col is optional — do NOT require it (would force-skip cohorts lacking it)

        # Require condition_col and donor_col to exist.
        return [condition_col, donor_col]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        """Build Milo neighborhoods, fit DA model, and return the DA table."""

        # Resolve config fields (all schema-driven; no hardcoded study assumptions).
        use_rep = config.get("use_rep", "X_pca")
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col")  # Optional
        case = config.get("case")
        control = config.get("control")
        k = int(config.get("k", 30))
        prop = float(config.get("prop", 0.1))
        spatial_fdr = float(config.get("spatial_fdr", 0.1))
        paired = bool(config.get("paired", False))
        timeout = int(config.get("timeout_seconds", 1800))

        # A comparison needs both case and control labels.
        if not case or not control:
            return self._skip("case/control labels not set in config")

        # Reduced-dim rep guard: resolve rep with fallback to X_pca.
        rep = use_rep
        if rep not in adata.obsm:
            if "X_pca" in adata.obsm:
                rep = "X_pca"
            else:
                return self._skip("no reduced-dim rep (use_rep/X_pca) in obsm", use_rep=use_rep)

        # Rscript + backend + package guards (hoisted to RAnalysisMethod).
        backend, skip = self._resolve_rscript_backend(context)
        if skip is not None:
            return skip

        # Extract reduced-dim rep (densify if sparse).
        embedding = np.asarray(adata.obsm[rep])

        # Write per-cell inputs to scratch.
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        rep_csv = scratch / "rep.csv"
        meta_csv = scratch / "meta.csv"

        # rep.csv: cells × dims embedding, first column = cell id (row index).
        pd.DataFrame(embedding, index=adata.obs_names).to_csv(rep_csv, index=True)

        # meta.csv: per-cell metadata, first column = cell id (row index).
        # MUST contain columns literally named by condition_col, donor_col, and
        # (if present) cell_type_col.
        meta_cols = [condition_col, donor_col]
        if cell_type_col and cell_type_col in adata.obs.columns:
            meta_cols.append(cell_type_col)
        meta = adata.obs[meta_cols].copy()
        meta.to_csv(meta_csv, index=True)

        # Prepare the output path in the run results directory.
        results_dir = Path(context.paths.results)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_csv = results_dir / "da_milo.csv"

        # Build milo.R CLI args.
        # Rscript milo.R <rep.csv> <meta.csv> <out.csv> <condition_col> <case>
        # <control> <donor_col> <k> <prop> [celltype_col] [paired]
        args = [
            str(rep_csv),
            str(meta_csv),
            str(out_csv),
            condition_col,
            case,
            control,
            donor_col,
            str(k),
            str(prop),
        ]

        # arg10: celltype_col is OPTIONAL and positional.
        # If cell_type_col exists in obs, pass it.
        # If cell_type_col does NOT exist BUT paired=true, pass empty string
        # so paired lands at position 11.
        # If no celltype AND not paired, omit both trailing args.
        has_celltype = cell_type_col and cell_type_col in adata.obs.columns
        if has_celltype:
            args.append(cell_type_col)
        elif paired:
            # No celltype but paired=true: pass empty string at position 10
            args.append("")

        # arg11: paired flag
        if has_celltype or paired:
            args.append("true" if paired else "false")

        # Invoke the milo script; non-zero exit -> recorded skip (never crash).
        try:
            proc = backend.run_script(_MILO_R, args, timeout=timeout)
        except FileNotFoundError as exc:
            return self._skip("R execution failed", error=str(exc)[:500])
        except subprocess.TimeoutExpired as exc:
            # A configured timeout must skip this method, not crash the stage
            # and abort the sibling methods still queued after it.
            return self._skip(f"R execution timed out after {timeout}s", error=str(exc)[:500])
        if proc.returncode != 0:
            return self._skip("milo script failed", stderr=proc.stderr.strip()[:500])

        # Read output CSV to compute metrics (skip-not-crash).
        n_nhoods = None
        n_da = None
        try:
            df = pd.read_csv(out_csv)
            n_nhoods = len(df)
            n_da = int((df["SpatialFDR"] < spatial_fdr).sum())
        except Exception:
            pass  # CSV should exist but don't crash if reading fails

        # Return the DA table as an artifact plus provenance metrics.
        return StageResult(
            adata=adata,
            artifacts=[
                StageArtifact(
                    name="da_results",
                    path=out_csv,
                    kind="csv",
                    description=(
                        f"Milo DA ({case} vs {control}), " f"k={k} prop={prop} paired={paired}."
                    ),
                )
            ],
            notes=[f"Milo DA: {case} vs {control}, k={k}, prop={prop}, paired={paired}."],
            metrics={
                "case": case,
                "control": control,
                "paired": paired,
                "k": k,
                "prop": prop,
                "use_rep": rep,
                "n_nhoods": n_nhoods,
                "n_da": n_da,
            },
            backend="rscript",
        )


__all__ = ["MiloMethod"]
