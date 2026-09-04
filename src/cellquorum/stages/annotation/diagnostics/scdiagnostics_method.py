"""scDiagnostics annotation-confidence diagnostic method (R)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.backends.script_paths import r_script_path
from cellquorum.core.contracts import DataContract
from cellquorum.core.exceptions import CellQuorumBackendError
from cellquorum.core.h5ad_io import write_h5ad
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod

if TYPE_CHECKING:
    import pandas as pd

# Path to the bundled scDiagnostics R script.
_SCDIAGNOSTICS_R = r_script_path("scdiagnostics.R")


class ScdiagnosticsMethod(RAnalysisMethod):
    """scDiagnostics annotation-confidence diagnostics (query-only or query+ref).

    Runs scDiagnostics R functions to assess annotation confidence:
    - detectAnomaly + calculateNearestNeighborProbabilities (when reference_h5ad
      provided)
    - Shannon categorization entropy (when soft_scores_obsm provided)

    READ-ONLY: adds scdiag_* obs columns; never modifies cell_type or embeddings.
    """

    name = "scdiagnostics"
    stage_category = "annotation_diagnostics"
    r_package = "scDiagnostics"

    def input_contract(self, config: dict) -> DataContract:
        """
        Return the required input contract.

        Two modes:
        - Reference mode (an R reference object is used): require the
          log-normalized expression layer, the label column, and ``X_pca``.
        - Query-only entropy mode (no reference, soft probabilities present):
          the entropy is computed directly from ``obsm[soft_scores_obsm]``, so
          only that matrix is required — no expression layer or ``X_pca``. This
          keeps the R-free fast path reachable through the standard
          ``AnalysisMethod.run`` contract validation.
        """
        cell_type_col = config.get("cell_type_col", "cell_type")
        expression_layer = config.get("expression_layer", "lognorm")
        reference_h5ad = config.get("reference_h5ad")
        soft_scores_obsm = config.get("soft_scores_obsm")

        # Query-only entropy mode: only the soft-probability matrix is needed.
        if not reference_h5ad and soft_scores_obsm:
            return DataContract(required_obsm=[soft_scores_obsm])

        # Reference mode: full lognorm + label + embedding contract.
        return DataContract(
            required_layers=[expression_layer],
            required_obs=[cell_type_col],
            required_obsm=["X_pca"],
            expression_layer=expression_layer,
            expected_kind="lognorm",
        )

    def _run(
        self,
        adata: ad.AnnData,
        config: dict,
        context: object,
    ) -> StageResult | MethodSkip:
        """Execute scDiagnostics via R; return read-only diagnostics."""

        # Resolve config fields.
        cell_type_col = config.get("cell_type_col", "cell_type")
        expression_layer = config.get("expression_layer", "lognorm")
        reference_h5ad = config.get("reference_h5ad")
        soft_scores_obsm = config.get("soft_scores_obsm")
        pc_subset = config.get("pc_subset", [1, 2, 3, 4, 5])
        n_tree = config.get("n_tree", 500)
        n_neighbor = config.get("n_neighbor", 15)
        timeout = config.get("timeout_seconds", 1800)

        # Resolve scratch directory for temp files.
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)

        # Query-only entropy can be computed directly from soft probabilities.
        # This avoids requiring an R reference object when ScArches already
        # produced calibrated per-label probabilities.
        if not reference_h5ad and soft_scores_obsm and soft_scores_obsm in adata.obsm:
            return self._run_probability_entropy_only(
                adata=adata,
                soft_scores_obsm=soft_scores_obsm,
                scratch=scratch,
            )

        # Rscript + backend + package guards (hoisted to RAnalysisMethod).
        backend, skip = self._resolve_rscript_backend(context, config)
        if skip is not None:
            return skip

        # Write query h5ad (lognorm layer + X_pca + cell_type).
        query_h5ad = scratch / "scdiag_query.h5ad"
        self._write_query_h5ad(adata, query_h5ad, cell_type_col, expression_layer)

        # Optional: write soft scores if provided.
        soft_scores_path = None
        if soft_scores_obsm and soft_scores_obsm in adata.obsm:
            soft_scores_path = scratch / "scdiag_soft_scores.csv"
            self._write_soft_scores(adata, soft_scores_obsm, soft_scores_path)

        # Resolve reference path (or "NONE" sentinel).
        ref_arg = (
            str(reference_h5ad) if reference_h5ad and Path(reference_h5ad).is_file() else "NONE"
        )

        # Prepare output CSV path.
        out_csv = scratch / "scdiag_results.csv"

        # Build R script args.
        args = [
            str(query_h5ad),
            str(out_csv),
            cell_type_col,
            ref_arg,
            str(soft_scores_path) if soft_scores_path else "NONE",
            ",".join(map(str, pc_subset)),
            str(n_tree),
            str(n_neighbor),
        ]

        # Run the R script (wrap errors → MethodSkip: diagnostics must not kill
        # the pipeline).
        try:
            result = backend.run_script(_SCDIAGNOSTICS_R, args, timeout=timeout)
            if result.returncode != 0:
                return self._skip(
                    "scDiagnostics R script failed", stderr=result.stderr.strip()[:500]
                )
        except (FileNotFoundError, CellQuorumBackendError) as e:
            return self._skip("R execution failed", error=str(e)[:500])

        # Read back diagnostic columns from the CSV (indexed by barcode).
        diag_df = self._read_diagnostic_csv(out_csv)

        # Join diagnostic columns onto obs by barcode (read-only; never
        # modify cell_type). Reindex to match adata.obs_names order so
        # values align to the correct cells.
        result_adata = adata.copy()

        if not diag_df.empty:
            # Reindex diagnostic DataFrame to adata.obs_names order.
            diag_df = diag_df.reindex(result_adata.obs_names)

            # Validate barcode alignment: ensure all cells have values.
            n_missing = diag_df.isnull().all(axis=1).sum()
            if n_missing > 0:
                raise CellQuorumBackendError(
                    f"scDiagnostics barcode misalignment: {n_missing} "
                    f"cells missing diagnostics after reindex. "
                    f"R script barcodes do not match adata.obs_names."
                )

            # Assign diagnostic columns to obs.
            for col in diag_df.columns:
                result_adata.obs[col] = diag_df[col].to_numpy()

        # Count which diagnostics were computed.
        diagnostics_run = [col for col in diag_df.columns if col.startswith("scdiag_")]
        notes = []
        if diagnostics_run:
            notes.append(
                f"Computed {len(diagnostics_run)} diagnostic columns: " f"{diagnostics_run}"
            )
        else:
            notes.append(
                "No diagnostics computed: provide reference_h5ad and/or "
                "soft_scores_obsm to enable diagnostics."
            )

        # Build artifacts list.
        artifacts = [
            StageArtifact(
                name="scdiagnostics_results",
                path=out_csv,
                kind="csv",
                description="Per-cell scDiagnostics confidence metrics.",
            )
        ]

        return StageResult(
            adata=result_adata,
            artifacts=artifacts,
            notes=notes,
            metrics={
                "n_diagnostics": len(diagnostics_run),
                "diagnostics_computed": diagnostics_run,
                "reference_used": ref_arg != "NONE",
            },
        )

    def _run_probability_entropy_only(
        self,
        *,
        adata: ad.AnnData,
        soft_scores_obsm: str,
        scratch: Path,
    ) -> StageResult:
        """Compute per-cell annotation entropy from an obsm probability matrix."""

        scores = np.asarray(adata.obsm[soft_scores_obsm], dtype=float)
        if scores.ndim != 2 or scores.shape[0] != adata.n_obs:
            raise CellQuorumBackendError(
                f"soft_scores_obsm '{soft_scores_obsm}' must be a 2D matrix with "
                f"{adata.n_obs} rows."
            )

        row_sums = scores.sum(axis=1, keepdims=True)
        probs = np.divide(scores, row_sums, out=np.zeros_like(scores), where=row_sums > 0)
        safe_probs = np.where(probs > 0, probs, 1.0)
        entropy_values = -(probs * np.log2(safe_probs)).sum(axis=1)

        result_adata = adata.copy()
        result_adata.obs["scdiag_entropy"] = entropy_values

        out_csv = scratch / "scdiag_results.csv"
        pd.DataFrame(
            {
                "barcode": result_adata.obs_names,
                "scdiag_entropy": entropy_values,
            }
        ).to_csv(out_csv, index=False)

        return StageResult(
            adata=result_adata,
            artifacts=[
                StageArtifact(
                    name="scdiagnostics_results",
                    path=out_csv,
                    kind="csv",
                    description="Per-cell annotation entropy from soft label probabilities.",
                )
            ],
            notes=[
                "Computed scdiag_entropy from soft label probabilities " f"('{soft_scores_obsm}')."
            ],
            metrics={
                "n_diagnostics": 1,
                "diagnostics_computed": ["scdiag_entropy"],
                "reference_used": False,
                "soft_scores_obsm": soft_scores_obsm,
            },
        )

    def _write_query_h5ad(
        self,
        adata: ad.AnnData,
        path: Path,
        cell_type_col: str,
        expression_layer: str = "lognorm",
    ) -> None:
        """Write query AnnData to h5ad (lognorm layer + X_pca + cell label)."""
        # Prepare a minimal h5ad for R consumption.
        query = ad.AnnData(X=adata.layers[expression_layer].copy())
        query.obs_names = adata.obs_names
        query.var_names = adata.var_names
        query.obs[cell_type_col] = adata.obs[cell_type_col].values
        query.obsm["X_pca"] = adata.obsm["X_pca"].copy()
        write_h5ad(query, path)

    def _write_soft_scores(
        self,
        adata: ad.AnnData,
        obsm_key: str,
        path: Path,
    ) -> None:
        """Write soft probability matrix to CSV (cells x cell_types)."""
        import pandas as pd

        scores = adata.obsm[obsm_key]
        df = pd.DataFrame(scores, index=adata.obs_names)
        df.to_csv(path)

    def _read_diagnostic_csv(self, path: Path) -> pd.DataFrame:
        """Read per-cell diagnostic CSV as a DataFrame indexed by barcode.

        Returns:
            pandas.DataFrame indexed by barcode with diagnostic columns.
        """
        import pandas as pd

        # Find the barcode column first (case-insensitive check).
        temp_df = pd.read_csv(path, nrows=0)
        barcode_col = None
        for col in temp_df.columns:
            if col.lower() in ("barcode", "cell"):
                barcode_col = col
                break

        if barcode_col is None:
            raise ValueError(
                "scDiagnostics CSV missing barcode column; " "cannot align diagnostics to cells"
            )

        # Read CSV with barcode column as string to match adata.obs_names.
        df = pd.read_csv(path, dtype={barcode_col: str})
        if df.empty:
            return pd.DataFrame()

        # Set barcode as index.
        df = df.set_index(barcode_col)
        return df


__all__ = ["ScdiagnosticsMethod"]
