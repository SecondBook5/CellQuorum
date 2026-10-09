"""MultiNicheNet method: tissue-wide differential CCC via multinichenetr (R)."""

from __future__ import annotations

import shutil  # noqa: F401 - test mocks through this module path
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import pandas as pd

from cellquorum.backends.script_paths import r_script_path
from cellquorum.core.context import resolve_n_jobs
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod
from cellquorum.stages.cell_cell_communication._nichenet_io import (
    export_sce_inputs,
    mnn_prioritization_to_canonical,
)

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext

_MNN_R = r_script_path("multinichenet.R")


class MultiNicheNetMethod(RAnalysisMethod):
    """Tissue-wide differential cell-cell communication (multinichenetr)."""

    name = "multinichenet"
    stage_category = "cell_cell_communication"
    r_package = "multinichenetr"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract(
            required_obs=[
                config.get("cell_type_col", "cell_type"),
                config.get("sample_col", "sample_id"),
                config.get("condition_col", "condition"),
            ],
        )

    def requires_obs(self, config: dict) -> list[str]:
        return [
            config.get("cell_type_col", "cell_type"),
            config.get("sample_col", "sample_id"),
            config.get("condition_col", "condition"),
        ]

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        cell_type_col = config.get("cell_type_col", "cell_type")
        sample_col = config.get("sample_col", "sample_id")
        condition_col = config.get("condition_col", "condition")
        case = config.get("case")
        control = config.get("control")
        seed = int(config.get("seed", 42))

        if not case or not control:
            return self._skip("no contrast (case/control) declared")

        # Verify the tokens are actually present in the data.
        observed = set(adata.obs[condition_col].astype(str).unique())
        if case not in observed or control not in observed:
            return self._skip(
                "case/control tokens absent from condition_col", observed=sorted(observed)
            )

        lt = config.get("nichenet_ligand_target_matrix")
        lr = config.get("nichenet_lr_network")
        if not lt or not lr or not Path(lt).is_file() or not Path(lr).is_file():
            return self._skip("prior-model paths missing")

        # Rscript + backend + package guards (hoisted to RAnalysisMethod).
        backend = self._resolve_rscript_backend(context)
        if isinstance(backend, MethodSkip):
            return backend

        scratch = Path(context.paths.scratch)
        covariates: list[str] = []
        n_pairs = None
        export_columns = [cell_type_col, sample_col, condition_col]
        analysis_adata = adata
        if config.get("paired", False):
            donor_col = config.get("donor_col")
            if not donor_col or donor_col not in adata.obs:
                raise ValueError("Paired MultiNicheNet requires an existing donor_col")
            obs = adata.obs
            if obs[[sample_col, donor_col, condition_col]].isna().any().any():
                raise ValueError("Missing sample, donor or condition in paired MultiNicheNet")
            sample_meta = obs[[sample_col, donor_col, condition_col]].drop_duplicates()
            if sample_meta[sample_col].duplicated().any():
                raise ValueError("Each sample must map to exactly one donor and condition")
            cell_counts = (
                obs.groupby([sample_col, cell_type_col], observed=True).size().unstack(fill_value=0)
            )
            minimum = int(config.get("nichenet_min_cells", 10))
            eligible_samples = cell_counts.index[cell_counts.ge(minimum).all(axis=1)]
            eligible_meta = sample_meta[sample_meta[sample_col].isin(eligible_samples)]
            pair_counts = pd.crosstab(eligible_meta[donor_col], eligible_meta[condition_col])
            pair_counts = pair_counts.reindex(columns=[control, case], fill_value=0)
            donors = pair_counts.index[pair_counts.eq(1).all(axis=1)]
            audit = sample_meta.copy()
            audit["all_celltypes_min_cells"] = audit[sample_col].isin(eligible_samples)
            audit["included_paired_analysis"] = audit[donor_col].isin(donors) & audit[
                condition_col
            ].isin([control, case])
            scratch.mkdir(parents=True, exist_ok=True)
            audit.to_csv(scratch / "mnn_pairing_audit.csv", index=False)
            if len(donors) < 3:
                raise ValueError("Paired MultiNicheNet requires >=3 complete eligible donor pairs")
            keep = obs[donor_col].isin(donors) & obs[condition_col].isin([control, case])
            analysis_adata = adata[keep].copy()
            n_pairs = len(donors)
            covariates = [donor_col]
            export_columns.append(donor_col)
        counts_layer = config.get("nichenet_counts_layer")
        if counts_layer:
            if counts_layer not in analysis_adata.layers:
                raise ValueError(f"MultiNicheNet counts layer missing: {counts_layer}")
            if analysis_adata is adata:
                analysis_adata = analysis_adata.copy()
            analysis_adata.X = analysis_adata.layers[counts_layer].copy()
        paths = export_sce_inputs(analysis_adata, export_columns, scratch, subdir="multinichenet")

        results_dir = Path(context.paths.results)
        results_dir.mkdir(parents=True, exist_ok=True)
        writer = StageArtifactWriter.from_context(context)
        native_csv = results_dir / "mnn_prioritization.csv"
        sample_csv = results_dir / "mnn_sample_prioritization.csv"
        source_csv = results_dir / "mnn_group_prioritization_source.csv"

        timeout = int(config.get("nichenet_timeout_seconds", 7200))
        args = [
            str(paths["counts"]),
            str(paths["genes"]),
            str(paths["barcodes"]),
            str(paths["obs"]),
            str(native_csv),
            cell_type_col,
            sample_col,
            condition_col,
            case,
            control,
            str(lt),
            str(lr),
            str(config.get("mnn_fraction_cutoff", 0.05)),
            str(config.get("mnn_min_sample_prop", 0.5)),
            str(config.get("mnn_logfc_threshold", 0.5)),
            str(config.get("mnn_p_val_threshold", 0.05)),
            "TRUE" if config.get("mnn_p_val_adj", False) else "FALSE",
            str(config.get("mnn_top_n_target", 250)),
            str(config.get("mnn_scenario", "regular")),
            # Unset inherits compute.n_jobs, so one knob governs both this and the
            # Python-side parallel steps instead of the R side keeping its own 4.
            str(resolve_n_jobs(context, config.get("nichenet_n_cores"))),
            str(seed),
            str(sample_csv),
            str(source_csv),
            ",".join(covariates),
            str(config.get("nichenet_min_cells", 10)),
        ]

        try:
            proc = backend.run_script(_MNN_R, args, timeout=timeout)
        except FileNotFoundError as exc:
            return self._skip("R execution failed", error=str(exc)[:500])
        except subprocess.TimeoutExpired as exc:
            return self._skip(f"R timed out after {timeout}s", error=str(exc)[:500])
        if proc.returncode != 0:
            if config.get("paired", False):
                raise RuntimeError(f"Paired MultiNicheNet failed: {proc.stderr.strip()[-3000:]}")
            return self._skip("multinichenet.R failed", stderr=proc.stderr.strip()[:500])

        artifacts = [
            StageArtifact(
                name="mnn_prioritization",
                path=native_csv,
                kind="csv",
                description=f"MultiNicheNet prioritization ({case} vs {control}).",
            )
        ]
        for path, name, description in (
            (
                sample_csv,
                "mnn_sample_prioritization",
                "Per-sample scaled ligand-receptor expression product "
                "(multinichenetr's sample_prioritization_tbl).",
            ),
            (
                source_csv,
                "mnn_group_prioritization_source",
                "Pre-trim prioritization table with ligand/receptor celltype-"
                "specificity scores (multinichenetr's group_prioritization_table_source).",
            ),
        ):
            if path.is_file():
                artifacts.append(
                    StageArtifact(name=name, path=path, kind="csv", description=description)
                )
        for path, name, description in (
            (
                scratch / "mnn_pairing_audit.csv",
                "mnn_pairing_audit",
                "Donor/sample eligibility and exclusions for the paired fit.",
            ),
            (
                Path(str(native_csv) + ".design.csv"),
                "mnn_design",
                "Validated donor-adjusted pseudobulk design matrix.",
            ),
        ):
            if config.get("paired", False) and path.is_file():
                artifacts.append(
                    StageArtifact(name=name, path=path, kind="csv", description=description)
                )
        n_prioritized = None
        warnings: list[str] = []
        try:
            native = pd.read_csv(native_csv)
            n_prioritized = len(native)
            canonical = mnn_prioritization_to_canonical(native)
            artifacts.append(
                writer.table(
                    canonical,
                    "mnn_canonical_lr.csv",
                    name="mnn_canonical_lr",
                    description="MultiNicheNet LR edges in canonical schema (for ccc_network).",
                    index=False,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- the native table is already recorded
            warnings.append(
                "MultiNicheNet finished but its results could not be converted to the shared "
                "ligand-receptor format, so ccc_network will have no MultiNicheNet edges "
                f"({type(exc).__name__}: {exc}). The native table mnn_prioritization.csv is intact."
            )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            warnings=warnings,
            notes=[
                f"MultiNicheNet: {case} vs {control}, top_n_target="
                f"{config.get('mnn_top_n_target', 250)}."
            ],
            metrics={
                "case": case,
                "control": control,
                "n_prioritized": n_prioritized,
                "paired": bool(config.get("paired", False)),
                "n_pairs": n_pairs,
                "covariates": covariates,
            },
            backend="rscript",
        )


__all__ = ["MultiNicheNetMethod"]
