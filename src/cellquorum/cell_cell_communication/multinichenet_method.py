"""MultiNicheNet method: tissue-wide differential CCC via multinichenetr (R)."""

from __future__ import annotations

import shutil  # noqa: F401 - test mocks through this module path
import subprocess
from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.cell_cell_communication._nichenet_io import (
    export_sce_inputs,
    mnn_prioritization_to_canonical,
)
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod

_MNN_R = Path(__file__).parent.parent / "backends" / "r_scripts" / "multinichenet.R"


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

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
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
        backend, skip = self._resolve_rscript_backend(context)
        if skip is not None:
            return skip

        scratch = Path(context.paths.scratch)
        paths = export_sce_inputs(adata, [cell_type_col, sample_col, condition_col], scratch)

        results_dir = Path(context.paths.results)
        results_dir.mkdir(parents=True, exist_ok=True)
        native_csv = results_dir / "mnn_prioritization.csv"

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
            str(config.get("nichenet_n_cores", 4)),
            str(seed),
        ]

        try:
            proc = backend.run_script(_MNN_R, args, timeout=timeout)
        except FileNotFoundError as exc:
            return self._skip("R execution failed", error=str(exc)[:500])
        except subprocess.TimeoutExpired as exc:
            return self._skip(f"R timed out after {timeout}s", error=str(exc)[:500])
        if proc.returncode != 0:
            return self._skip("multinichenet.R failed", stderr=proc.stderr.strip()[:500])

        artifacts = [
            StageArtifact(
                name="mnn_prioritization",
                path=native_csv,
                kind="csv",
                description=f"MultiNicheNet prioritization ({case} vs {control}).",
            )
        ]
        n_prioritized = None
        canonical_csv = results_dir / "mnn_canonical_lr.csv"
        try:
            native = pd.read_csv(native_csv)
            n_prioritized = len(native)
            canonical = mnn_prioritization_to_canonical(native)
            canonical.to_csv(canonical_csv, index=False)
            artifacts.append(
                StageArtifact(
                    name="mnn_canonical_lr",
                    path=canonical_csv,
                    kind="csv",
                    description="MultiNicheNet LR edges in canonical schema (for ccc_network).",
                )
            )
        except Exception:
            pass  # native artifact already recorded; never crash on post-processing

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[
                f"MultiNicheNet: {case} vs {control}, top_n_target="
                f"{config.get('mnn_top_n_target', 250)}."
            ],
            metrics={"case": case, "control": control, "n_prioritized": n_prioritized},
            backend="rscript",
        )


__all__ = ["MultiNicheNetMethod"]
