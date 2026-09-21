"""MetaCell2 partitioning method."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.backends.metacells_backend import RUN_METACELL2_PY
from cellquorum.core.contracts.data_contract import DataContract
from cellquorum.core.h5ad_io import write_h5ad
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.metacells.aggregate import outlier_summary


class MetacellsMethod(AnalysisMethod):
    """Partition cells into near-homogeneous metacells via MetaCell2 in an isolated env."""

    name = "metacells"
    stage_category = "metacells"
    backend = "metacells"

    def input_contract(self, config: dict) -> DataContract:
        layer = config.get("layer", "counts")
        return DataContract(
            required_layers=[layer],
            required_obs=[],
            expression_layer=layer,
            expected_kind="counts",
        )

    def requires_obs(self, config: dict) -> list[str]:
        return []

    def _run(self, adata, config, context) -> StageResult | MethodSkip:  # noqa: ANN001
        # 1. Resolve config
        layer = config.get("layer", "counts")
        rep_key = config.get("rep_key", "X_pca_harmony")
        group_by = config.get("group_by", "ref_cell_type_granular_coarse")
        target_metacell_size = int(config.get("target_metacell_size", 96))
        min_cells = int(config.get("min_cells", 5000))
        seed = int(config.get("seed", 0))
        launcher = config.get("launcher", "micromamba")
        timeout_seconds = int(config.get("timeout_seconds", 3600))

        # 2. Guards -> MethodSkip
        if adata.n_obs < min_cells:
            return self._skip(
                f"too few cells ({adata.n_obs} < {min_cells})", n_obs=int(adata.n_obs)
            )
        if group_by not in adata.obs.columns:
            return self._skip(f"group_by column '{group_by}' absent", group_by=group_by)
        if rep_key not in adata.obsm:
            return self._skip(f"rep_key '{rep_key}' absent from obsm", rep_key=rep_key)
        if shutil.which(launcher) is None:
            return self._skip(f"launcher '{launcher}' not found on PATH", launcher=launcher)
        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("metacells")
            except Exception:
                backend = None
        if backend is None:
            return self._skip("metacells backend unavailable")
        try:
            module_ok = backend._py_module_available("metacells")
        except Exception:
            module_ok = False
        if not module_ok:
            return self._skip("metacells module unavailable in env")

        # 3. Write scratch h5ad (raw counts + rep + group_by only)
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        h5ad = scratch / "metacells_input.h5ad"
        write_h5ad(
            ad.AnnData(
                X=adata.layers[layer] if layer != "X" else adata.X,
                obs=adata.obs[[group_by]].copy(),
                var=adata.var[[]].copy(),
                obsm={rep_key: adata.obsm[rep_key]},
            ),
            h5ad,
        )

        out_dir = Path(getattr(context.paths, "results", ".")) / "metacells"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 4. Run the in-env script
        args = [
            str(h5ad),
            str(out_dir),
            rep_key,
            "X",  # the scratch h5ad already carries counts as X
            group_by,
            str(target_metacell_size),
            str(min_cells),
            str(seed),
        ]
        try:
            proc = backend.run_script(RUN_METACELL2_PY, args, timeout=timeout_seconds)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            return self._skip(
                "metacells script execution failed or timed out", error=str(exc)[:500]
            )
        if proc.returncode != 0:
            return self._skip("metacells script failed", stderr=proc.stderr.strip()[:500])

        assignments_csv = out_dir / "metacell_assignments.csv"
        skip_marker = out_dir / "metacells_SKIPPED.txt"
        if skip_marker.exists() or not assignments_csv.exists():
            reason = (
                skip_marker.read_text().strip()
                if skip_marker.exists()
                else "no assignments produced"
            )
            return self._skip(reason)
        try:
            assignments = pd.read_csv(assignments_csv)
        except Exception as exc:
            return self._skip("could not read assignments CSV", error=str(exc)[:500])
        if len(assignments) == 0:
            return self._skip("no cell assignments produced")

        # 5. Write metacell_id onto the main object
        id_by_cell = dict(zip(assignments["cell_id"], assignments["metacell_id"], strict=False))
        adata.obs["metacell_id"] = (
            adata.obs_names.to_series().map(id_by_cell).fillna("Outliers").astype(str).to_numpy()
        )

        # 6. Artifacts
        artifacts: list[StageArtifact] = [
            StageArtifact(
                name="metacell_assignments",
                path=assignments_csv,
                kind="csv",
                description="Per-cell metacell assignment (including the Outliers bucket)",
            ),
        ]
        mc_adata_path = out_dir / "metacell_adata.h5ad"
        if mc_adata_path.exists():
            artifacts.append(
                StageArtifact(
                    name="metacell_adata",
                    path=mc_adata_path,
                    kind="h5ad",
                    description="Aggregated (pseudobulk) metacell AnnData",
                )
            )

        # 7. Metrics
        summary = outlier_summary(assignments)
        metrics = {
            "n_metacells": summary["n_metacells"],
            "n_outlier_cells": summary["n_outlier_cells"],
            "pct_outlier": summary["pct_outlier"],
            "group_by": group_by,
            "n_obs": int(adata.n_obs),
        }

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            warnings=[],
            metrics=metrics,
            backend="metacells",
        )


__all__ = ["MetacellsMethod"]
