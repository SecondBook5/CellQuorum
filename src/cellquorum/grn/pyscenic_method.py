"""Classic pySCENIC gene-regulatory network inference method."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from cellquorum.backends.pyscenic_backend import PYSCENIC_AUCELL_PY, PYSCENIC_GRN_PY
from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.grn import regulon_figures as rf
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class PyscenicMethod(AnalysisMethod):
    """Directed TF->target GRN inference via classic pySCENIC in an isolated env."""

    name = "pyscenic"
    stage_category = "grn"
    backend = "pyscenic"

    def input_contract(self, config: dict) -> DataContract:
        """Require the counts layer; group_by is NOT a hard obs requirement.

        group_by falls back to cell_type -> leiden -> "all", so requiring it here
        would hard-fail the contract instead of allowing the graceful fallback.
        """
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
        group_by_cfg = config.get("group_by")
        num_workers = int(config.get("num_workers", 8))
        max_cells = int(config.get("max_cells", 20000))
        min_cells_total = int(config.get("min_cells_total", 200))
        top_n = int(config.get("top_n", 5))
        seed = int(config.get("seed", 0))
        launcher = config.get("launcher", "micromamba")
        timeout_seconds = int(config.get("timeout_seconds", 7200))
        tfs_path = config.get("tfs_path")
        motifs_path = config.get("motifs_path")
        rankings_glob = config.get("rankings_glob")

        # 2. Resolve group_by generically
        if group_by_cfg:
            group_by = group_by_cfg
        elif "cell_type" in adata.obs.columns:
            group_by = "cell_type"
        elif "leiden" in adata.obs.columns:
            group_by = "leiden"
        else:
            group_by = "all"

        # 3. Guards -> MethodSkip
        if adata.n_obs < min_cells_total:
            return self._skip(
                f"too few cells ({adata.n_obs} < {min_cells_total})", n_obs=int(adata.n_obs)
            )
        if shutil.which(launcher) is None:
            return self._skip(f"launcher '{launcher}' not found on PATH", launcher=launcher)
        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("pyscenic")
            except Exception:
                backend = None
        if backend is None:
            return self._skip("pyscenic backend unavailable")
        try:
            module_ok = backend._py_module_available("pyscenic")
        except Exception:
            module_ok = False
        if not module_ok:
            return self._skip("pyscenic module unavailable in env")
        # cisTarget resources: unset OR unresolvable -> skip before spawning subprocess
        rankings_files = sorted(glob.glob(rankings_glob)) if rankings_glob else []
        if not rankings_files and rankings_glob and os.path.exists(rankings_glob):
            rankings_files = [rankings_glob]
        missing_db = []
        if not tfs_path or not os.path.exists(tfs_path):
            missing_db.append("tfs_path")
        if not motifs_path or not os.path.exists(motifs_path):
            missing_db.append("motifs_path")
        if not rankings_files:
            missing_db.append("rankings_glob")
        if missing_db:
            return self._skip(
                "cisTarget resources unset/missing "
                f"({', '.join(missing_db)}); download the cisTarget DBs and set "
                "grn.tfs_path / motifs_path / rankings_glob",
                missing=missing_db,
            )

        # 4. Write counts h5ad to scratch
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        h5ad = scratch / "grn_input.h5ad"
        if layer and layer != "X" and layer in adata.layers:
            a2 = adata.copy()
            a2.X = a2.layers[layer]
            a2.write_h5ad(h5ad)
        else:
            adata.write_h5ad(h5ad)

        out_dir = Path(getattr(context.paths, "results", ".")) / "grn"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "grn"

        # 5. Run grn -> ctx in-env script
        grn_args = [
            "--h5ad",
            str(h5ad),
            "--tfs",
            str(tfs_path),
            "--motifs",
            str(motifs_path),
            "--rankings",
            str(rankings_glob),
            "--out-dir",
            str(out_dir),
            "--tag",
            tag,
            "--num-workers",
            str(num_workers),
            "--max-cells",
            str(max_cells),
            "--layer",
            str(layer),
            "--seed",
            str(seed),
        ]
        try:
            proc = backend.run_script(PYSCENIC_GRN_PY, grn_args, timeout=timeout_seconds)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            return self._skip("grn script execution failed or timed out", error=str(exc)[:500])
        if proc.returncode != 0:
            return self._skip("grn/ctx script failed", stderr=proc.stderr.strip()[:500])

        regulons_csv = out_dir / f"scenic_regulons_{tag}.csv"
        adjacencies_tsv = out_dir / f"scenic_adjacencies_{tag}.tsv"
        loom_path = out_dir / f"scenic_input_{tag}.loom"
        skip_marker = out_dir / f"grn_SKIPPED_{tag}.txt"
        if skip_marker.exists() or not regulons_csv.exists():
            reason = (
                skip_marker.read_text().strip() if skip_marker.exists() else "no regulons produced"
            )
            return self._skip(reason)
        try:
            regulons_df = pd.read_csv(regulons_csv)
        except Exception as exc:
            return self._skip("could not read regulons CSV", error=str(exc)[:500])
        if len(regulons_df) == 0:
            return self._skip("no regulons detected")

        # 6. AUCell
        auc_parquet = out_dir / f"scenic_auc_mtx_{tag}.parquet"
        aucell_args = [
            "--loom",
            str(loom_path),
            "--regulons",
            str(regulons_csv),
            "--out",
            str(auc_parquet),
            "--num-workers",
            str(num_workers),
        ]
        notes: list[str] = []
        auc = None
        try:
            aproc = backend.run_script(PYSCENIC_AUCELL_PY, aucell_args, timeout=timeout_seconds)
            if aproc.returncode == 0 and auc_parquet.exists():
                auc = pd.read_parquet(auc_parquet)
                if auc.empty:
                    auc = None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            notes.append(f"AUCell skipped: {str(exc)[:200]}")

        # 7. Figures (in cellquorum env) when AUC available
        figs: list[Path] = []
        if auc is not None and auc.shape[1] > 0:
            groups = None
            if group_by in adata.obs.columns:
                groups = adata.obs[group_by].astype(str)
                groups.index = adata.obs_names
            common = auc.index.intersection(adata.obs_names)
            if groups is not None and len(common) > 0:
                for fn, kwargs in (
                    (rf.plot_rss_panels, {"group_label": group_by, "top_n": top_n}),
                    (rf.plot_regulon_clustermap, {"group_label": group_by, "top_n": top_n}),
                ):
                    try:
                        figs.extend(fn(auc, groups, out_dir, **kwargs))
                    except Exception as exc:
                        notes.append(f"{fn.__name__} failed: {str(exc)[:150]}")
                try:
                    ann = groups.to_frame(group_by)
                    figs.extend(rf.plot_regulon_cell_clustermap(auc, ann, out_dir, top_n=top_n))
                except Exception as exc:
                    notes.append(f"cell clustermap failed: {str(exc)[:150]}")
            if "X_umap" in adata.obsm and len(common) > 0:
                try:
                    umap = pd.DataFrame(
                        adata.obsm["X_umap"][:, :2],
                        index=adata.obs_names,
                        columns=["UMAP1", "UMAP2"],
                    )
                    figs.extend(rf.plot_regulon_umap(auc, umap, out_dir, groups=groups, top_n=12))
                except Exception as exc:
                    notes.append(f"regulon-UMAP failed: {str(exc)[:150]}")

        # 8. Artifacts
        artifacts: list[StageArtifact] = [
            StageArtifact(
                name="regulons",
                path=regulons_csv,
                kind="csv",
                description="pySCENIC ctx regulons (TF->target enrichments)",
            ),
        ]
        if adjacencies_tsv.exists():
            artifacts.append(
                StageArtifact(
                    name="adjacencies",
                    path=adjacencies_tsv,
                    kind="tsv",
                    description="GRNBoost2 TF-target adjacencies",
                )
            )
        if auc_parquet.exists():
            artifacts.append(
                StageArtifact(
                    name="auc_mtx",
                    path=auc_parquet,
                    kind="parquet",
                    description="AUCell per-cell regulon activity (cells x regulons)",
                )
            )
        for fig_path in figs:
            artifacts.append(
                StageArtifact(
                    name=f"figure_{fig_path.stem}",
                    path=fig_path,
                    kind="figure",
                    description="pySCENIC regulon figure",
                )
            )

        # 9. Metrics
        n_tfs = int(regulons_df.iloc[:, 0].nunique()) if len(regulons_df.columns) else 0
        metrics = {
            "n_regulons": int(auc.shape[1]) if auc is not None else int(len(regulons_df)),
            "n_tfs": n_tfs,
            "n_cells_scored": int(auc.shape[0]) if auc is not None else 0,
            "group_by": group_by,
            "n_obs": int(adata.n_obs),
        }

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics=metrics,
            backend="pyscenic",
        )


__all__ = ["PyscenicMethod"]
