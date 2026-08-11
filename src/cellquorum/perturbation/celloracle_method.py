"""In-silico transcription-factor knockout via CellOracle in an isolated env."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from cellquorum.backends.celloracle_backend import CELLORACLE_KO_PY
from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.perturbation import perturbation_figures as pfig


class CellOracleMethod(AnalysisMethod):
    """Directed in-silico TF knockout + fate-shift ranking via CellOracle."""

    name = "celloracle"
    stage_category = "perturbation"
    backend = "celloracle"

    def input_contract(self, config: dict) -> DataContract:
        """Require the counts layer; condition/cluster keys are NOT hard obs reqs.

        They fall back generically, so requiring them here would hard-fail the
        contract instead of allowing the fallback (the grn/hdWGCNA lesson).
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
        organism = config.get("organism", "human")
        min_cells_total = int(config.get("min_cells_total", 200))
        n_top_targets = int(config.get("n_top_targets", 20))
        filter_edge_number = int(config.get("filter_edge_number", 10000))
        knn_n_neighbors = int(config.get("knn_n_neighbors", 200))
        n_propagation = int(config.get("n_propagation", 3))
        seed = int(config.get("seed", 0))
        launcher = config.get("launcher", "micromamba")
        timeout_seconds = int(config.get("timeout_seconds", 10800))
        condition_key = config.get("condition_key")
        healthy_label = config.get("healthy_label")
        tf_list = config.get("tf_list")

        # 2. Resolve generic keys
        cluster_key = config.get("cluster_key")
        if not cluster_key:
            cluster_key = (
                "cell_type"
                if "cell_type" in adata.obs.columns
                else "leiden"
                if "leiden" in adata.obs.columns
                else "all"
            )
        rep_key = config.get("rep_key")
        if not rep_key:
            rep_key = (
                "X_pca"
                if "X_pca" in adata.obsm
                else ("X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca")
            )
        embedding_key = config.get("embedding_key") or "X_umap"

        # 3. Guards -> MethodSkip
        if adata.n_obs < min_cells_total:
            return MethodSkip(
                reason=f"celloracle skipped: too few cells ({adata.n_obs} < {min_cells_total})",
                details={"method": self.name, "n_obs": int(adata.n_obs)},
            )
        if shutil.which(launcher) is None:
            return MethodSkip(
                reason=f"celloracle skipped: launcher '{launcher}' not found on PATH",
                details={"method": self.name, "launcher": launcher},
            )
        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("celloracle")
            except Exception:
                backend = None
        if backend is None:
            return MethodSkip(
                reason="celloracle skipped: celloracle backend unavailable",
                details={"method": self.name},
            )
        try:
            module_ok = backend._py_module_available("celloracle")
        except Exception:
            module_ok = False
        if not module_ok:
            return MethodSkip(
                reason="celloracle skipped: celloracle module unavailable in env",
                details={"method": self.name},
            )

        # 4. Write counts h5ad to scratch
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        h5ad = scratch / "perturbation_input.h5ad"
        if layer and layer != "X" and layer in adata.layers:
            a2 = adata.copy()
            a2.X = a2.layers[layer]
            a2.write_h5ad(h5ad)
        else:
            adata.write_h5ad(h5ad)

        out_dir = Path(getattr(context.paths, "results", ".")) / "perturbation"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "perturbation"

        # 5. Run the in-env KO script
        ko_args = [
            "--h5ad",
            str(h5ad),
            "--out-dir",
            str(out_dir),
            "--tag",
            tag,
            "--organism",
            str(organism),
            "--cluster-key",
            str(cluster_key),
            "--rep-key",
            str(rep_key),
            "--embedding-key",
            str(embedding_key),
            "--condition-key",
            str(condition_key or ""),
            "--healthy-label",
            str(healthy_label or ""),
            "--tf-list",
            " ".join(tf_list) if tf_list else "",
            "--n-top-targets",
            str(n_top_targets),
            "--filter-edge-number",
            str(filter_edge_number),
            "--knn-n-neighbors",
            str(knn_n_neighbors),
            "--n-propagation",
            str(n_propagation),
            "--seed",
            str(seed),
        ]
        try:
            proc = backend.run_script(CELLORACLE_KO_PY, ko_args, timeout=timeout_seconds)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            return MethodSkip(
                reason="celloracle skipped: KO script execution failed or timed out",
                details={"method": self.name, "error": str(exc)[:500]},
            )
        if proc.returncode != 0:
            return MethodSkip(
                reason="celloracle skipped: KO script failed",
                details={
                    "method": self.name,
                    "stderr": str(getattr(proc, "stderr", "")).strip()[:500],
                },
            )

        ranking_csv = out_dir / "perturbation_ranking.csv"
        failed_marker = out_dir / f"perturbation_FAILED_{tag}.txt"
        skip_marker = out_dir / f"perturbation_SKIPPED_{tag}.txt"
        if failed_marker.exists():
            return MethodSkip(
                reason=f"celloracle skipped: {failed_marker.read_text().strip()[:300]}",
                details={"method": self.name},
            )
        if skip_marker.exists():
            return MethodSkip(
                reason=f"celloracle skipped: {skip_marker.read_text().strip()[:300]}",
                details={"method": self.name},
            )
        if not ranking_csv.exists():
            return MethodSkip(
                reason="celloracle skipped: no ranking produced",
                details={"method": self.name},
            )

        try:
            ranking = pd.read_csv(ranking_csv)
        except Exception as exc:
            return MethodSkip(
                reason="celloracle skipped: could not read ranking CSV",
                details={"method": self.name, "error": str(exc)[:500]},
            )

        # 6. Figures (in cellquorum env) — never let one failure sink the stage
        notes: list[str] = []
        figs: list[Path] = []
        if len(ranking) > 0:
            try:
                figs.extend(pfig.plot_target_ranking(ranking, out_dir, n_top=n_top_targets))
            except Exception as exc:
                notes.append(f"target-ranking figure failed: {str(exc)[:150]}")
            grn_summary = out_dir / "grn_summary.csv"
            if grn_summary.exists():
                try:
                    figs.extend(
                        pfig.plot_grn_connectivity(
                            pd.read_csv(grn_summary), out_dir, n_top=n_top_targets
                        )
                    )
                except Exception as exc:
                    notes.append(f"grn-connectivity figure failed: {str(exc)[:150]}")
            # shift-field for the top TF, if its shift vectors + embedding are present
            if embedding_key in adata.obsm and len(ranking) > 0:
                top_tf = str(ranking.sort_values("score", ascending=False).iloc[0]["tf"])
                shift_pq = out_dir / f"shift_vectors_{top_tf}.parquet"
                if shift_pq.exists():
                    shift_df = None
                    try:
                        shift_df = pd.read_parquet(shift_pq)
                        emb = pd.DataFrame(
                            adata.obsm[embedding_key][:, :2],
                            index=adata.obs_names,
                            columns=["DIM1", "DIM2"],
                        )
                        groups = (
                            adata.obs[cluster_key].astype(str)
                            if cluster_key in adata.obs.columns
                            else None
                        )
                        figs.extend(
                            pfig.plot_ko_shift_field(
                                shift_df, emb, out_dir, tf=top_tf, groups=groups
                            )
                        )
                    except Exception as exc:
                        notes.append(f"shift-field figure failed: {str(exc)[:150]}")
                    # Gridded vector field (CellOracle-style) — the publication view
                    if shift_df is not None:
                        try:
                            emb_grid = pd.DataFrame(
                                adata.obsm[embedding_key][:, :2],
                                index=adata.obs_names,
                                columns=["DIM1", "DIM2"],
                            )
                            groups_grid = (
                                adata.obs[cluster_key].astype(str)
                                if cluster_key in adata.obs.columns
                                else None
                            )
                            figs.extend(
                                pfig.plot_ko_shift_grid(
                                    shift_df, emb_grid, out_dir, tf=top_tf, groups=groups_grid
                                )
                            )
                        except Exception as exc:
                            notes.append(f"shift-grid figure failed: {str(exc)[:150]}")
                    # Fate summary: per-cluster mean shift magnitude (direction-agnostic)
                    if shift_df is not None and cluster_key in adata.obs.columns:
                        try:
                            common = shift_df.index.intersection(adata.obs_names)
                            if len(common) > 0:
                                mags = np.linalg.norm(
                                    shift_df.loc[common].iloc[:, :2].to_numpy(), axis=1
                                )
                                fate_df = (
                                    pd.DataFrame(
                                        {
                                            "cluster": adata.obs.loc[common, cluster_key]
                                            .astype(str)
                                            .to_numpy(),
                                            "delta": mags,
                                        }
                                    )
                                    .groupby("cluster", as_index=False)["delta"]
                                    .mean()
                                )
                                figs.extend(pfig.plot_ko_fate_summary(fate_df, out_dir, tf=top_tf))
                        except Exception as exc:
                            notes.append(f"fate-summary figure failed: {str(exc)[:150]}")

        # 7. Artifacts
        artifacts: list[StageArtifact] = [
            StageArtifact(
                name="ranking",
                path=ranking_csv,
                kind="csv",
                description="Ranked in-silico knockout targets (disease->healthy shift)",
            )
        ]
        grn_summary = out_dir / "grn_summary.csv"
        if grn_summary.exists():
            artifacts.append(
                StageArtifact(
                    name="grn_summary",
                    path=grn_summary,
                    kind="csv",
                    description="CellOracle fitted GRN per-cluster top regulators",
                )
            )
        for shift_pq in sorted(out_dir.glob("shift_vectors_*.parquet")):
            artifacts.append(
                StageArtifact(
                    name=f"shift_{shift_pq.stem}",
                    path=shift_pq,
                    kind="parquet",
                    description="Per-cell KO shift vectors on the embedding",
                )
            )
        for fig_path in figs:
            artifacts.append(
                StageArtifact(
                    name=f"figure_{fig_path.stem}",
                    path=fig_path,
                    kind="figure",
                    description="CellOracle in-silico KO figure",
                )
            )

        # 8. Metrics
        condition_scored = bool(
            "direction" in ranking.columns and (ranking["direction"] == "directional").any()
        )
        metrics = {
            "n_tfs_screened": int(len(ranking)),
            "n_top_targets": int(min(n_top_targets, len(ranking))),
            "condition_scored": condition_scored,
            "cluster_key": cluster_key,
            "n_obs": int(adata.n_obs),
        }

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics=metrics,
            backend="celloracle",
        )


__all__ = ["CellOracleMethod"]
