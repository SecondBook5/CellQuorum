"""Classic pySCENIC gene-regulatory network inference method."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from cellquorum.backends.pyscenic_backend import PYSCENIC_AUCELL_PY, PYSCENIC_GRN_PY
from cellquorum.core.context import resolve_n_jobs
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.comparative.enrichment.priors import PriorFetchError, get_net
from cellquorum.stages.gene_regulation._shared import resolve_group_by
from cellquorum.stages.gene_regulation.grn import radial_figures as rgf
from cellquorum.stages.gene_regulation.grn import regulon_figures as rf
from cellquorum.stages.gene_regulation.grn.regulon_curvature import (
    compute_regulon_curvature_tables,
    load_regulons_csv,
    plot_regulon_curvature_from_tables,
)
from cellquorum.visualization import figstyle
from cellquorum.visualization.figstyle import render_figure


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

    def _skip_or_partial(
        self,
        adata,  # noqa: ANN001
        reason: str,
        *,
        figs: list[Path],
        warnings: list[str],
        **skip_kwargs: object,
    ) -> StageResult | MethodSkip:
        """``_skip(reason)`` unless the radial GRN figure already rendered.

        The radial figure (step 3) needs no pySCENIC output, so a plain
        skip at any LATER pySCENIC-pipeline guard would otherwise silently
        discard a figure that has nothing to do with why pySCENIC itself
        failed. Every pySCENIC guard/failure site routes through this
        instead of ``self._skip`` directly, so that invariant holds no
        matter where the pipeline stops.
        """
        if figs:
            warnings.append(f"pySCENIC inference skipped: {reason}")
            artifacts = [
                StageArtifact(
                    name=f"figure_{fig_path.stem}",
                    path=fig_path,
                    kind="figure",
                    description="Radial GRN figure (pySCENIC inference unavailable).",
                )
                for fig_path in figs
            ]
            return StageResult(
                adata=adata,
                artifacts=artifacts,
                warnings=warnings,
                metrics={"method": self.name, "pyscenic_skipped": reason},
                backend="python",
            )
        return self._skip(reason, **skip_kwargs)

    def _render_radial_grn(
        self,
        *,
        context,  # noqa: ANN001
        figures_dir: Path,
        radial_collection: str,
        radial_de_results_filename: str,
        radial_min_size: int,
        radial_n_tf: int,
        radial_per_tf: int,
        radial_n_sectors: int,
        radial_case_color: str | None,
        radial_control_color: str | None,
        organism: str,
        group_by: str,
        case: str | None,
        control: str | None,
        figs: list[Path],
        warnings: list[str],
    ) -> None:
        """Render the radial DoRothEA/CollecTRI GRN figure, or record why not.

        Reads the DE stage's own CSV directly (never AUCell/GRNBoost2), so
        this can run before -- and independent of -- every pySCENIC guard
        in ``_run``.
        """

        def _fig() -> list[Path]:
            de_path = Path(getattr(context.paths, "results", ".")) / radial_de_results_filename
            if not de_path.exists():
                warnings.append(f"radial GRN skipped: no DE results at {de_path}")
                return []
            try:
                de_raw = pd.read_csv(de_path)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"radial GRN skipped: could not read DE CSV ({str(exc)[:150]})")
                return []
            if "gene" not in de_raw.columns or de_raw.empty:
                warnings.append("radial GRN skipped: DE table missing 'gene' column or empty")
                return []
            try:
                net = get_net(radial_collection, organism=organism, license="academic")
            except PriorFetchError as exc:
                warnings.append(f"radial GRN skipped: {str(exc)[:200]}")
                return []
            tf_activity = rgf.tf_activity_from_de(de_raw, net, tmin=radial_min_size)
            if tf_activity.empty:
                warnings.append("radial GRN skipped: no TF activity scored")
                return []
            de_indexed = de_raw.set_index("gene")
            title = (
                f"Regulatory network — {group_by}"
                if group_by and group_by != "all"
                else "Regulatory network"
            )
            fig = rgf.radial_grn(
                tf_activity,
                de_indexed,
                net,
                title=title,
                collection_label=radial_collection.capitalize(),
                case=case,
                control=control,
                case_color=radial_case_color,
                control_color=radial_control_color,
                n_tf=radial_n_tf,
                per_tf=radial_per_tf,
                n_sectors=radial_n_sectors,
            )
            return figstyle.save_figure(fig, figures_dir, "grn_radial")

        render_figure("radial GRN", _fig, figures=figs, warnings=warnings)

    def _run(self, adata, config, context) -> StageResult | MethodSkip:  # noqa: ANN001
        # 1. Resolve config
        layer = config.get("layer", "counts")
        group_by_cfg = config.get("group_by")
        organism = config.get("organism", "human")
        # Unset inherits compute.n_jobs (see GrnConfig.num_workers): this count
        # goes to a dask LocalCluster inside the pySCENIC env, so asking for more
        # workers than the run was configured for costs memory, not just cores.
        num_workers = resolve_n_jobs(context, config.get("num_workers"))
        max_cells = int(config.get("max_cells", 20000))
        min_cells_total = int(config.get("min_cells_total", 200))
        top_n = int(config.get("top_n", 5))
        seed = int(config.get("seed", 0))
        launcher = config.get("launcher", "micromamba")
        timeout_seconds = int(config.get("timeout_seconds", 7200))
        tfs_path = config.get("tfs_path")
        motifs_path = config.get("motifs_path")
        rankings_glob = config.get("rankings_glob")
        curvature_top_k_targets = int(config.get("curvature_top_k_targets", 15))
        curvature_ricci_alpha = float(config.get("curvature_ricci_alpha", 0.5))
        curvature_plot_top_k = int(config.get("curvature_plot_top_k", 30))
        curvature_radial_n_tf = int(config.get("curvature_radial_n_tf", 12))
        curvature_radial_per_tf = int(config.get("curvature_radial_per_tf", 6))
        curvature_radial_n_sectors = int(config.get("curvature_radial_n_sectors", 4))
        cluster_labels = config.get("cluster_labels") or {}
        radial_enabled = bool(config.get("radial_enabled", True))
        radial_collection = config.get("radial_collection", "dorothea")
        radial_de_results_filename = config.get(
            "radial_de_results_filename", "de_pseudobulk_edger.csv"
        )
        radial_min_size = int(config.get("radial_min_size", 5))
        radial_n_tf = int(config.get("radial_n_tf", 12))
        radial_per_tf = int(config.get("radial_per_tf", 6))
        radial_n_sectors = int(config.get("radial_n_sectors", 4))
        radial_case_color = config.get("radial_case_color")
        radial_control_color = config.get("radial_control_color")
        case = config.get("case")
        control = config.get("control")

        # 2. Resolve group_by generically
        group_by = resolve_group_by(adata, group_by_cfg, default="all")

        figures_dir = Path(getattr(context.paths, "figures", ".")) / "grn"
        figures_dir.mkdir(parents=True, exist_ok=True)
        figs: list[Path] = []
        warnings: list[str] = []

        # 3. Radial DoRothEA/CollecTRI GRN figure: computed FIRST and kept
        # independent of every pySCENIC-specific guard below -- it needs
        # only the DE stage's own CSV and a curated prior net, never
        # GRNBoost2/cisTarget/AUCell, so a run with no cisTarget DBs
        # configured (a real, common case) must not also silently lose
        # this figure just because it lives in the same method.
        if radial_enabled:
            self._render_radial_grn(
                context=context,
                figures_dir=figures_dir,
                radial_collection=radial_collection,
                radial_de_results_filename=radial_de_results_filename,
                radial_min_size=radial_min_size,
                radial_n_tf=radial_n_tf,
                radial_per_tf=radial_per_tf,
                radial_n_sectors=radial_n_sectors,
                radial_case_color=radial_case_color,
                radial_control_color=radial_control_color,
                organism=organism,
                group_by=group_by,
                case=case,
                control=control,
                figs=figs,
                warnings=warnings,
            )

        # 4. Guards -> MethodSkip, unless the radial figure above already
        # produced real output (a plain skip would silently discard it).
        skip_reason: str | None = None
        skip_kwargs: dict[str, object] = {}
        if adata.n_obs < min_cells_total:
            skip_reason = f"too few cells ({adata.n_obs} < {min_cells_total})"
            skip_kwargs = {"n_obs": int(adata.n_obs)}
        elif shutil.which(launcher) is None:
            skip_reason = f"launcher '{launcher}' not found on PATH"
            skip_kwargs = {"launcher": launcher}
        else:
            registry = getattr(context, "backend_registry", None)
            backend = None
            if registry is not None:
                try:
                    backend = registry.get("pyscenic")
                except Exception:
                    backend = None
            if backend is None:
                skip_reason = "pyscenic backend unavailable"
            else:
                try:
                    module_ok = backend._py_module_available("pyscenic")
                except Exception:
                    module_ok = False
                if not module_ok:
                    skip_reason = "pyscenic module unavailable in env"
                else:
                    # cisTarget resources: unset OR unresolvable -> skip before
                    # spawning subprocess.
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
                        skip_reason = (
                            "cisTarget resources unset/missing "
                            f"({', '.join(missing_db)}); download the cisTarget DBs and set "
                            "grn.tfs_path / motifs_path / rankings_glob"
                        )
                        skip_kwargs = {"missing": missing_db}
        if skip_reason is not None:
            return self._skip_or_partial(
                adata, skip_reason, figs=figs, warnings=warnings, **skip_kwargs
            )

        # A skip_reason of None means every check above passed, which needs a backend.
        assert backend is not None

        # 5. Write counts h5ad to scratch
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        h5ad = scratch / "grn_input.h5ad"
        # Through the shared writer: this hands the working object to another
        # process, so a column h5py cannot encode would fail the stage here rather
        # than in pySCENIC. See cellquorum.core.h5ad_io.
        from cellquorum.core.h5ad_io import write_h5ad

        # Genuinely shallow, matching the same fix already applied in this stage
        # category (coexpression/hdwgcna_method.py): `adata.copy()` duplicates
        # every layer and obsm entry just to change which matrix is X, and then
        # write_h5ad serializes all of it to a file the GRN script never reads --
        # verified directly against pyscenic_scripts/pyscenic_grn.py, which only
        # ever reads /obs and /var axis names plus the requested matrix.
        if layer and layer != "X" and layer in adata.layers:
            write_h5ad(
                ad.AnnData(
                    X=adata.layers[layer],
                    obs=adata.obs.copy(),
                    var=adata.var.copy(),
                ),
                h5ad,
            )
        else:
            write_h5ad(adata, h5ad)

        out_dir = Path(getattr(context.paths, "results", ".")) / "grn"
        out_dir.mkdir(parents=True, exist_ok=True)
        # figures_dir (figures/, not results/) was already created in step 3
        # so the radial GRN figure could be rendered before any pySCENIC
        # guard; out_dir here stays results-rooted because it's also the
        # pySCENIC subprocess's own --out-dir for its raw data outputs
        # (regulons/adjacencies/loom/AUC).
        tag = "grn"

        # 6. Run grn -> ctx in-env script
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
            return self._skip_or_partial(
                adata,
                "grn script execution failed or timed out",
                figs=figs,
                warnings=warnings,
                error=str(exc)[:500],
            )
        if proc.returncode != 0:
            return self._skip_or_partial(
                adata,
                "grn/ctx script failed",
                figs=figs,
                warnings=warnings,
                stderr=proc.stderr.strip()[:500],
            )

        regulons_csv = out_dir / f"scenic_regulons_{tag}.csv"
        adjacencies_tsv = out_dir / f"scenic_adjacencies_{tag}.tsv"
        loom_path = out_dir / f"scenic_input_{tag}.loom"
        skip_marker = out_dir / f"grn_SKIPPED_{tag}.txt"
        if skip_marker.exists() or not regulons_csv.exists():
            reason = (
                skip_marker.read_text().strip() if skip_marker.exists() else "no regulons produced"
            )
            return self._skip_or_partial(adata, reason, figs=figs, warnings=warnings)
        try:
            regulons_df = load_regulons_csv(regulons_csv)
        except Exception as exc:
            return self._skip_or_partial(
                adata,
                "could not read regulons CSV",
                figs=figs,
                warnings=warnings,
                error=str(exc)[:500],
            )
        if len(regulons_df) == 0:
            return self._skip_or_partial(
                adata, "no regulons detected", figs=figs, warnings=warnings
            )

        # 7. AUCell
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
        auc = None
        try:
            aproc = backend.run_script(PYSCENIC_AUCELL_PY, aucell_args, timeout=timeout_seconds)
            if aproc.returncode == 0 and auc_parquet.exists():
                auc = pd.read_parquet(auc_parquet)
                if auc.empty:
                    auc = None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            # A warning, not a note: without AUC there are no regulon activities,
            # so every figure below is skipped too and the stage's real output is
            # the regulon table alone.
            warnings.append(f"AUCell skipped: {str(exc)[:200]}")

        # 7b. Regulon curvature: structural, needs only the regulon table
        # itself (not AUC), so it renders even when AUCell above failed.
        # Ricci curvature is computed ONCE and rendered twice (the
        # force-directed hairball, then the same radial layout the
        # DoRothEA figure above uses, colored by curvature instead of DE
        # direction) -- confirmed the expensive part is the curvature
        # computation, not either rendering step.
        def _curvature_figs() -> list[Path]:
            G, edge_df, node_df = compute_regulon_curvature_tables(
                regulons_df,
                top_k_targets=curvature_top_k_targets,
                ricci_alpha=curvature_ricci_alpha,
            )
            out = plot_regulon_curvature_from_tables(
                G, edge_df, node_df, figures_dir, plot_top_k=curvature_plot_top_k
            )
            if not edge_df.empty:
                radial_fig = rgf.radial_regulon_curvature(
                    edge_df,
                    node_df,
                    G,
                    n_tf=curvature_radial_n_tf,
                    per_tf=curvature_radial_per_tf,
                    n_sectors=curvature_radial_n_sectors,
                )
                out = out + figstyle.save_figure(
                    radial_fig, figures_dir, "regulon_curvature_radial"
                )
            return out

        render_figure("regulon curvature", _curvature_figs, figures=figs, warnings=warnings)

        # 8. Figures (in cellquorum env) when AUC available
        if auc is not None and auc.shape[1] > 0:
            groups = None
            if group_by in adata.obs.columns:
                groups = adata.obs[group_by].astype(str)
                groups.index = adata.obs_names
            common = auc.index.intersection(adata.obs_names)
            if groups is not None and len(common) > 0:
                figure_kwargs = {
                    "group_label": group_by,
                    "top_n": top_n,
                    "cluster_labels": cluster_labels,
                }
                figure_calls: tuple[Callable[..., Any], ...] = (
                    rf.plot_rss_panels,
                    rf.plot_regulon_clustermap,
                )
                for fn in figure_calls:
                    # partial binds fn now: closing over the loop variable would draw the
                    # last function twice.
                    render_figure(
                        fn.__name__,
                        partial(fn, auc, groups, figures_dir, **figure_kwargs),
                        figures=figs,
                        warnings=warnings,
                    )
                render_figure(
                    "cell clustermap",
                    lambda: rf.plot_regulon_cell_clustermap(
                        auc,
                        groups.to_frame(group_by),
                        figures_dir,
                        top_n=top_n,
                        cluster_labels=cluster_labels,
                        cluster_col=group_by,
                    ),
                    figures=figs,
                    warnings=warnings,
                )
            if "X_umap" in adata.obsm and len(common) > 0:

                def _regulon_umap() -> list[Path]:
                    umap = pd.DataFrame(
                        adata.obsm["X_umap"][:, :2],
                        index=adata.obs_names,
                        columns=["UMAP1", "UMAP2"],
                    )
                    return rf.plot_regulon_umap(auc, umap, figures_dir, groups=groups, top_n=12)

                render_figure("regulon-UMAP", _regulon_umap, figures=figs, warnings=warnings)

        # 9. Artifacts
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

        # 10. Metrics
        n_tfs = int(regulons_df["TF"].nunique()) if "TF" in regulons_df.columns else 0
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
            warnings=warnings,
            metrics=metrics,
            backend="pyscenic",
        )


__all__ = ["PyscenicMethod"]
