"""Classic pySCENIC gene-regulatory network inference method."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
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

_TAG = "grn"


class _PipelineStop(Exception):
    """A guard or failure that ends the pySCENIC pipeline (not the radial figure)."""

    def __init__(self, reason: str, **details: object) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


@dataclass(frozen=True)
class _Settings:
    """Every config value ``PyscenicMethod._run`` reads, resolved once."""

    layer: str
    group_by_cfg: str | None
    organism: str
    num_workers: int
    max_cells: int
    min_cells_total: int
    top_n: int
    seed: int
    launcher: str
    timeout_seconds: int
    tfs_path: str | None
    motifs_path: str | None
    rankings_glob: str | None
    curvature_top_k_targets: int
    curvature_ricci_alpha: float
    curvature_plot_top_k: int
    curvature_radial_n_tf: int
    curvature_radial_per_tf: int
    curvature_radial_n_sectors: int
    cluster_labels: dict
    radial_enabled: bool
    radial_collection: str
    radial_de_results_filename: str
    radial_min_size: int
    radial_n_tf: int
    radial_per_tf: int
    radial_n_sectors: int
    radial_case_color: str | None
    radial_control_color: str | None
    case: str | None
    control: str | None

    @classmethod
    def from_config(cls, config: dict, context) -> _Settings:  # noqa: ANN001
        return cls(
            layer=config.get("layer", "counts"),
            group_by_cfg=config.get("group_by"),
            organism=config.get("organism", "human"),
            # Unset inherits compute.n_jobs (see GrnConfig.num_workers): this count
            # goes to a dask LocalCluster inside the pySCENIC env, so asking for
            # more workers than the run was configured for costs memory, not just
            # cores.
            num_workers=resolve_n_jobs(context, config.get("num_workers")),
            max_cells=int(config.get("max_cells", 20000)),
            min_cells_total=int(config.get("min_cells_total", 200)),
            top_n=int(config.get("top_n", 5)),
            seed=int(config.get("seed", 0)),
            launcher=config.get("launcher", "micromamba"),
            timeout_seconds=int(config.get("timeout_seconds", 7200)),
            tfs_path=config.get("tfs_path"),
            motifs_path=config.get("motifs_path"),
            rankings_glob=config.get("rankings_glob"),
            curvature_top_k_targets=int(config.get("curvature_top_k_targets", 15)),
            curvature_ricci_alpha=float(config.get("curvature_ricci_alpha", 0.5)),
            curvature_plot_top_k=int(config.get("curvature_plot_top_k", 30)),
            curvature_radial_n_tf=int(config.get("curvature_radial_n_tf", 12)),
            curvature_radial_per_tf=int(config.get("curvature_radial_per_tf", 6)),
            curvature_radial_n_sectors=int(config.get("curvature_radial_n_sectors", 4)),
            cluster_labels=config.get("cluster_labels") or {},
            radial_enabled=bool(config.get("radial_enabled", True)),
            radial_collection=config.get("radial_collection", "dorothea"),
            radial_de_results_filename=config.get(
                "radial_de_results_filename", "de_pseudobulk_edger.csv"
            ),
            radial_min_size=int(config.get("radial_min_size", 5)),
            radial_n_tf=int(config.get("radial_n_tf", 12)),
            radial_per_tf=int(config.get("radial_per_tf", 6)),
            radial_n_sectors=int(config.get("radial_n_sectors", 4)),
            radial_case_color=config.get("radial_case_color"),
            radial_control_color=config.get("radial_control_color"),
            case=config.get("case"),
            control=config.get("control"),
        )


@dataclass(frozen=True)
class _Outputs:
    """Where the scratch input and the pySCENIC scripts' outputs live."""

    h5ad: Path
    out_dir: Path
    regulons_csv: Path
    adjacencies_tsv: Path
    loom_path: Path
    skip_marker: Path
    auc_parquet: Path

    @classmethod
    def under(cls, scratch: Path, out_dir: Path) -> _Outputs:
        return cls(
            h5ad=scratch / "grn_input.h5ad",
            out_dir=out_dir,
            regulons_csv=out_dir / f"scenic_regulons_{_TAG}.csv",
            adjacencies_tsv=out_dir / f"scenic_adjacencies_{_TAG}.tsv",
            loom_path=out_dir / f"scenic_input_{_TAG}.loom",
            skip_marker=out_dir / f"grn_SKIPPED_{_TAG}.txt",
            auc_parquet=out_dir / f"scenic_auc_mtx_{_TAG}.parquet",
        )


def _missing_cistarget_resources(settings: _Settings) -> list[str]:
    """Names of the cisTarget settings that are unset or point at nothing."""
    rankings_glob = settings.rankings_glob
    rankings_files = sorted(glob.glob(rankings_glob)) if rankings_glob else []
    if not rankings_files and rankings_glob and os.path.exists(rankings_glob):
        rankings_files = [rankings_glob]
    missing = []
    if not settings.tfs_path or not os.path.exists(settings.tfs_path):
        missing.append("tfs_path")
    if not settings.motifs_path or not os.path.exists(settings.motifs_path):
        missing.append("motifs_path")
    if not rankings_files:
        missing.append("rankings_glob")
    return missing


def _check_prerequisites(adata, settings: _Settings, context) -> Any:  # noqa: ANN001
    """Return the pySCENIC backend, or raise ``_PipelineStop`` naming what is missing."""
    if adata.n_obs < settings.min_cells_total:
        raise _PipelineStop(
            f"too few cells ({adata.n_obs} < {settings.min_cells_total})",
            n_obs=int(adata.n_obs),
        )
    if shutil.which(settings.launcher) is None:
        raise _PipelineStop(
            f"launcher '{settings.launcher}' not found on PATH", launcher=settings.launcher
        )
    registry = getattr(context, "backend_registry", None)
    backend = None
    if registry is not None:
        try:
            backend = registry.get("pyscenic")
        except Exception:
            backend = None
    if backend is None:
        raise _PipelineStop("pyscenic backend unavailable")
    try:
        module_ok = backend._py_module_available("pyscenic")
    except Exception:
        module_ok = False
    if not module_ok:
        raise _PipelineStop("pyscenic module unavailable in env")
    # cisTarget resources: unset OR unresolvable -> skip before spawning subprocess.
    missing_db = _missing_cistarget_resources(settings)
    if missing_db:
        raise _PipelineStop(
            "cisTarget resources unset/missing "
            f"({', '.join(missing_db)}); download the cisTarget DBs and set "
            "grn.tfs_path / motifs_path / rankings_glob",
            missing=missing_db,
        )
    return backend


def _write_scratch_input(adata, settings: _Settings, context) -> _Outputs:  # noqa: ANN001
    """Write the counts h5ad the GRN script reads and create the output directories."""
    scratch = Path(getattr(context.paths, "scratch", "."))
    scratch.mkdir(parents=True, exist_ok=True)
    # figures/ was created before any pySCENIC guard so the radial GRN figure
    # could be rendered first; out_dir stays results-rooted because it is also
    # the pySCENIC subprocess's own --out-dir for its raw data outputs
    # (regulons/adjacencies/loom/AUC).
    out_dir = Path(getattr(context.paths, "results", ".")) / "grn"
    outputs = _Outputs.under(scratch, out_dir)
    # Through the shared writer: this hands the working object to another
    # process, so a column h5py cannot encode would fail the stage here rather
    # than in pySCENIC. See cellquorum.core.h5ad_io.
    from cellquorum.core.h5ad_io import write_h5ad

    layer = settings.layer
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
            outputs.h5ad,
        )
    else:
        write_h5ad(adata, outputs.h5ad)
    out_dir.mkdir(parents=True, exist_ok=True)
    return outputs


def _infer_regulons(backend, outputs: _Outputs, settings: _Settings) -> pd.DataFrame:  # noqa: ANN001
    """Run the grn -> ctx script and load its regulon table, or raise ``_PipelineStop``."""
    grn_args = [
        "--h5ad",
        str(outputs.h5ad),
        "--tfs",
        str(settings.tfs_path),
        "--motifs",
        str(settings.motifs_path),
        "--rankings",
        str(settings.rankings_glob),
        "--out-dir",
        str(outputs.out_dir),
        "--tag",
        _TAG,
        "--num-workers",
        str(settings.num_workers),
        "--max-cells",
        str(settings.max_cells),
        "--layer",
        str(settings.layer),
        "--seed",
        str(settings.seed),
    ]
    try:
        proc = backend.run_script(PYSCENIC_GRN_PY, grn_args, timeout=settings.timeout_seconds)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise _PipelineStop(
            "grn script execution failed or timed out", error=str(exc)[:500]
        ) from exc
    if proc.returncode != 0:
        raise _PipelineStop("grn/ctx script failed", stderr=proc.stderr.strip()[:500])

    if outputs.skip_marker.exists() or not outputs.regulons_csv.exists():
        raise _PipelineStop(
            outputs.skip_marker.read_text().strip()
            if outputs.skip_marker.exists()
            else "no regulons produced"
        )
    try:
        regulons_df = load_regulons_csv(outputs.regulons_csv)
    except Exception as exc:
        raise _PipelineStop("could not read regulons CSV", error=str(exc)[:500]) from exc
    if len(regulons_df) == 0:
        raise _PipelineStop("no regulons detected")
    return regulons_df


def _run_aucell(
    backend,  # noqa: ANN001
    outputs: _Outputs,
    settings: _Settings,
    warnings: list[str],
) -> pd.DataFrame | None:
    """Score regulon activity per cell; ``None`` when AUCell produced nothing usable."""
    aucell_args = [
        "--loom",
        str(outputs.loom_path),
        "--regulons",
        str(outputs.regulons_csv),
        "--out",
        str(outputs.auc_parquet),
        "--num-workers",
        str(settings.num_workers),
        "--seed",
        str(settings.seed),
    ]
    auc = None
    try:
        aproc = backend.run_script(
            PYSCENIC_AUCELL_PY, aucell_args, timeout=settings.timeout_seconds
        )
        if aproc.returncode == 0 and outputs.auc_parquet.exists():
            auc = pd.read_parquet(outputs.auc_parquet)
            if auc.empty:
                auc = None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        # A warning, not a note: without AUC there are no regulon activities,
        # so every figure below is skipped too and the stage's real output is
        # the regulon table alone.
        warnings.append(f"AUCell skipped: {str(exc)[:200]}")
    return auc


def _render_curvature_figures(
    regulons_df: pd.DataFrame,
    settings: _Settings,
    figures_dir: Path,
    figs: list[Path],
    warnings: list[str],
) -> None:
    """Regulon curvature: structural, needs only the regulon table (not AUC).

    Ricci curvature is computed ONCE and rendered twice (the force-directed
    hairball, then the same radial layout the DoRothEA figure uses, colored by
    curvature instead of DE direction) -- confirmed the expensive part is the
    curvature computation, not either rendering step.
    """

    def _curvature_figs() -> list[Path]:
        G, edge_df, node_df = compute_regulon_curvature_tables(
            regulons_df,
            top_k_targets=settings.curvature_top_k_targets,
            ricci_alpha=settings.curvature_ricci_alpha,
        )
        out = plot_regulon_curvature_from_tables(
            G, edge_df, node_df, figures_dir, plot_top_k=settings.curvature_plot_top_k
        )
        if not edge_df.empty:
            radial_fig = rgf.radial_regulon_curvature(
                edge_df,
                node_df,
                G,
                n_tf=settings.curvature_radial_n_tf,
                per_tf=settings.curvature_radial_per_tf,
                n_sectors=settings.curvature_radial_n_sectors,
            )
            out = out + figstyle.save_figure(radial_fig, figures_dir, "regulon_curvature_radial")
        return out

    render_figure("regulon curvature", _curvature_figs, figures=figs, warnings=warnings)


def _render_activity_figures(
    adata,  # noqa: ANN001
    auc: pd.DataFrame,
    group_by: str,
    settings: _Settings,
    figures_dir: Path,
    figs: list[Path],
    warnings: list[str],
) -> None:
    """Figures drawn from the AUCell matrix (run in the cellquorum env)."""
    top_n = settings.top_n
    cluster_labels = settings.cluster_labels
    groups = None
    if group_by in adata.obs.columns:
        groups = adata.obs[group_by].astype(str)
        groups.index = adata.obs_names
    elif settings.group_by_cfg:
        # A configured grouping that is not a column is a typo; without this the group-level
        # figures just vanish and the run looks complete.
        warnings.append(
            f"group_by column '{settings.group_by_cfg}' absent from obs; the group-level "
            f"regulon figures were not drawn"
        )
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


def _collect_artifacts(outputs: _Outputs, figs: list[Path]) -> list[StageArtifact]:
    artifacts: list[StageArtifact] = [
        StageArtifact(
            name="regulons",
            path=outputs.regulons_csv,
            kind="csv",
            description="pySCENIC ctx regulons (TF->target enrichments)",
        ),
    ]
    if outputs.adjacencies_tsv.exists():
        artifacts.append(
            StageArtifact(
                name="adjacencies",
                path=outputs.adjacencies_tsv,
                kind="tsv",
                description="GRNBoost2 TF-target adjacencies",
            )
        )
    if outputs.auc_parquet.exists():
        artifacts.append(
            StageArtifact(
                name="auc_mtx",
                path=outputs.auc_parquet,
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
    return artifacts


def _build_metrics(
    adata,  # noqa: ANN001
    regulons_df: pd.DataFrame,
    auc: pd.DataFrame | None,
    group_by: str,
) -> dict[str, Any]:
    n_tfs = int(regulons_df["TF"].nunique()) if "TF" in regulons_df.columns else 0
    return {
        "n_regulons": int(auc.shape[1]) if auc is not None else int(len(regulons_df)),
        "n_tfs": n_tfs,
        "n_cells_scored": int(auc.shape[0]) if auc is not None else 0,
        "group_by": group_by,
        "n_obs": int(adata.n_obs),
    }


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
        settings = _Settings.from_config(config, context)
        group_by = resolve_group_by(adata, settings.group_by_cfg, default="all")

        figures_dir = Path(getattr(context.paths, "figures", ".")) / "grn"
        figures_dir.mkdir(parents=True, exist_ok=True)
        figs: list[Path] = []
        warnings: list[str] = []

        # Radial DoRothEA/CollecTRI GRN figure: computed FIRST and kept
        # independent of every pySCENIC-specific guard below -- it needs
        # only the DE stage's own CSV and a curated prior net, never
        # GRNBoost2/cisTarget/AUCell, so a run with no cisTarget DBs
        # configured (a real, common case) must not also silently lose
        # this figure just because it lives in the same method.
        if settings.radial_enabled:
            self._render_radial_grn(
                context=context,
                figures_dir=figures_dir,
                radial_collection=settings.radial_collection,
                radial_de_results_filename=settings.radial_de_results_filename,
                radial_min_size=settings.radial_min_size,
                radial_n_tf=settings.radial_n_tf,
                radial_per_tf=settings.radial_per_tf,
                radial_n_sectors=settings.radial_n_sectors,
                radial_case_color=settings.radial_case_color,
                radial_control_color=settings.radial_control_color,
                organism=settings.organism,
                group_by=group_by,
                case=settings.case,
                control=settings.control,
                figs=figs,
                warnings=warnings,
            )

        # Every stop below goes through _skip_or_partial, so a plain MethodSkip
        # is returned only when the radial figure above produced nothing.
        try:
            backend = _check_prerequisites(adata, settings, context)
            outputs = _write_scratch_input(adata, settings, context)
            regulons_df = _infer_regulons(backend, outputs, settings)
        except _PipelineStop as stop:
            return self._skip_or_partial(
                adata, stop.reason, figs=figs, warnings=warnings, **stop.details
            )

        auc = _run_aucell(backend, outputs, settings, warnings)

        _render_curvature_figures(regulons_df, settings, figures_dir, figs, warnings)
        if auc is not None and auc.shape[1] > 0:
            _render_activity_figures(adata, auc, group_by, settings, figures_dir, figs, warnings)

        return StageResult(
            adata=adata,
            artifacts=_collect_artifacts(outputs, figs),
            warnings=warnings,
            metrics=_build_metrics(adata, regulons_df, auc, group_by),
            backend="pyscenic",
        )


__all__ = ["PyscenicMethod"]
