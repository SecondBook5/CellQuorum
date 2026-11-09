"""hdWGCNA co-expression network analysis method."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import pandas as pd

from cellquorum.backends.hdwgcna_backend import HDWGCNA_R
from cellquorum.core.contracts import DataContract
from cellquorum.core.h5ad_io import write_h5ad
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.gene_regulation._shared import resolve_group_by
from cellquorum.stages.gene_regulation.coexpression.community_network import (
    diagnostic_odds_ratio,
    plot_community_network,
)
from cellquorum.stages.gene_regulation.coexpression.module_curvature import (
    plot_module_curvature,
)
from cellquorum.stages.gene_regulation.coexpression.module_umap_plot import (
    plot_module_umap,
)
from cellquorum.visualization.figstyle import render_figure, save_figure

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


class _PipelineStop(Exception):
    """A guard or failure that ends the method with a recorded skip."""

    def __init__(self, reason: str, **details: object) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


@dataclass(frozen=True)
class _Settings:
    """Every config value ``HdwgcnaMethod._run`` reads, resolved once."""

    layer: str
    group_by: str
    group_by_explicit: bool
    condition_col: str | None
    n_hvg: int
    k: int
    min_cells: int
    min_cells_total: int
    soft_power: Any
    seed: int
    launcher: str
    timeout_seconds: int
    curvature_top_k_per_module: int
    curvature_ricci_alpha: float
    edge_quantile: float
    community_network_max_edges: int
    community_network_group_col: str

    @classmethod
    def from_config(cls, config: dict, adata: ad.AnnData) -> _Settings:
        layer = config.get("layer", "counts")
        group_by_cfg = config.get("group_by")
        condition_col = config.get("condition_col")
        n_hvg = int(config.get("n_hvg", 3000))
        k = int(config.get("k", 25))
        min_cells = int(config.get("min_cells", 50))
        min_cells_total = int(config.get("min_cells_total", 100))
        soft_power = config.get("soft_power")
        seed = int(config.get("seed", 0))
        launcher = config.get("launcher", "micromamba")
        timeout_seconds = int(config.get("timeout_seconds", 3600))
        curvature_top_k_per_module = int(config.get("curvature_top_k_per_module", 15))
        curvature_ricci_alpha = float(config.get("curvature_ricci_alpha", 0.5))
        edge_quantile = float(config.get("edge_quantile", 0.993))
        community_network_max_edges = int(config.get("community_network_max_edges", 1500))
        community_network_group_col = config.get("community_network_group_col")

        # Resolve group_by: config OR first-present of (cell_type, leiden) OR "cell_type"
        group_by = resolve_group_by(adata, group_by_cfg, default="cell_type")
        return cls(
            layer=layer,
            group_by=group_by,
            group_by_explicit=bool(group_by_cfg),
            condition_col=condition_col,
            n_hvg=n_hvg,
            k=k,
            min_cells=min_cells,
            min_cells_total=min_cells_total,
            soft_power=soft_power,
            seed=seed,
            launcher=launcher,
            timeout_seconds=timeout_seconds,
            curvature_top_k_per_module=curvature_top_k_per_module,
            curvature_ricci_alpha=curvature_ricci_alpha,
            edge_quantile=edge_quantile,
            community_network_max_edges=community_network_max_edges,
            community_network_group_col=community_network_group_col or group_by,
        )


def _check_prerequisites(adata: ad.AnnData, settings: _Settings, context: PipelineContext) -> Any:
    """Return the hdWGCNA backend, or raise ``_PipelineStop`` naming what is missing."""
    if adata.n_obs < settings.min_cells_total:
        raise _PipelineStop(
            f"too few cells ({adata.n_obs} < {settings.min_cells_total})",
            n_obs=int(adata.n_obs),
            min_cells_total=settings.min_cells_total,
        )
    # A configured grouping that is not a column is a typo, and R would not say so: its script
    # falls back to one pooled group, so the run succeeds with modules built on all cells
    # together and nothing in the output says the grouping was ignored.
    if settings.group_by_explicit and settings.group_by not in adata.obs.columns:
        raise _PipelineStop(
            f"group_by column '{settings.group_by}' absent from obs", group_by=settings.group_by
        )
    if shutil.which(settings.launcher) is None:
        raise _PipelineStop(
            f"launcher '{settings.launcher}' not found on PATH", launcher=settings.launcher
        )
    registry = getattr(context, "backend_registry", None)
    backend = None
    if registry is not None:
        try:
            backend = registry.get("hdwgcna_r")
        except Exception:
            backend = None
    if backend is None:
        raise _PipelineStop("hdwgcna_r backend unavailable")
    if not backend._r_package_available("hdWGCNA"):
        raise _PipelineStop("hdWGCNA R package unavailable")
    return backend


def _write_scratch_input(adata: ad.AnnData, settings: _Settings, context: PipelineContext) -> Path:
    """Write the counts h5ad the R script reads."""
    scratch = Path(getattr(context.paths, "scratch", "."))
    scratch.mkdir(parents=True, exist_ok=True)
    h5ad = scratch / "coexpr_input.h5ad"
    layer = settings.layer

    # Genuinely shallow, which the old comment claimed and `adata.copy()` did not do. It
    # duplicated the whole cohort to change which matrix is .X, and then wrote every layer
    # and embedding into a file R has to read back -- paying for the duplication twice, once
    # in memory and once in what zellkonverter must parse.
    #
    # hdWGCNA needs the expression matrix, the gene names and the grouping columns in obs.
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
    return h5ad


def _prepare_output_dirs(context: PipelineContext) -> tuple[Path, Path]:
    out_dir = Path(getattr(context.paths, "results", ".")) / "coexpression"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Figures belong under figures/, not results/ -- out_dir above stays
    # results-rooted because it's also the R script's own output dir for
    # its raw data (modules.csv/eigengenes.csv/module_condition_corr.csv).
    figures_dir = Path(getattr(context.paths, "figures", ".")) / "coexpression"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, figures_dir


# The R script wraps each optional step (TOM edges, dendrogram, module networks, module UMAP,
# condition correlation) in its own tryCatch and only prints "[hdwgcna] <step> skipped: <why>" when
# one fails, so the stage still succeeds with that output missing. These lines are how the engine
# finds out.
_R_STEP_FAILURE = re.compile(r"^\[hdwgcna\] (?P<what>.*(?:skipped|failed).*)$", re.MULTILINE)


def _r_step_warnings(stderr: str) -> list[str]:
    """One warning per optional R step that failed, so a missing output is never silent."""
    return [
        f"hdWGCNA R step did not finish: {m.group('what').strip()}"
        for m in _R_STEP_FAILURE.finditer(stderr or "")
        if not m.group("what").startswith("SKIPPED")
    ]


def _run_hdwgcna(
    backend: Any, h5ad: Path, out_dir: Path, figures_dir: Path, settings: _Settings
) -> tuple[pd.DataFrame, list[str]]:
    """Run the R script and load the module table, or raise ``_PipelineStop``.

    Returns the module table and a warning for each optional R step that failed.
    """
    soft_power = settings.soft_power
    args = [
        str(h5ad),
        str(out_dir),
        settings.group_by,
        settings.condition_col or "condition",
        str(settings.n_hvg),
        str(settings.k),
        str(settings.min_cells),
        "NA" if soft_power is None else str(soft_power),
        str(settings.seed),
        str(figures_dir),
        str(settings.edge_quantile),
    ]
    try:
        proc = backend.run_script(HDWGCNA_R, args, timeout=settings.timeout_seconds)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise _PipelineStop("R script execution failed or timed out", error=str(exc)[:500]) from exc

    if proc.returncode != 0:
        raise _PipelineStop("hdWGCNA script failed", stderr=proc.stderr.strip()[:500])

    # Check sentinel files and module results
    modules_csv = out_dir / "modules.csv"
    if not modules_csv.exists():
        raise _PipelineStop("modules.csv not found")

    skip_file = out_dir / "hdwgcna_SKIPPED.txt"
    if skip_file.exists():
        skip_text = skip_file.read_text().strip()
        raise _PipelineStop(skip_text, skip_file_content=skip_text)

    try:
        modules_df = pd.read_csv(modules_csv)
    except Exception as exc:
        raise _PipelineStop("could not read modules.csv", error=str(exc)[:500]) from exc

    if len(modules_df) == 0:
        raise _PipelineStop("no modules detected")

    # Guard against missing 'module' column
    if "module" not in modules_df.columns:
        raise _PipelineStop("modules.csv missing 'module' column")
    return modules_df, _r_step_warnings(proc.stderr)


def _render_figures(
    adata: ad.AnnData,
    modules_df: pd.DataFrame,
    out_dir: Path,
    figures_dir: Path,
    settings: _Settings,
    figs: list[Path],
    warnings: list[str],
) -> None:
    """Module UMAP, module curvature and the community network; one failure never sinks the rest."""
    module_umap_csv = out_dir / "module_umap.csv"
    if module_umap_csv.exists():
        render_figure(
            "module UMAP",
            lambda: plot_module_umap(module_umap_csv, figures_dir),
            figures=figs,
            warnings=warnings,
        )
    render_figure(
        "module curvature",
        lambda: plot_module_curvature(
            modules_df,
            figures_dir,
            top_k_per_module=settings.curvature_top_k_per_module,
            ricci_alpha=settings.curvature_ricci_alpha,
        ),
        figures=figs,
        warnings=warnings,
    )
    _render_community_network(adata, modules_df, out_dir, figures_dir, settings, figs, warnings)


def _render_community_network(
    adata: ad.AnnData,
    modules_df: pd.DataFrame,
    out_dir: Path,
    figures_dir: Path,
    settings: _Settings,
    figs: list[Path],
    warnings: list[str],
) -> None:
    """Community network: needs real TOM edges (tom_edges.csv), not just module labels.

    modules.csv alone can only fake within-module cliques (module_curvature.py's proxy
    graph above). Marker scores (diagnostic_odds_ratio) are computed only over the genes
    actually drawn, not the full transcriptome.
    """
    tom_edges_csv = out_dir / "tom_edges.csv"
    if not tom_edges_csv.exists():
        return
    try:
        tom_edges_df = pd.read_csv(tom_edges_csv)
    except Exception as exc:
        warnings.append(
            f"community network skipped: could not read tom_edges.csv ({str(exc)[:150]})"
        )
        return
    if tom_edges_df.empty:
        return

    network_genes = set(tom_edges_df["gene1"]) | set(tom_edges_df["gene2"])
    present_genes = sorted(network_genes & set(adata.var_names))
    marker_log_dor = None
    if settings.community_network_group_col in adata.obs.columns and present_genes:
        try:
            marker_log_dor = diagnostic_odds_ratio(
                adata,
                group_col=settings.community_network_group_col,
                layer=settings.layer,
                genes=present_genes,
            )
        except Exception as exc:
            warnings.append(f"community network marker scores skipped: {str(exc)[:150]}")

    def _community_network_fig() -> list[Path]:
        fig = plot_community_network(
            tom_edges_df,
            modules_df,
            marker_log_dor,
            max_edges=settings.community_network_max_edges,
            seed=settings.seed,
            title="Coexpression community network",
        )
        return save_figure(fig, figures_dir, "coexpression_community_network")

    render_figure("community network", _community_network_fig, figures=figs, warnings=warnings)


def _collect_artifacts(out_dir: Path, figures_dir: Path, figs: list[Path]) -> list[StageArtifact]:
    """Register the tables and figures that exist (only for files that exist)."""
    artifacts = [
        StageArtifact(
            name="modules",
            path=out_dir / "modules.csv",
            kind="csv",
            description="Gene-to-module assignments from hdWGCNA",
        )
    ]

    optional_tables = (
        ("eigengenes", "eigengenes.csv", "Module eigengene expression values"),
        (
            "tom_edges",
            "tom_edges.csv",
            "Real TOM-thresholded coexpression edges (community network input)",
        ),
        (
            "module_umap_data",
            "module_umap.csv",
            "Module UMAP coordinates and hub gene annotations",
        ),
        (
            "module_condition_corr",
            "module_condition_corr.csv",
            "Module-condition correlation results",
        ),
    )
    for name, filename, description in optional_tables:
        path = out_dir / filename
        if path.exists():
            artifacts.append(
                StageArtifact(name=name, path=path, kind="csv", description=description)
            )

    # Add figure artifacts (module UMAP + module curvature network)
    for fig_path in figs:
        artifacts.append(
            StageArtifact(
                name=f"figure_{fig_path.stem}",
                path=fig_path,
                kind="figure",
                description="hdWGCNA co-expression figure",
            )
        )

    # hdWGCNA's own canonical figures, rendered directly by the R script
    # (not post-processed from a CSV like the two above) -- registered
    # here rather than via render_figure/figs since there's no Python
    # call to wrap, just files to discover after the script succeeded.
    dendrogram_pdf = figures_dir / "gene_dendrogram.pdf"
    if dendrogram_pdf.exists():
        artifacts.append(
            StageArtifact(
                name="figure_gene_dendrogram",
                path=dendrogram_pdf,
                kind="figure",
                description="hdWGCNA gene dendrogram with module color bar",
            )
        )
    module_networks_dir = figures_dir / "module_networks"
    if module_networks_dir.is_dir():
        for net_path in sorted(module_networks_dir.glob("*.pdf")):
            artifacts.append(
                StageArtifact(
                    name=f"figure_{net_path.stem}",
                    path=net_path,
                    kind="figure",
                    description="hdWGCNA per-module hub-gene network",
                )
            )
    return artifacts


def _build_metrics(adata: ad.AnnData, modules_df: pd.DataFrame, settings: _Settings) -> dict:
    # Exclude the unassigned "grey" bin (WGCNA/hdWGCNA label for genes not
    # placed in any real module) so counts reflect true co-expression modules.
    real_modules = modules_df[modules_df["module"].astype(str).str.lower() != "grey"]
    return {
        "n_modules": int(real_modules["module"].nunique()),
        "n_genes_assigned": int(len(real_modules)),
        "group_by": settings.group_by,
        "soft_power": settings.soft_power,
        "n_obs": int(adata.n_obs),
    }


class HdwgcnaMethod(AnalysisMethod):
    """Co-expression network analysis via hdWGCNA in an isolated R environment.

    hdWGCNA (hierarchical Weighted Gene Co-expression Network Analysis) identifies
    gene modules from scRNA-seq data. This method runs hdWGCNA in an isolated
    micromamba environment, produces module assignments, eigengenes, and a
    publication-grade module UMAP visualization.
    """

    name = "hdwgcna"
    stage_category = "coexpression"
    backend = "hdwgcna_r"

    def input_contract(self, config: dict) -> DataContract:
        """Require the counts layer for co-expression analysis.

        ``group_by`` is deliberately NOT a hard obs requirement: when the
        configured column is absent the R script falls back to a single "all"
        group (see hdwgcna.R), so requiring it here would hard-fail the
        contract instead of allowing that graceful fallback.
        """
        layer = config.get("layer", "counts")

        return DataContract(
            required_layers=[layer],
            required_obs=[],
            expression_layer=layer,
            expected_kind="counts",
        )

    def requires_obs(self, config: dict) -> list[str]:
        """Return obs columns required for co-expression analysis.

        Returns empty list to avoid over-constraining since group_by can fall back
        to obs-present columns or R "all" fallback.
        """
        # Don't hard-require columns that may be absent - the R script handles fallbacks
        return []

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        """Run hdWGCNA co-expression analysis."""
        settings = _Settings.from_config(config, adata)
        try:
            backend = _check_prerequisites(adata, settings, context)
            h5ad = _write_scratch_input(adata, settings, context)
            out_dir, figures_dir = _prepare_output_dirs(context)
            modules_df, r_warnings = _run_hdwgcna(backend, h5ad, out_dir, figures_dir, settings)
        except _PipelineStop as stop:
            return self._skip(stop.reason, **stop.details)

        figs: list[Path] = []
        warnings: list[str] = list(r_warnings)
        _render_figures(adata, modules_df, out_dir, figures_dir, settings, figs, warnings)

        return StageResult(
            adata=adata,
            artifacts=_collect_artifacts(out_dir, figures_dir, figs),
            warnings=warnings,
            metrics=_build_metrics(adata, modules_df, settings),
            backend="hdwgcna_r",
        )


__all__ = ["HdwgcnaMethod"]
