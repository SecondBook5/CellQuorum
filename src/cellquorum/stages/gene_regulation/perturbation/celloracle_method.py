"""In-silico transcription-factor knockout via CellOracle in an isolated env."""

from __future__ import annotations

import shutil
import signal
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from cellquorum.backends.celloracle_backend import CELLORACLE_KO_PY
from cellquorum.core.contracts import DataContract
from cellquorum.core.h5ad_io import write_h5ad
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.gene_regulation.perturbation import perturbation_figures as pfig
from cellquorum.visualization.figstyle import render_figure

_TAG = "perturbation"


class _PipelineStop(Exception):
    """A guard or failure that ends the stage with a recorded skip."""

    def __init__(self, reason: str, **details: object) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


@dataclass(frozen=True)
class _Settings:
    """Every config value ``CellOracleMethod._run`` reads, resolved once."""

    layer: str
    organism: str
    min_cells_total: int
    n_top_targets: int
    filter_edge_number: int
    knn_n_neighbors: int
    max_cells: int
    sampled_fraction: float
    filter_p: float
    n_propagation: int
    seed: int
    launcher: str
    timeout_seconds: int
    condition_key: str | None
    healthy_label: str | None
    tf_list: list[str] | None
    cluster_labels: dict | None
    n_top_genes: int
    cluster_key: str
    rep_key: str
    embedding_key: str

    @classmethod
    def from_config(cls, config: dict, adata) -> _Settings:  # noqa: ANN001
        # Generic keys fall back to what the object carries.
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
        return cls(
            layer=config.get("layer", "counts"),
            organism=config.get("organism", "human"),
            min_cells_total=int(config.get("min_cells_total", 200)),
            n_top_targets=int(config.get("n_top_targets", 20)),
            filter_edge_number=int(config.get("filter_edge_number", 10000)),
            knn_n_neighbors=int(config.get("knn_n_neighbors", 200)),
            max_cells=int(config.get("max_cells", 0)),
            sampled_fraction=float(config.get("sampled_fraction", 1.0)),
            filter_p=float(config.get("filter_p", 0.001)),
            n_propagation=int(config.get("n_propagation", 3)),
            seed=int(config.get("seed", 0)),
            launcher=config.get("launcher", "micromamba"),
            timeout_seconds=int(config.get("timeout_seconds", 10800)),
            condition_key=config.get("condition_key"),
            healthy_label=config.get("healthy_label"),
            tf_list=config.get("tf_list"),
            cluster_labels=config.get("cluster_labels"),
            n_top_genes=int(config.get("n_top_genes", 2000)),
            cluster_key=cluster_key,
            rep_key=rep_key,
            embedding_key=config.get("embedding_key") or "X_umap",
        )


def _check_prerequisites(adata, settings: _Settings, context) -> Any:  # noqa: ANN001
    """Return the CellOracle backend, or raise ``_PipelineStop`` naming what is missing."""
    if adata.n_obs < settings.min_cells_total:
        raise _PipelineStop(
            f"too few cells ({adata.n_obs} < {settings.min_cells_total})", n_obs=int(adata.n_obs)
        )
    if shutil.which(settings.launcher) is None:
        raise _PipelineStop(
            f"launcher '{settings.launcher}' not found on PATH", launcher=settings.launcher
        )
    registry = getattr(context, "backend_registry", None)
    backend = None
    if registry is not None:
        try:
            backend = registry.get("celloracle")
        except Exception:
            backend = None
    if backend is None:
        raise _PipelineStop("celloracle backend unavailable")
    try:
        module_ok = backend._py_module_available("celloracle")
    except Exception:
        module_ok = False
    if not module_ok:
        raise _PipelineStop("celloracle module unavailable in env")
    return backend


def _write_scratch_input(adata, settings: _Settings, context) -> Path:  # noqa: ANN001
    """Write the counts h5ad (highly-variable genes only) the KO script reads."""
    scratch = Path(getattr(context.paths, "scratch", "."))
    scratch.mkdir(parents=True, exist_ok=True)
    h5ad = scratch / "perturbation_input.h5ad"
    layer = settings.layer
    # Genuinely shallow, matching the same fix already applied in this stage
    # category (grn/pyscenic_method.py, coexpression/hdwgcna_method.py):
    # adata.copy() duplicates every LAYER (each a full gene-space matrix) just
    # to change which one is X, then write_h5ad serializes all of it. Unlike
    # those two methods' scripts, celloracle_ko.py DOES read
    # adata.obsm[embedding_key] (verified directly), so obsm is preserved here
    # -- it is cheap (a handful of dimensions per cell) next to the layers,
    # which are the actual multi-GB cost.
    if layer and layer != "X" and layer in adata.layers:
        work = ad.AnnData(
            X=adata.layers[layer],
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            obsm=dict(adata.obsm),
        )
    else:
        work = adata

    # Restrict to highly-variable genes before CellOracle ever sees the
    # object. CellOracle's own perform_PCA() runs a dense SVD over
    # whatever gene panel it is handed; on a real ~33k-gene, ~70k-cell
    # object that overflows LAPACK's 32-bit dense-indexing ceiling
    # ("Indexing a matrix... would incur an in integer overflow in
    # LAPACK") -- confirmed live. flavor="seurat_v3" operates directly
    # on raw counts (work.X here), matching what this layer already is.
    if work.n_vars > settings.n_top_genes:
        hvg = sc.pp.highly_variable_genes(
            work, n_top_genes=settings.n_top_genes, flavor="seurat_v3", inplace=False
        )
        work = work[:, hvg["highly_variable"].to_numpy()].copy()

    write_h5ad(work, h5ad)
    return h5ad


def _prepare_output_dirs(context) -> tuple[Path, Path]:  # noqa: ANN001
    """Create the results and figures directories and clear stale outcome markers."""
    out_dir = Path(getattr(context.paths, "results", ".")) / "perturbation"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Figures belong under figures/, not results/ -- out_dir above stays
    # results-rooted because it's also the KO subprocess's own --out-dir
    # for its raw data outputs (ranking/grn_summary/shift_vectors).
    figures_dir = Path(getattr(context.paths, "figures", ".")) / "perturbation"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # A marker left in this directory by an earlier attempt (a resumed run reuses it) says
    # nothing about this one; the script writes a fresh marker if it skips or fails.
    for stale in (
        out_dir / f"perturbation_SKIPPED_{_TAG}.txt",
        out_dir / f"perturbation_FAILED_{_TAG}.txt",
    ):
        stale.unlink(missing_ok=True)
    return out_dir, figures_dir


def _ko_args(h5ad: Path, out_dir: Path, settings: _Settings) -> list[str]:
    return [
        "--h5ad",
        str(h5ad),
        "--out-dir",
        str(out_dir),
        "--tag",
        _TAG,
        "--organism",
        str(settings.organism),
        "--cluster-key",
        str(settings.cluster_key),
        "--rep-key",
        str(settings.rep_key),
        "--embedding-key",
        str(settings.embedding_key),
        "--condition-key",
        str(settings.condition_key or ""),
        "--healthy-label",
        str(settings.healthy_label or ""),
        "--tf-list",
        " ".join(settings.tf_list) if settings.tf_list else "",
        "--n-top-targets",
        str(settings.n_top_targets),
        "--filter-edge-number",
        str(settings.filter_edge_number),
        "--knn-n-neighbors",
        str(settings.knn_n_neighbors),
        "--max-cells",
        str(settings.max_cells),
        "--sampled-fraction",
        str(settings.sampled_fraction),
        "--filter-p",
        str(settings.filter_p),
        "--n-propagation",
        str(settings.n_propagation),
        "--seed",
        str(settings.seed),
    ]


def _run_knockout(backend, h5ad: Path, out_dir: Path, settings: _Settings) -> pd.DataFrame:  # noqa: ANN001
    """Run the in-env KO script and load its ranking, or raise ``_PipelineStop``."""
    try:
        proc = backend.run_script(
            CELLORACLE_KO_PY, _ko_args(h5ad, out_dir, settings), timeout=settings.timeout_seconds
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise _PipelineStop(
            "KO script execution failed or timed out", error=str(exc)[:500]
        ) from exc
    if proc.returncode != 0:
        rc = proc.returncode
        if rc < 0:
            try:
                sig_name = signal.Signals(-rc).name
            except ValueError:
                sig_name = f"signal {-rc}"
            reason = f"KO script killed by {sig_name} (signal {-rc}) -- likely OOM"
        else:
            reason = "KO script failed"
        raise _PipelineStop(
            reason,
            returncode=rc,
            stderr=str(getattr(proc, "stderr", "")).strip()[:500],
        )

    ranking_csv = out_dir / "perturbation_ranking.csv"
    failed_marker = out_dir / f"perturbation_FAILED_{_TAG}.txt"
    skip_marker = out_dir / f"perturbation_SKIPPED_{_TAG}.txt"
    if failed_marker.exists():
        raise _PipelineStop(failed_marker.read_text().strip()[:300])
    if skip_marker.exists():
        raise _PipelineStop(skip_marker.read_text().strip()[:300])
    if not ranking_csv.exists():
        raise _PipelineStop("no ranking produced")

    try:
        return pd.read_csv(ranking_csv)
    except Exception as exc:
        raise _PipelineStop("could not read ranking CSV", error=str(exc)[:500]) from exc


def _render_figures(
    adata,  # noqa: ANN001
    ranking: pd.DataFrame,
    out_dir: Path,
    figures_dir: Path,
    settings: _Settings,
    figs: list[Path],
    warnings: list[str],
) -> None:
    """Draw the ranking, connectivity and shift figures; one failure never sinks the rest."""
    n_top = settings.n_top_targets
    render_figure(
        "target-ranking",
        lambda: pfig.plot_target_ranking(ranking, figures_dir, n_top=n_top),
        figures=figs,
        warnings=warnings,
    )
    grn_summary = out_dir / "grn_summary.csv"
    if grn_summary.exists():
        render_figure(
            "grn-connectivity",
            lambda: pfig.plot_grn_connectivity(pd.read_csv(grn_summary), figures_dir, n_top=n_top),
            figures=figs,
            warnings=warnings,
        )
    # shift-field for the top TF, if its shift vectors + embedding are present
    if settings.embedding_key in adata.obsm:
        _render_shift_figures(adata, ranking, out_dir, figures_dir, settings, figs, warnings)


def _render_shift_figures(
    adata,  # noqa: ANN001
    ranking: pd.DataFrame,
    out_dir: Path,
    figures_dir: Path,
    settings: _Settings,
    figs: list[Path],
    warnings: list[str],
) -> None:
    """Shift field, gridded field and fate summary for the top-ranked TF."""
    top_tf = str(ranking.sort_values("score", ascending=False).iloc[0]["tf"])
    shift_pq = out_dir / f"shift_vectors_{top_tf}.parquet"
    if not shift_pq.exists():
        return
    # Read outside render_figure: the parquet is also what the figures below gate
    # on, so a read failure has to be distinguishable from a draw failure.
    try:
        shift_df = pd.read_parquet(shift_pq)
    except Exception as exc:  # noqa: BLE001 — skip-not-crash
        warnings.append(f"could not read {shift_pq.name}: {str(exc)[:150]}")
        return

    render_figure(
        "shift-field",
        lambda: _draw_shift_figure(
            pfig.plot_ko_shift_field, shift_df, adata, figures_dir, top_tf, settings
        ),
        figures=figs,
        warnings=warnings,
    )
    # Gridded vector field (CellOracle-style) — the publication view
    render_figure(
        "shift-grid",
        lambda: _draw_shift_figure(
            pfig.plot_ko_shift_grid, shift_df, adata, figures_dir, top_tf, settings
        ),
        figures=figs,
        warnings=warnings,
    )
    # Fate summary: per-cluster mean shift magnitude (direction-agnostic)
    if settings.cluster_key in adata.obs.columns:
        render_figure(
            "fate-summary",
            lambda: _draw_fate_summary(shift_df, adata, figures_dir, top_tf, settings),
            figures=figs,
            warnings=warnings,
        )


def _umap_palette(adata, settings: _Settings) -> dict[str, str] | None:  # noqa: ANN001
    """A state keeps the colour the UMAP figures gave it."""
    if settings.cluster_key not in adata.obs.columns:
        return None
    from cellquorum.stages.integration.embeddings.plots import column_palette

    return column_palette(
        np.asarray(adata.obsm[settings.embedding_key])[:, :2], adata.obs[settings.cluster_key]
    )


def _draw_shift_figure(
    plot_fn: Callable[..., list[Path] | None],
    shift_df: pd.DataFrame,
    adata,  # noqa: ANN001
    figures_dir: Path,
    top_tf: str,
    settings: _Settings,
) -> list[Path] | None:
    """Draw one shift-vector figure (field or grid) on the stage's embedding."""
    emb = pd.DataFrame(
        adata.obsm[settings.embedding_key][:, :2],
        index=adata.obs_names,
        columns=["DIM1", "DIM2"],
    )
    groups = (
        adata.obs[settings.cluster_key].astype(str)
        if settings.cluster_key in adata.obs.columns
        else None
    )
    return plot_fn(
        shift_df,
        emb,
        figures_dir,
        tf=top_tf,
        groups=groups,
        group_labels=settings.cluster_labels,
        palette=_umap_palette(adata, settings),
    )


def _draw_fate_summary(
    shift_df: pd.DataFrame,
    adata,  # noqa: ANN001
    figures_dir: Path,
    top_tf: str,
    settings: _Settings,
) -> list[Path] | None:
    common = shift_df.index.intersection(adata.obs_names)
    if len(common) == 0:
        return None
    mags = np.linalg.norm(shift_df.loc[common].iloc[:, :2].to_numpy(), axis=1)
    fate_df = (
        pd.DataFrame(
            {
                "cluster": adata.obs.loc[common, settings.cluster_key].astype(str).to_numpy(),
                "delta": mags,
            }
        )
        .groupby("cluster")["delta"]
        .mean()
        .reset_index()
    )
    return pfig.plot_ko_fate_summary(fate_df, figures_dir, tf=top_tf)


def _collect_artifacts(out_dir: Path, figs: list[Path]) -> list[StageArtifact]:
    artifacts: list[StageArtifact] = [
        StageArtifact(
            name="ranking",
            path=out_dir / "perturbation_ranking.csv",
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
    return artifacts


def _build_metrics(adata, ranking: pd.DataFrame, settings: _Settings) -> dict[str, Any]:  # noqa: ANN001
    condition_scored = bool(
        "direction" in ranking.columns and (ranking["direction"] == "directional").any()
    )
    return {
        "n_tfs_screened": int(len(ranking)),
        "n_top_targets": int(min(settings.n_top_targets, len(ranking))),
        "condition_scored": condition_scored,
        "cluster_key": settings.cluster_key,
        "n_obs": int(adata.n_obs),
    }


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
        settings = _Settings.from_config(config, adata)
        try:
            backend = _check_prerequisites(adata, settings, context)
            h5ad = _write_scratch_input(adata, settings, context)
            out_dir, figures_dir = _prepare_output_dirs(context)
            ranking = _run_knockout(backend, h5ad, out_dir, settings)
        except _PipelineStop as stop:
            return self._skip(stop.reason, **stop.details)

        # Figures (in cellquorum env) — never let one failure sink the stage
        notes: list[str] = []
        warnings: list[str] = []
        figs: list[Path] = []
        if len(ranking) > 0:
            _render_figures(adata, ranking, out_dir, figures_dir, settings, figs, warnings)

        return StageResult(
            adata=adata,
            artifacts=_collect_artifacts(out_dir, figs),
            notes=notes,
            warnings=warnings,
            metrics=_build_metrics(adata, ranking, settings),
            backend="celloracle",
        )


__all__ = ["CellOracleMethod"]
