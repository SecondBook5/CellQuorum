"""Case-minus-control density figure on an embedding, one file per stratum.

Opt-in: name ``density_difference_umap`` in ``embeddings.methods`` (setting that list replaces the
default one, so keep the compute and render methods you still want). The figure itself, and why
every specimen is weighted equally, is in
:mod:`cellquorum.stages.integration.embeddings.density_difference`.

Method keys (all optional unless the design block cannot supply them):
    sample_col: obs column identifying the specimen. Falls back to the factorial design's
        ``sample_col``, then the design's ``donor_col``.
    condition_col, case, control: fall back to the design block.
    stratify_by: obs column; one figure per level (a tissue, say). Empty draws one figure.
    label_column: obs column whose groups are named at their median position.
    basis: ``umap`` (default) or ``phate``.
    bandwidth, grid_size, min_cells_per_specimen, max_cells_per_specimen, mask_fraction,
    min_cells_per_stratum, cmap, figure_formats, dpi.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.integration.embeddings.density_difference import (
    DensityDifferenceError,
    density_difference,
    plot_density_difference,
    strata,
)
from cellquorum.stages.integration.embeddings.methods import _seed, _view_percentile
from cellquorum.stages.integration.embeddings.plots import EMBEDDING_REGISTRY
from cellquorum.visualization.figio import figure_artifacts
from cellquorum.visualization.figstyle import apply_theme, save_figure

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


def _slug(text: str) -> str:
    """A filename-safe form of a stratum name."""
    return re.sub(r"[^0-9A-Za-z_.-]", "_", text)


class DensityDifferenceUmapMethod(AnalysisMethod):
    """Render the specimen-weighted case-minus-control cell density on an embedding."""

    name = "density_difference_umap"
    stage_category = "embeddings"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        design = getattr(getattr(context, "config", None), "design", None)
        condition_col = (
            config.get("condition_col") or getattr(design, "condition_col", None) or "condition"
        )
        case = config.get("case") or getattr(design, "case", None)
        control = config.get("control") or getattr(design, "control", None)
        if not case or not control:
            return self._skip("case and control are not resolvable (no design block, none given)")
        factorial = getattr(design, "factorial", None)
        sample_col = (
            config.get("sample_col")
            or getattr(factorial, "sample_col", None)
            or getattr(design, "donor_col", None)
        )
        stratify_by = config.get("stratify_by") or None
        label_column = config.get("label_column") or None
        required = [c for c in (condition_col, sample_col, stratify_by, label_column) if c]
        missing = [c for c in required if c not in adata.obs.columns]
        if not sample_col or missing:
            return self._skip(f"obs column(s) not found: {missing or ['sample_col']}")
        basis = config.get("basis", "umap")
        spec = EMBEDDING_REGISTRY.get(basis)
        if spec is None or spec["obsm"] not in adata.obsm:
            return self._skip(f"embedding '{basis}' not present")

        xy = np.asarray(adata.obsm[spec["obsm"]])[:, :2]
        specimen = adata.obs[sample_col].astype(str).to_numpy()
        condition = adata.obs[condition_col].astype(str).to_numpy()
        stratum_cells = strata(
            adata.obs[stratify_by] if stratify_by else None,
            adata.n_obs,
            int(config.get("min_cells_per_stratum", 500)),
        )
        if not stratum_cells:
            return self._skip(f"no '{stratify_by}' level has enough cells")

        apply_theme("notebook")
        figures_dir = Path(context.paths.figures) / "embeddings"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        label_names = config.get("cluster_labels") or {}
        artifacts, notes, warnings, drawn = [], [], [], []
        for level, mask in stratum_cells:
            where = "" if level == "all" else f"{level}: "
            try:
                result = density_difference(
                    xy[mask],
                    specimen[mask],
                    condition[mask],
                    case=str(case),
                    control=str(control),
                    view_percentile=_view_percentile(config),
                    grid_size=int(config.get("grid_size", 220)),
                    bandwidth=float(config.get("bandwidth", 0.08)),
                    min_cells_per_specimen=int(config.get("min_cells_per_specimen", 50)),
                    max_cells_per_specimen=int(config.get("max_cells_per_specimen", 8000)),
                    seed=_seed(config, context),
                )
            except DensityDifferenceError as exc:
                warnings.append(f"density_difference_umap: {where}{exc}")
                continue
            for name, n in result.dropped:
                warnings.append(
                    f"density_difference_umap: {where}specimen '{name}' has {n} cells, below "
                    "the minimum, and is left out"
                )
            fig = plot_density_difference(
                xy[mask],
                result,
                title=f"{level}: {control} vs {case}" if level != "all" else f"{control} vs {case}",
                axis_labels=spec["axis"],
                case_label=str(case),
                control_label=str(control),
                mask_fraction=float(config.get("mask_fraction", 0.08)),
                labels=adata.obs[label_column][mask] if label_column else None,
                label_names=label_names,
                min_label_frac=float(config.get("min_label_frac", 0.001)),
                **({"cmap": config["cmap"]} if config.get("cmap") else {}),
            )
            stem = "density_difference_" + basis + ("" if level == "all" else f"_{_slug(level)}")
            paths = save_figure(fig, figures_dir, stem, formats=formats, dpi=dpi)
            artifacts += figure_artifacts(
                paths,
                name="density_difference_umap",
                description=(
                    f"{control} vs {case} cell density on the {basis}, each specimen weighted "
                    f"equally ({len(result.control_specimens)} v {len(result.case_specimens)} "
                    "specimens); descriptive."
                ),
            )
            drawn.append(level)
            notes.append(
                f"density_difference_umap: {where}{len(result.control_specimens)} {control} and "
                f"{len(result.case_specimens)} {case} specimens (column '{sample_col}')"
            )
        if not artifacts:
            return self._skip("no figures produced", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            warnings=warnings,
            metrics={"method": self.name, "strata": drawn, "sample_col": sample_col},
            backend="python",
        )


__all__ = ["DensityDifferenceUmapMethod"]
