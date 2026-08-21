"""Volcano visualization method: renders the pseudobulk DE volcano."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.comparative.differential_expression.viz import plots
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.visualization import figstyle

_CSV_NAME = "de_pseudobulk_edger.csv"
_LOGFC_ALIASES = ("logFC", "log2FoldChange", "log2fc", "avg_log2FC")
_FDR_ALIASES = ("FDR", "padj", "adj_pvalue", "qvalue")
_GENE_ALIASES = ("gene", "gene_name", "symbol", "names")


def _first_present(columns: pd.Index, aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def load_de_table(results_dir: Path) -> pd.DataFrame | None:
    """Return a normalized (gene, logFC, FDR, ...) frame, or None if unusable."""
    results_dir = Path(results_dir)
    path = results_dir / _CSV_NAME
    if not path.exists():
        return None
    try:
        raw = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    if raw.empty:
        return None
    gene_col = _first_present(raw.columns, _GENE_ALIASES)
    fc_col = _first_present(raw.columns, _LOGFC_ALIASES)
    fdr_col = _first_present(raw.columns, _FDR_ALIASES)
    if gene_col is None or fc_col is None or fdr_col is None:
        return None
    out = raw.rename(columns={gene_col: "gene", fc_col: "logFC", fdr_col: "FDR"})
    # Reorder so gene/logFC/FDR lead; keep any extra columns (e.g. likely_ambient).
    lead = ["gene", "logFC", "FDR"]
    rest = [c for c in out.columns if c not in lead]
    out = out[lead + rest]
    if out.empty:
        return None
    return out


class VolcanoVizMethod(AnalysisMethod):
    """Render a pseudobulk volcano from the DE stage's CSV."""

    name = "volcano_viz"
    stage_category = "de_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        results_dir = Path(context.paths.results)
        df = load_de_table(results_dir)
        if df is None:
            return self._skip("no usable de_pseudobulk_edger.csv in results")

        case = config.get("case")
        control = config.get("control")
        case_color = config.get("case_color") or figstyle.LE_RED
        control_color = config.get("control_color") or figstyle.NORMAL_BLUE
        x_label = config.get("x_label")
        if not x_label:
            if case and control:
                x_label = f"log2 fold change ({case} - {control})"
            else:
                x_label = "log2 fold change (case - control)"

        figures_dir = Path(context.paths.figures) / "differential_expression"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))

        try:
            fig = plots.volcano(
                df,
                fc_cut=float(config.get("fc_cut", 1.0)),
                fdr_cut=float(config.get("fdr_cut", 0.05)),
                case_color=case_color,
                control_color=control_color,
                x_label=x_label,
                top_n_labels=int(config.get("top_n_labels", 40)),
            )
        except Exception as exc:  # noqa: BLE001
            return self._skip(f"render failed ({str(exc)[:200]})")

        paths = figstyle.save_figure(fig, figures_dir, "volcano", formats=formats, dpi=dpi)
        artifacts = [
            StageArtifact(
                name="de_figure", path=p, kind="figure", description="Pseudobulk DE volcano."
            )
            for p in paths
        ]
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"volcano_viz rendered {len(paths)} files."],
            warnings=[],
            metrics={"method": self.name, "n_genes": int(len(df))},
            backend="python",
        )


__all__ = ["VolcanoVizMethod", "load_de_table"]
