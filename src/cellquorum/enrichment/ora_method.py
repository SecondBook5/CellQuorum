"""Over-representation analysis method (decoupler dc.mt.ora over DE up/down sets)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.enrichment.priors import PriorFetchError, get_net
from cellquorum.methods.base import AnalysisMethod, MethodSkip


def _bh(pvalues: np.ndarray, method: str) -> np.ndarray:
    out = np.full_like(pvalues, np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if finite.sum() > 0:
        out[finite] = multipletests(pvalues[finite], method=method)[1]
    return out


class OraMethod(AnalysisMethod):
    """ORA on DE up/down gene sets; background = all tested genes."""

    name = "ora"
    stage_category = "enrichment"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        de_name = config.get("de_results_filename", "de_pseudobulk_edger.csv")
        collections = config.get("gene_set_collections", ["hallmark", "reactome"])
        gmt_path = config.get("gmt_path")
        organism = config.get("organism", "human")
        license = config.get("license", "academic")
        min_size = int(config.get("min_size", 10))
        lfc_threshold = float(config.get("lfc_threshold", 0.0))
        fg_padj = float(config.get("fg_padj", 0.05))
        min_fg = int(config.get("min_foreground_genes", 5))
        fdr_method = config.get("fdr_method", "fdr_bh")

        results_dir = Path(context.paths.results)
        de_path = results_dir / de_name
        if not de_path.exists():
            return MethodSkip(
                reason=f"ora skipped: no DE results table at {de_path}",
                details={"method": self.name},
            )

        de = pd.read_csv(de_path)
        background = list(pd.unique(de["gene"].dropna()))
        up = set(de.loc[(de["FDR"] < fg_padj) & (de["logFC"] > lfc_threshold), "gene"])
        down = set(de.loc[(de["FDR"] < fg_padj) & (de["logFC"] < -lfc_threshold), "gene"])
        directions = {"up": up, "down": down}

        try:
            import decoupler as dc
        except Exception as exc:
            return MethodSkip(
                reason="ora skipped: decoupler unavailable",
                details={"method": self.name, "error": str(exc)[:300]},
            )
        if dc is None:
            return MethodSkip(
                reason="ora skipped: decoupler unavailable", details={"method": self.name}
            )

        results_dir.mkdir(parents=True, exist_ok=True)
        artifacts, done, skipped = [], [], []
        for collection in collections:
            try:
                net = get_net(collection, organism=organism, gmt_path=gmt_path, license=license)
            except PriorFetchError as exc:
                skipped.append({"collection": collection, "reason": str(exc)[:300]})
                continue

            rows = []
            for direction, fg in directions.items():
                fg = fg & set(background)
                if len(fg) < min_fg:
                    continue
                members = pd.DataFrame(
                    [[1.0 if g in fg else 0.0 for g in background]],
                    index=["contrast"],
                    columns=background,
                )
                try:
                    es, pv = dc.mt.ora(members, net, tmin=min_size)
                except Exception as exc:
                    skipped.append(
                        {"collection": collection, "direction": direction, "reason": str(exc)[:200]}
                    )
                    continue
                score = es.loc["contrast"]
                pvalue = pv.loc["contrast"]
                rows.append(
                    pd.DataFrame(
                        {
                            "source": score.index,
                            "direction": direction,
                            "score": score.values,
                            "pvalue": pvalue.values,
                            "collection": collection,
                        }
                    )
                )

            if not rows:
                continue
            out = pd.concat(rows, ignore_index=True)
            out["padj"] = _bh(out["pvalue"].values.astype(float), fdr_method)
            out = out[["source", "direction", "score", "pvalue", "padj", "collection"]]
            out_csv = results_dir / f"enrichment_ora_{collection}.csv"
            out.to_csv(out_csv, index=False)
            artifacts.append(
                StageArtifact(
                    name="enrichment_results",
                    path=out_csv,
                    kind="csv",
                    description=f"ORA ({collection}), background=tested genes.",
                )
            )
            done.append(collection)

        if not done:
            return MethodSkip(
                reason="ora skipped: no collection/direction produced results",
                details={"method": self.name, "skipped": skipped},
            )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"ORA over {done}."],
            metrics={
                "method": self.name,
                "n_collections": len(done),
                "collections": done,
                "n_up": len(up),
                "n_down": len(down),
                "n_background": len(background),
                "skipped": skipped,
            },
            backend="python",
        )


__all__ = ["OraMethod"]
