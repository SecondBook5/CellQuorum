# src/cellquorum/enrichment/gsea_method.py
"""Preranked GSEA enrichment method (decoupler dc.mt.gsea over a DE table)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.enrichment.priors import PriorFetchError, get_net
from cellquorum.enrichment.ranking import de_table_to_ranking
from cellquorum.methods.base import AnalysisMethod, MethodSkip


def _bh(pvalues: np.ndarray, method: str) -> np.ndarray:
    """BH-adjust p-values, masking non-finite entries then reindexing."""
    out = np.full_like(pvalues, np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if finite.sum() > 0:
        out[finite] = multipletests(pvalues[finite], method=method)[1]
    return out


class GseaMethod(AnalysisMethod):
    """Preranked GSEA over the upstream DE table (the enrichment anchor)."""

    name = "gsea"
    stage_category = "enrichment"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """GSEA reads the DE CSV, not the matrix — no obs/layer requirement."""
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        de_name = config.get("de_results_filename", "de_pseudobulk_edger.csv")
        collections = config.get("gene_set_collections", ["hallmark", "reactome"])
        gmt_path = config.get("gmt_path")
        organism = config.get("organism", "human")
        license = config.get("license", "academic")
        min_size = int(config.get("min_size", 10))
        max_size = int(config.get("max_size", 500))
        permutations = int(config.get("gsea_permutations", 1000))
        seed = int(config.get("seed", 42))
        fdr_method = config.get("fdr_method", "fdr_bh")
        fdr = float(config.get("fdr", 0.05))

        results_dir = Path(context.paths.results)
        de_path = results_dir / de_name
        if not de_path.exists():
            return MethodSkip(
                reason=f"gsea skipped: no DE results table at {de_path}",
                details={"method": self.name, "de_path": str(de_path)},
            )

        ranking = de_table_to_ranking(pd.read_csv(de_path))
        if ranking.shape[1] == 0:
            return MethodSkip(
                reason="gsea skipped: DE table yielded an empty ranking",
                details={"method": self.name},
            )

        try:
            import decoupler as dc
        except Exception as exc:
            return MethodSkip(
                reason="gsea skipped: decoupler unavailable",
                details={"method": self.name, "error": str(exc)[:300]},
            )
        if dc is None:
            return MethodSkip(
                reason="gsea skipped: decoupler unavailable",
                details={"method": self.name},
            )

        results_dir.mkdir(parents=True, exist_ok=True)
        artifacts, done, skipped = [], [], []
        for collection in collections:
            try:
                net = get_net(collection, organism=organism, gmt_path=gmt_path, license=license)
            except PriorFetchError as exc:
                skipped.append({"collection": collection, "reason": str(exc)[:300]})
                continue

            # Enforce the max_size gene-set filter against the ranked universe
            # (min_size maps to decoupler's tmin below); drop oversized sources.
            present = net[net["target"].isin(ranking.columns)]
            sizes = present.groupby("source")["target"].nunique()
            keep_sources = set(sizes[sizes <= max_size].index)
            net = net[net["source"].isin(keep_sources)]

            try:
                # We call gsea through decoupler's low-level building blocks rather
                # than dc.mt.gsea(...) because the high-level Method has test=True
                # and applies its OWN across-source BH before returning — so its
                # "p-value" is already an across-source q-value. BH-ing that again
                # here would double-correct (decoupler's BH is not idempotent).
                # Reconstructing gsea.func gives the GENUINE raw permutation p, so
                # we label it `pvalue` and apply BH exactly once for `padj`.
                mat, obs, var = dc.pp.extract(ranking, empty=True, shuffle=True, verbose=False)
                pruned = dc.pp.prune(features=var, net=net, tmin=min_size, verbose=False)
                sources, cnct, starts, offsets = dc.pp.idxmat(
                    features=var, net=pruned, verbose=False
                )
                es_arr, pv_arr = dc.mt.gsea.func(
                    mat, cnct, starts, offsets, times=permutations, seed=seed, verbose=False
                )
                # Single-contrast ranking → single row of scores and raw p-values.
                score = pd.Series(es_arr[0], index=sources)
                pvalue = pd.Series(pv_arr[0], index=sources)
            except Exception as exc:
                skipped.append({"collection": collection, "reason": str(exc)[:300]})
                continue

            padj = _bh(pvalue.values.astype(float), fdr_method)
            out = pd.DataFrame(
                {
                    "source": score.index,
                    "score": score.values,
                    "pvalue": pvalue.values,
                    "padj": padj,
                    "significant": padj < fdr,
                    "collection": collection,
                }
            )
            out_csv = results_dir / f"enrichment_gsea_{collection}.csv"
            out.to_csv(out_csv, index=False)
            artifacts.append(
                StageArtifact(
                    name="enrichment_results",
                    path=out_csv,
                    kind="csv",
                    description=f"GSEA ({collection}), signed -log10p ranking.",
                )
            )
            done.append(collection)

        if not done:
            return MethodSkip(
                reason="gsea skipped: no collection produced results",
                details={"method": self.name, "skipped": skipped},
            )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"GSEA over {done}."],
            metrics={
                "method": self.name,
                "n_collections": len(done),
                "collections": done,
                "skipped_collections": skipped,
                "seed": seed,
            },
            backend="python",
        )


__all__ = ["GseaMethod"]
