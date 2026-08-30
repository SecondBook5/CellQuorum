# src/cellquorum/enrichment/gsea_method.py
"""Preranked GSEA enrichment method (decoupler dc.mt.gsea over a DE table)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from cellquorum.stages.comparative.enrichment.priors import PriorFetchError, get_net
from cellquorum.stages.comparative.enrichment.ranking import de_table_to_ranking
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip


def _bh(pvalues: np.ndarray, method: str) -> np.ndarray:
    """BH-adjust p-values, masking non-finite entries then reindexing."""
    out = np.full_like(pvalues, np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if finite.sum() > 0:
        out[finite] = multipletests(pvalues[finite], method=method)[1]
    return out


# Cap on how many sources get a persisted running-ES walk per collection, to
# bound file size. Significant sources are taken first; the remainder is filled
# by |score| descending (deterministic).
_RUNNING_ES_MAX_SOURCES = 20


def running_es_walk(metric: pd.Series, members: set[str]) -> pd.DataFrame | None:
    """Weighted (p=1) Subramanian running-enrichment walk down a ranked metric.

    Args:
        metric: Gene → signed ranking metric (indexed by gene).
        members: Gene-set membership (gene names).

    Returns:
        DataFrame with columns ``rank, running_es, hit, metric`` in ranked order,
        or ``None`` for a degenerate set (no hits, or every gene is a hit).
    """
    ranked = metric.sort_values(ascending=False, kind="mergesort")
    genes = ranked.index.to_numpy()
    vals = ranked.to_numpy(dtype=float)
    hit = np.fromiter((g in members for g in genes), dtype=bool, count=len(genes))
    n = len(genes)
    n_h = int(hit.sum())
    if n_h == 0 or n_h == n:
        return None
    n_r = float(np.abs(vals[hit]).sum())
    if n_r == 0.0:
        return None
    p_hit = np.where(hit, np.abs(vals) / n_r, 0.0).cumsum()
    p_miss = np.where(~hit, 1.0 / (n - n_h), 0.0).cumsum()
    running = p_hit - p_miss
    return pd.DataFrame(
        {
            "rank": np.arange(1, n + 1),
            "running_es": running,
            "hit": hit.astype(int),
            "metric": vals,
        }
    )


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
            return self._skip(f"no DE results table at {de_path}", de_path=str(de_path))

        ranking = de_table_to_ranking(pd.read_csv(de_path))
        if ranking.shape[1] == 0:
            return self._skip("DE table yielded an empty ranking")

        try:
            import decoupler as dc
        except Exception as exc:
            return self._skip("decoupler unavailable", error=str(exc)[:300])
        if dc is None:
            return self._skip("decoupler unavailable")

        results_dir.mkdir(parents=True, exist_ok=True)
        writer = StageArtifactWriter.from_context(context)
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
            artifacts.append(
                writer.table(
                    out,
                    f"enrichment_gsea_{collection}.csv",
                    name="enrichment_results",
                    description=f"GSEA ({collection}), signed -log10p ranking.",
                    index=False,
                )
            )

            # --- 5a: persist the running-ES walk for the top / significant sources.
            # Guarded so a failure to build the walk for one collection degrades to
            # a skipped-collection note, never a crash (skip-not-crash).
            try:
                metric = ranking.iloc[0]  # single-contrast ranking → one Series
                sig_sources = list(out.loc[out["significant"], "source"])
                by_mag = list(
                    out.reindex(
                        out["score"].abs().sort_values(ascending=False, kind="mergesort").index
                    )["source"]
                )
                ordered_sources: list[str] = []
                for src in sig_sources + by_mag:
                    if src not in ordered_sources:
                        ordered_sources.append(src)
                    if len(ordered_sources) >= _RUNNING_ES_MAX_SOURCES:
                        break
                source_targets = net.groupby("source")["target"].agg(lambda t: set(map(str, t)))
                walk_rows = []
                for src in ordered_sources:
                    members = source_targets.get(src, set()) & set(ranking.columns)
                    walk = running_es_walk(metric, members)
                    if walk is None:
                        continue
                    walk = walk.copy()
                    walk.insert(0, "source", src)
                    walk_rows.append(walk)
                if walk_rows:
                    running_df = pd.concat(walk_rows, ignore_index=True)
                    running_df = running_df[["source", "rank", "running_es", "hit", "metric"]]
                    artifacts.append(
                        writer.table(
                            running_df,
                            f"enrichment_gsea_runningES_{collection}.csv",
                            name="enrichment_results",
                            description=f"GSEA running-ES walk ({collection}).",
                            index=False,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - degrade to a note, never crash
                skipped.append(
                    {"collection": collection, "reason": f"runningES skipped: {str(exc)[:200]}"}
                )

            done.append(collection)

        if not done:
            return self._skip("no collection produced results", skipped=skipped)

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
