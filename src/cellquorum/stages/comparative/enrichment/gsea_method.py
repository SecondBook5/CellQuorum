# src/cellquorum/enrichment/gsea_method.py
"""Preranked GSEA enrichment method (decoupler dc.mt.gsea over a DE table)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.comparative.enrichment.priors import (
    PriorFetchError,
    collection_names,
    get_net,
)
from cellquorum.stages.comparative.enrichment.ranking import de_table_to_ranking
from cellquorum.stats.multiple_testing import bh_fdr

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext

# Cap on how many sources get a persisted running-ES walk per collection, to
# bound file size. Rank by significance, adjusted p-value and absolute NES.
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
            "gene": genes,
        }
    )


def leading_edge(metric: pd.Series, members: set[str]) -> tuple[float, list[str]] | None:
    """The enrichment score and the genes that account for it.

    The leading edge is the subset of the gene set that appears before (for a positive score)
    or after (for a negative one) the peak of the running walk — the genes that actually
    carry the enrichment, as opposed to the set's other members. Without it a GSEA table
    says only that a pathway moved; a reader cannot ask *which genes*, and no downstream
    membership analysis is reproducible from the output.

    Uses the same p = 1 Subramanian weighting as :func:`running_es_walk`, but in closed form
    over the hit positions rather than walking every gene, because this runs for every source
    in a collection rather than the twenty that get a persisted walk. The two agreeing on the
    score is pinned by a test, so the table and the plotted walk cannot drift apart.

    Args:
        metric: Gene → signed ranking metric (indexed by gene).
        members: Gene-set membership, already intersected with the ranked universe.

    Returns:
        ``(score, genes)`` with ``genes`` in ranked order, or ``None`` for a degenerate set
        (no hits, every gene a hit, or hits carrying zero total weight).
    """
    ranked = metric.sort_values(ascending=False, kind="mergesort")
    genes = ranked.index.to_numpy()
    values = np.abs(ranked.to_numpy(dtype=float))
    n = len(genes)
    hit_positions = np.flatnonzero(np.isin(genes, list(members)))
    n_hits = hit_positions.size
    if n_hits == 0 or n_hits == n:
        return None

    weights = values[hit_positions]
    total_weight = float(weights.sum())
    if total_weight == 0.0:
        return None

    hit_index = np.arange(n_hits)
    misses_before = hit_positions - hit_index
    cumulative = np.cumsum(weights) / total_weight
    miss_penalty = misses_before / (n - n_hits)

    # The walk only falls between hits, so its maximum is at a hit and its minimum is
    # immediately before one. Both are compared against 0, which the walk reaches at the
    # end of the list — a set with no deviation in either direction scores 0, not the
    # least-bad of two deviations.
    running_at_hits = cumulative - miss_penalty
    running_before_hits = np.concatenate(([0.0], cumulative[:-1])) - miss_penalty
    peak = int(np.argmax(running_at_hits))
    trough = int(np.argmin(running_before_hits))
    high = max(float(running_at_hits[peak]), 0.0)
    low = min(float(running_before_hits[trough]), 0.0)

    if high >= -low:
        # Positive score: the genes from the top of the list down to the peak.
        return high, [str(gene) for gene in genes[hit_positions[: peak + 1]]]
    # Negative score: the genes from the trough down to the bottom of the list.
    return low, [str(gene) for gene in genes[hit_positions[trough:]]]


def _add_leading_edge(
    out: pd.DataFrame,
    metric: pd.Series,
    net: pd.DataFrame,
    skipped: list[dict],
    collection: str,
) -> pd.DataFrame:
    """Attach set size, the unnormalized ES, and the leading edge to a GSEA result table.

    ``es`` is here because ``score`` is decoupler's *normalized* enrichment score — divided
    by the mean of the same-sign permutation scores, so it is unbounded and its magnitude
    depends on how the null came out. The raw ES is bounded in [-1, 1], is the quantity the
    persisted running walk peaks at, and is the one that can be checked by hand against the
    leading-edge genes in the same row. Reporting only the normalized score leaves nothing
    in the table that the figure beside it can be reconciled with.

    The columns are always added, even when the computation fails: a downstream reader that
    checks ``if "leading_edge" in table`` would otherwise silently take its no-leading-edge
    branch on a table where the walk merely errored, which is the difference between "this
    run cannot answer that" and "this pathway has no leading edge".
    """
    out = out.copy()
    out.insert(cast("int", out.columns.get_loc("score")) + 1, "es", cast("Any", pd.NA))
    out["set_size"] = pd.NA
    out["leading_edge_size"] = pd.NA
    out["leading_edge"] = pd.NA
    try:
        universe = set(map(str, metric.index))
        by_source = {
            source: set(map(str, targets)) for source, targets in net.groupby("source")["target"]
        }
        sizes: list[int] = []
        raw_scores: list[Any] = []
        edge_sizes: list[Any] = []
        edges: list[Any] = []
        for source in out["source"]:
            members = by_source.get(source, set()) & universe
            found = leading_edge(metric, members) if members else None
            sizes.append(len(members))
            if found is None:
                raw_scores.append(pd.NA)
                edge_sizes.append(pd.NA)
                edges.append(pd.NA)
                continue
            score, genes = found
            raw_scores.append(score)
            edge_sizes.append(len(genes))
            # Semicolon-joined: a gene symbol never contains one, and a comma would need
            # quoting in the CSV this is written to.
            edges.append(";".join(genes))
        out["set_size"] = sizes
        out["es"] = raw_scores
        out["leading_edge_size"] = edge_sizes
        out["leading_edge"] = edges
    except Exception as exc:  # noqa: BLE001 - degrade to a note, never lose the collection
        skipped.append(
            {"collection": collection, "reason": f"leading edge skipped: {str(exc)[:200]}"}
        )
    return out


def filter_net_by_size(net: pd.DataFrame, universe: set[str], max_size: int) -> pd.DataFrame:
    """Drop gene sets larger than ``max_size`` once restricted to the ranked ``universe``.

    ``min_size`` maps to decoupler's ``tmin`` inside :func:`gsea_table`; the maximum has no
    decoupler equivalent, so it is enforced here, against the genes actually ranked.
    """
    present = net[net["target"].isin(universe)]
    sizes = present.groupby("source")["target"].nunique()
    keep_sources = set(sizes[sizes <= max_size].index)
    return net[net["source"].isin(keep_sources)]


def gsea_table(
    metric: pd.Series,
    net: pd.DataFrame,
    *,
    min_size: int,
    permutations: int,
    seed: int,
    fdr_method: str,
    fdr: float,
    skipped: list[dict],
    collection: str,
) -> pd.DataFrame:
    """Preranked GSEA of one ranked ``metric`` (gene -> signed score) over the sets in ``net``.

    Called through decoupler's low-level building blocks rather than ``dc.mt.gsea(...)``
    because the high-level Method has test=True and applies its OWN across-source BH before
    returning, so its "p-value" is already an across-source q-value, and BH-ing that again
    would double-correct (decoupler's BH is not idempotent). Reconstructing ``gsea.func``
    gives the GENUINE raw permutation p, which is floored at the test's resolution and
    BH-adjusted exactly once.

    Returns:
        One row per gene set: ``score`` (decoupler's normalized ES, the NES), ``es`` (the raw
        ES), ``pvalue``, ``padj``, ``significant``, the resolution-limit flags, the set size
        and the leading edge. Raises if decoupler cannot score the sets.
    """
    import decoupler as dc

    ranking = pd.DataFrame([metric], index=["contrast"])
    mat, obs, var = dc.pp.extract(ranking, empty=True, shuffle=True, verbose=False)
    pruned = dc.pp.prune(features=var, net=net, tmin=min_size, verbose=False)
    sources, cnct, starts, offsets = dc.pp.idxmat(features=var, net=pruned, verbose=False)
    es_arr, pv_arr = dc.mt.gsea.func(
        mat, cnct, starts, offsets, times=permutations, seed=seed, verbose=False
    )
    score = pd.Series(es_arr[0], index=sources)
    pvalue = pd.Series(pv_arr[0], index=sources)

    # Older decoupler versions can return zero. Enforce the global permutation floor
    # while preserving larger p-values from sign-specific corrected null distributions.
    resolution = 1.0 / (permutations + 1)
    raw = pvalue.to_numpy(dtype=float)
    at_limit = np.isfinite(raw) & (raw <= resolution)
    floored = np.where(at_limit, resolution, raw)
    padj = bh_fdr(floored, fdr_method)
    out = pd.DataFrame(
        {
            "source": score.index,
            "score": score.values,
            "pvalue": floored,
            "padj": padj,
            "significant": padj < fdr,
            "p_at_resolution_limit": at_limit,
            "p_resolution_limit": resolution,
            "permutations": permutations,
            "collection": collection,
        }
    )
    return _add_leading_edge(out, metric, net, skipped, collection)


class GseaMethod(AnalysisMethod):
    """Preranked GSEA over the upstream DE table (the enrichment anchor)."""

    name = "gsea"
    stage_category = "enrichment"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """GSEA reads the DE CSV, not the matrix — no obs/layer requirement."""
        return DataContract()

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        de_name = config.get("de_results_filename", "de_pseudobulk_edger.csv")
        collections = collection_names(
            config.get("gene_set_collections", ["hallmark", "reactome"]), config.get("gmt_path")
        )
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

            net = filter_net_by_size(net, set(ranking.columns), max_size)
            try:
                out = gsea_table(
                    ranking.iloc[0],
                    net,
                    min_size=min_size,
                    permutations=permutations,
                    seed=seed,
                    fdr_method=fdr_method,
                    fdr=fdr,
                    skipped=skipped,
                    collection=collection,
                )
            except Exception as exc:
                skipped.append({"collection": collection, "reason": str(exc)[:300]})
                continue

            artifacts.append(
                writer.table(
                    out,
                    f"enrichment_gsea_{collection}.csv",
                    name="enrichment_results",
                    description=(
                        f"GSEA ({collection}), signed -log10p ranking. `score` is the "
                        "normalized ES, `es` the raw ES the walk peaks at."
                    ),
                    index=False,
                )
            )

            # --- 5a: persist the running-ES walk for the top / significant sources.
            # Guarded so a failure to build the walk for one collection degrades to
            # a skipped-collection note, never a crash (skip-not-crash).
            try:
                metric = ranking.iloc[0]  # single-contrast ranking → one Series
                # The table is in gene-set name order. Pick examples by evidence so the
                # six walks that reach the report do not just start at the letter A.
                ranked_sets = out.assign(magnitude=out["score"].abs()).sort_values(
                    ["significant", "padj", "magnitude", "source"],
                    ascending=[False, True, False, True],
                )
                ordered_sources = ranked_sets["source"].head(_RUNNING_ES_MAX_SOURCES)
                source_targets = {
                    source: set(map(str, targets))
                    for source, targets in net.groupby("source")["target"]
                }
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
                    running_df = running_df[
                        ["source", "rank", "running_es", "hit", "metric", "gene"]
                    ]
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

        # A configured collection that produced nothing is reported in the notes, not only
        # in the metrics. A run configured for [hallmark, reactome] that silently writes one
        # table looks complete on the filesystem, and the only trace of the missing half was
        # a dict a reader has to know to go looking for.
        missing = [name for name in collections if name not in done]
        notes = [f"GSEA over {done}."]
        if missing:
            reasons = {str(entry["collection"]): str(entry["reason"]) for entry in skipped}
            named = (f"{name} ({reasons.get(name, 'no reason recorded')})" for name in missing)
            notes.append("GSEA produced no table for " + "; ".join(named))

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
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
