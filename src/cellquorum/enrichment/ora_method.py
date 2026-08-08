"""Over-representation analysis method (direct hypergeometric over DE up/down sets).

ORA is a set-membership test: the DE-derived foreground gene set (up or down) is
tested for over-representation in each gene-set of a collection, against the
background of all tested genes (the DE table's gene universe). We compute the
contingency-table Fisher/hypergeometric test directly (scipy) rather than routing
a 0/1 expression row through ``dc.mt.ora`` — that decoupler path ranks the row and
selects a top-n% of features against a fixed ``n_bg=20000``, which silently
violates the design (wrong foreground selection, wrong background) and, because
``empty=True`` drops the all-zero columns, degenerates the table entirely. The
direct test yields a genuine raw p-value, so BH/FDR is applied exactly once here.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
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
        max_size = int(config.get("max_size", 500))
        lfc_threshold = float(config.get("lfc_threshold", 0.0))
        fg_padj = float(config.get("fg_padj", 0.05))
        min_fg = int(config.get("min_foreground_genes", 5))
        fdr_method = config.get("fdr_method", "fdr_bh")
        fdr = float(config.get("fdr", 0.05))

        results_dir = Path(context.paths.results)
        de_path = results_dir / de_name
        if not de_path.exists():
            return MethodSkip(
                reason=f"ora skipped: no DE results table at {de_path}",
                details={"method": self.name},
            )

        de = pd.read_csv(de_path)
        # Background universe = every gene tested by DE (the design's tested-gene
        # background). This is the hypergeometric population, NOT a fixed 20000.
        background = set(pd.unique(de["gene"].dropna()))
        n_background = len(background)
        # Foreground = DE-significant genes split by direction; tested separately
        # so the directional (up/down) stacking of the output schema is preserved.
        up = set(de.loc[(de["FDR"] < fg_padj) & (de["logFC"] > lfc_threshold), "gene"])
        down = set(de.loc[(de["FDR"] < fg_padj) & (de["logFC"] < -lfc_threshold), "gene"])
        directions = {"up": up, "down": down}

        results_dir.mkdir(parents=True, exist_ok=True)
        artifacts, done, skipped = [], [], []
        for collection in collections:
            try:
                net = get_net(collection, organism=organism, gmt_path=gmt_path, license=license)
            except PriorFetchError as exc:
                skipped.append({"collection": collection, "reason": str(exc)[:300]})
                continue

            # Restrict each gene-set to the tested universe, then apply the same
            # min_size/max_size filters decoupler's tmin would (post-restriction).
            genesets: dict[str, set[str]] = {}
            for source, targets in net.groupby("source")["target"]:
                members = set(map(str, targets)) & background
                if min_size <= len(members) <= max_size:
                    genesets[str(source)] = members
            if not genesets:
                skipped.append(
                    {"collection": collection, "reason": "no gene-set passed size filters"}
                )
                continue

            rows = []
            for direction, fg in directions.items():
                # Foreground must live inside the tested universe (a gene absent
                # from the DE table cannot be drawn from the population).
                fg = fg & background
                n_fg = len(fg)
                if n_fg < min_fg:
                    continue
                sources = list(genesets)
                # Contingency counts per gene-set: overlap k, set size K, drawn N,
                # population M. Fisher/hypergeometric survival gives the raw right-
                # tail p (P[overlap >= k]); log2 odds ratio is the reported score.
                scores, pvalues, counts, ratios = [], [], [], []
                for source in sources:
                    members = genesets[source]
                    k = len(fg & members)  # a: foreground ∩ set
                    K = len(members)  # a + c: set size in universe
                    p = float(hypergeom.sf(k - 1, n_background, K, n_fg))
                    # 2x2 odds ratio with a Haldane-Anscombe +0.5 continuity term.
                    a, b = k, n_fg - k
                    c, d = K - k, n_background - K - (n_fg - k)
                    odds = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
                    scores.append(float(np.log2(odds)))
                    pvalues.append(p)
                    counts.append(int(k))
                    ratios.append(float(k / n_fg) if n_fg else 0.0)
                rows.append(
                    pd.DataFrame(
                        {
                            "source": sources,
                            "direction": direction,
                            "score": scores,
                            "pvalue": pvalues,
                            "collection": collection,
                            "count": counts,
                            "gene_ratio": ratios,
                        }
                    )
                )

            if not rows:
                continue
            out = pd.concat(rows, ignore_index=True)
            # Single BH/FDR over the genuine raw hypergeometric p-values.
            out["padj"] = _bh(out["pvalue"].values.astype(float), fdr_method)
            out["significant"] = out["padj"] < fdr
            out = out[
                [
                    "source",
                    "direction",
                    "score",
                    "pvalue",
                    "padj",
                    "significant",
                    "collection",
                    "count",
                    "gene_ratio",
                ]
            ]
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
