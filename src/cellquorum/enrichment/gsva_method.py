# src/cellquorum/enrichment/gsva_method.py
"""GSVA pathway-activity method: pseudobulk GSVA + per-source group contrast."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.differential_expression.pseudobulk import aggregate_pseudobulk
from cellquorum.enrichment.priors import PriorFetchError, get_net
from cellquorum.methods.base import AnalysisMethod, MethodSkip


def _bh(pvalues: np.ndarray, method: str) -> np.ndarray:
    out = np.full_like(pvalues, np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if finite.sum() > 0:
        out[finite] = multipletests(pvalues[finite], method=method)[1]
    return out


class GsvaMethod(AnalysisMethod):
    """Pseudobulk GSVA activity + paired/unpaired group contrast per source."""

    name = "gsva"
    stage_category = "enrichment"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        counts_layer = config.get("counts_layer", "counts")
        return DataContract(
            required_obs=[condition_col, donor_col],
            required_layers=[counts_layer],
        )

    def requires_obs(self, config: dict) -> list[str]:
        return [config.get("condition_col", "condition"), config.get("donor_col", "patient_id")]

    def requires_layers(self) -> list[str]:
        # Base hook is config-less (the scvi pattern); guard on the default
        # counts-layer name. A non-default counts_layer must exist regardless,
        # and the contract re-checks it before _run.
        return ["counts"]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        counts_layer = config.get("counts_layer", "counts")
        case = config.get("case")
        control = config.get("control")
        paired = bool(config.get("paired", False))
        collections = config.get("gene_set_collections", ["hallmark", "reactome"])
        gmt_path = config.get("gmt_path")
        organism = config.get("organism", "human")
        license = config.get("license", "academic")
        min_size = int(config.get("min_size", 10))
        fdr_method = config.get("fdr_method", "fdr_bh")

        if not case or not control:
            return MethodSkip(
                reason="gsva skipped: case/control labels not set in config",
                details={"method": self.name},
            )

        pb = aggregate_pseudobulk(
            adata, layer=counts_layer, donor_col=donor_col, condition_col=condition_col
        )
        counts = pb.counts
        meta = pb.sample_meta
        keep = meta[condition_col].isin([case, control])
        counts, meta = counts[keep.values], meta[keep]
        n_case = int((meta[condition_col] == case).sum())
        n_control = int((meta[condition_col] == control).sum())
        if n_case < 2 or n_control < 2:
            return MethodSkip(
                reason="gsva skipped: need ≥2 pseudobulk samples per arm",
                details={"method": self.name, "n_case": n_case, "n_control": n_control},
            )

        # CPM + log1p normalize pseudobulk counts.
        lib = counts.sum(axis=1).replace(0, np.nan)
        data = np.log1p(counts.div(lib, axis=0) * 1e6).fillna(0.0)

        try:
            import decoupler as dc
        except Exception as exc:
            return MethodSkip(
                reason="gsva skipped: decoupler unavailable",
                details={"method": self.name, "error": str(exc)[:300]},
            )
        if dc is None:
            return MethodSkip(
                reason="gsva skipped: decoupler unavailable", details={"method": self.name}
            )

        results_dir = Path(context.paths.results)
        results_dir.mkdir(parents=True, exist_ok=True)
        cond = meta[condition_col].values
        artifacts, done, skipped = [], [], []
        for collection in collections:
            try:
                net = get_net(collection, organism=organism, gmt_path=gmt_path, license=license)
            except PriorFetchError as exc:
                skipped.append({"collection": collection, "reason": str(exc)[:300]})
                continue
            try:
                es, _ = dc.mt.gsva(data, net, tmin=min_size)
            except Exception as exc:
                skipped.append({"collection": collection, "reason": str(exc)[:300]})
                continue

            scores_csv = results_dir / f"enrichment_gsva_scores_{collection}.csv"
            es.to_csv(scores_csv)
            artifacts.append(
                StageArtifact(
                    name="enrichment_results",
                    path=scores_csv,
                    kind="csv",
                    description=f"GSVA per-sample scores ({collection}).",
                )
            )

            rows = []
            for source in es.columns:
                case_vals = es.loc[cond == case, source].values
                control_vals = es.loc[cond == control, source].values
                try:
                    if paired and len(case_vals) == len(control_vals) and len(case_vals) >= 2:
                        res = stats.ttest_rel(case_vals, control_vals)
                    else:
                        res = stats.ttest_ind(case_vals, control_vals, equal_var=False)
                    statistic, pvalue = res.statistic, res.pvalue
                except Exception:
                    statistic, pvalue = np.nan, np.nan
                rows.append(
                    {
                        "source": source,
                        "case_mean": float(np.mean(case_vals)),
                        "control_mean": float(np.mean(control_vals)),
                        "statistic": statistic,
                        "pvalue": pvalue,
                        "collection": collection,
                    }
                )
            contrast = pd.DataFrame(rows)
            contrast["padj"] = _bh(contrast["pvalue"].values.astype(float), fdr_method)
            contrast = contrast[
                ["source", "case_mean", "control_mean", "statistic", "pvalue", "padj", "collection"]
            ]
            contrast_csv = results_dir / f"enrichment_gsva_contrast_{collection}.csv"
            contrast.to_csv(contrast_csv, index=False)
            artifacts.append(
                StageArtifact(
                    name="enrichment_results",
                    path=contrast_csv,
                    kind="csv",
                    description=f"GSVA contrast {case} vs {control} ({collection}).",
                )
            )
            done.append(collection)

        if not done:
            return MethodSkip(
                reason="gsva skipped: no collection produced results",
                details={"method": self.name, "skipped": skipped},
            )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"GSVA over {done}: {case} vs {control}."],
            metrics={
                "method": self.name,
                "case": case,
                "control": control,
                "paired": paired,
                "n_collections": len(done),
                "collections": done,
                "n_case": n_case,
                "n_control": n_control,
                "skipped": skipped,
            },
            backend="python",
        )


__all__ = ["GsvaMethod"]
