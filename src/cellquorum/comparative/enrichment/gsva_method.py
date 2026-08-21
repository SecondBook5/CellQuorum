# src/cellquorum/enrichment/gsva_method.py
"""GSVA pathway-activity method: pseudobulk GSVA + per-source group contrast."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from cellquorum.comparative.differential_expression.pseudobulk import aggregate_pseudobulk
from cellquorum.comparative.enrichment.priors import PriorFetchError, get_net
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
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
        max_size = int(config.get("max_size", 500))
        fdr_method = config.get("fdr_method", "fdr_bh")
        fdr = float(config.get("fdr", 0.05))

        if not case or not control:
            return self._skip("case/control labels not set in config")

        pb = aggregate_pseudobulk(
            adata, layer=counts_layer, donor_col=donor_col, condition_col=condition_col
        )
        counts = pb.counts
        meta = pb.sample_meta
        keep = meta[condition_col].isin([case, control])
        counts, meta = counts[keep.values], meta[keep]

        # Drop zero-library pseudobulk samples up front. Left in, they would be
        # CPM-normalized into all-zero rows that decoupler.gsva silently drops
        # (empty=True), shrinking the returned frame and desynchronizing any
        # positional condition mask. Removing them here keeps counts/meta aligned
        # with what decoupler will actually score.
        lib = counts.sum(axis=1)
        nonzero = lib > 0
        counts, meta = counts[nonzero.values], meta[nonzero.values]

        n_case = int((meta[condition_col] == case).sum())
        n_control = int((meta[condition_col] == control).sum())
        if n_case < 2 or n_control < 2:
            return self._skip(
                "need ≥2 pseudobulk samples per arm", n_case=n_case, n_control=n_control
            )

        # Auto-promote to a paired contrast when the design is fully matched, so
        # the sample-level GSVA test blocks on donor (ttest_rel) instead of the
        # weaker independent test — mirrors the pseudobulk edgeR auto-promotion.
        _case_donors = set(meta.loc[meta[condition_col] == case, donor_col])
        _control_donors = set(meta.loc[meta[condition_col] == control, donor_col])
        if (
            not paired
            and _case_donors
            and _case_donors == _control_donors
            and len(_case_donors) >= 2
        ):
            paired = True

        # CPM + log1p normalize the surviving (nonzero-library) pseudobulk counts.
        data = np.log1p(counts.div(counts.sum(axis=1), axis=0) * 1e6)

        try:
            import decoupler as dc
        except Exception as exc:
            return self._skip("decoupler unavailable", error=str(exc)[:300])
        if dc is None:
            return self._skip("decoupler unavailable")

        writer = StageArtifactWriter.from_context(context)
        artifacts, done, skipped = [], [], []
        for collection in collections:
            try:
                net = get_net(collection, organism=organism, gmt_path=gmt_path, license=license)
            except PriorFetchError as exc:
                skipped.append({"collection": collection, "reason": str(exc)[:300]})
                continue

            # Enforce the max_size gene-set filter (min_size is decoupler's tmin):
            # keep only sources whose target count present in the data does not
            # exceed max_size, so oversized sets are excluded here as designed.
            present = net[net["target"].isin(data.columns)]
            sizes = present.groupby("source")["target"].nunique()
            keep_sources = set(sizes[sizes <= max_size].index)
            net = net[net["source"].isin(keep_sources)]

            # The gsva call through the aligned t-test is guarded together: gsva
            # may drop samples (empty=True), so deriving the condition vector from
            # the RETURNED es.index — never the pre-drop meta order — and running
            # the contrast happen inside one try/except that degrades to a skip.
            try:
                es, _ = dc.mt.gsva(data, net, tmin=min_size)
                # Align conditions to decoupler's returned sample order.
                cond = meta[condition_col].reindex(es.index)
                donor_of = meta[donor_col].reindex(es.index)
                case_mask = (cond == case).values
                control_mask = (cond == control).values
                if int(case_mask.sum()) < 2 or int(control_mask.sum()) < 2:
                    skipped.append(
                        {
                            "collection": collection,
                            "reason": "an arm lost samples to decoupler filtering (<2 remain)",
                        }
                    )
                    continue

                # For a paired test, arms must be aligned by donor — never by
                # positional mask order, which is not donor-consistent. Build the
                # ordered donor list present in BOTH arms (a collection can lose a
                # donor to decoupler filtering, so recompute per collection).
                case_donor = donor_of[case_mask]
                control_donor = donor_of[control_mask]
                paired_donors = sorted(set(case_donor) & set(control_donor))
                use_paired = paired and len(paired_donors) >= 2

                rows = []
                for source in es.columns:
                    if use_paired:
                        # Index each arm by donor, then align on the shared donors
                        # so ttest_rel compares matched case/control per patient.
                        case_by_donor = pd.Series(
                            es.loc[case_mask, source].values, index=case_donor.values
                        ).loc[paired_donors]
                        control_by_donor = pd.Series(
                            es.loc[control_mask, source].values, index=control_donor.values
                        ).loc[paired_donors]
                        case_vals = case_by_donor.values
                        control_vals = control_by_donor.values
                        res = stats.ttest_rel(case_vals, control_vals)
                    else:
                        case_vals = es.loc[case_mask, source].values
                        control_vals = es.loc[control_mask, source].values
                        res = stats.ttest_ind(case_vals, control_vals, equal_var=False)
                    rows.append(
                        {
                            "source": source,
                            "case_mean": float(np.mean(case_vals)),
                            "control_mean": float(np.mean(control_vals)),
                            "statistic": res.statistic,
                            "pvalue": res.pvalue,
                            "collection": collection,
                        }
                    )
            except Exception as exc:
                skipped.append({"collection": collection, "reason": str(exc)[:300]})
                continue

            artifacts.append(
                writer.table(
                    es,
                    f"enrichment_gsva_scores_{collection}.csv",
                    name="enrichment_results",
                    description=f"GSVA per-sample scores ({collection}).",
                    index=True,
                )
            )

            contrast = pd.DataFrame(rows)
            # Single BH/FDR over the t-test's own raw p-values.
            contrast["padj"] = _bh(contrast["pvalue"].values.astype(float), fdr_method)
            contrast["significant"] = contrast["padj"] < fdr
            contrast = contrast[
                [
                    "source",
                    "case_mean",
                    "control_mean",
                    "statistic",
                    "pvalue",
                    "padj",
                    "significant",
                    "collection",
                ]
            ]
            artifacts.append(
                writer.table(
                    contrast,
                    f"enrichment_gsva_contrast_{collection}.csv",
                    name="enrichment_results",
                    description=f"GSVA contrast {case} vs {control} ({collection}).",
                    index=False,
                )
            )
            done.append(collection)

        if not done:
            return self._skip("no collection produced results", skipped=skipped)

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
