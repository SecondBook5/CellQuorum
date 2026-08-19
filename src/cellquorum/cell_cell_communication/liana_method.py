"""LIANA consensus ligand-receptor method (per-sample rank_aggregate)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class LianaMethod(AnalysisMethod):
    """Per-sample LIANA rank_aggregate consensus → uns['liana_res'] + CSV."""

    name = "liana"
    stage_category = "cell_cell_communication"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        cell_type_col = config.get("cell_type_col", "cell_type")
        sample_col = config.get("sample_col", "sample_id")
        layer = config.get("layer", "cellquorum_normalized")
        return DataContract(
            required_obs=[cell_type_col, sample_col],
            required_layers=[layer] if layer != "X" else [],
            expression_layer=layer,
            expected_kind="lognorm",
        )

    def requires_obs(self, config: dict) -> list[str]:
        return [
            config.get("cell_type_col", "cell_type"),
            config.get("sample_col", "sample_id"),
        ]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        cell_type_col = config.get("cell_type_col", "cell_type")
        sample_col = config.get("sample_col", "sample_id")
        layer = config.get("layer", "cellquorum_normalized")
        seed = int(config.get("seed", 42))

        # Eligibility: need ≥2 cell types to have any inter-type communication.
        n_types = int(adata.obs[cell_type_col].nunique())
        if n_types < 2:
            return self._skip(f"need >=2 cell types, found {n_types}", n_cell_types=n_types)

        try:
            import liana as li
        except Exception as exc:  # pragma: no cover - env dependent
            return self._skip("liana unavailable", error=str(exc)[:300])

        from pandas import concat

        # Start from a clean slate so a skip never leaves a stale/partial result
        # for a downstream method (e.g. tensor_c2c) to trip over.
        adata.uns.pop("liana_res", None)

        # WHY we don't call li.mt.rank_aggregate.by_sample directly: liana's
        # by_sample loop has no per-sample error handling — if ANY single sample
        # fails its LR computation (e.g. ZeroDivisionError when a sample's
        # clusters are too sparse to score), the whole call aborts and leaves a
        # partial ``{sample: df}`` dict in uns['liana_res']. On sparse slices that
        # sinks the entire CCC stage. We iterate ourselves and tolerate per-sample
        # failures, keeping every sample that scores and recording the rest.
        samples = adata.obs[sample_col].astype("category").cat.categories
        per_sample: dict[str, object] = {}
        skipped: dict[str, str] = {}
        for sample in samples:
            sub = adata[adata.obs[sample_col] == sample]
            sub = sub.to_memory().copy() if sub.isbacked else sub.copy()
            # Inter-type communication needs >=2 cell types within the sample.
            if int(sub.obs[cell_type_col].nunique()) < 2:
                skipped[str(sample)] = "fewer than 2 cell types present"
                continue
            try:
                sample_res = li.mt.rank_aggregate(
                    sub,
                    groupby=cell_type_col,
                    resource_name=config.get("resource_name", "consensus"),
                    expr_prop=float(config.get("expr_prop", 0.1)),
                    min_cells=int(config.get("min_cells", 5)),
                    use_raw=False,
                    layer=layer if layer != "X" else None,
                    n_perms=int(config.get("n_perms", 100)),
                    seed=seed,
                    verbose=False,
                    inplace=False,
                )
            except Exception as exc:
                skipped[str(sample)] = f"{type(exc).__name__}: {str(exc)[:120]}"
                continue
            if sample_res is not None and len(sample_res) > 0:
                per_sample[str(sample)] = sample_res
            else:
                skipped[str(sample)] = "no interactions returned"

        if not per_sample:
            return self._skip(
                "no sample produced interactions",
                n_samples_total=int(len(samples)),
                n_samples_skipped=len(skipped),
            )

        # Concatenate per-sample frames into one long table with a "sample"
        # column (mirrors liana's own by_sample concat, minus the fragility).
        res = (
            concat(per_sample)
            .reset_index(level=1, drop=True)
            .reset_index()
            .rename(columns={"index": "sample"})
        )
        adata.uns["liana_res"] = res
        n_samples_scored = int(len(per_sample))

        artifacts: list[StageArtifact] = []
        try:
            results_dir = Path(context.paths.results) / "cell_cell_communication"
            results_dir.mkdir(parents=True, exist_ok=True)
            sort_cols = [c for c in ("sample", "magnitude_rank") if c in res.columns]
            ordered = res.sort_values(sort_cols, kind="mergesort") if sort_cols else res
            out_csv = results_dir / "liana_ranks.csv"
            ordered.to_csv(out_csv, index=False)
            artifacts.append(
                StageArtifact(
                    name="ccc_liana_ranks",
                    path=out_csv,
                    kind="csv",
                    description="Per-sample LIANA rank_aggregate consensus ranks.",
                )
            )
        except Exception as exc:
            # skip-not-crash: a write failure must not abort the stage.
            return StageResult(
                adata=adata,
                artifacts=[],
                notes=[f"liana ran but artifact write failed: {str(exc)[:200]}"],
                metrics={"method": self.name, "n_interactions": int(len(res))},
                backend="python",
            )

        n_samples = int(res["sample"].nunique()) if "sample" in res.columns else 0
        note = f"LIANA per-sample consensus over {n_samples} samples."
        if skipped:
            note += f" {len(skipped)} sample(s) skipped (too sparse to score)."
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[note],
            metrics={
                "method": self.name,
                "n_interactions": int(len(res)),
                "n_samples": n_samples,
                "n_samples_scored": n_samples_scored,
                "n_samples_skipped": len(skipped),
            },
            backend="python",
        )


__all__ = ["LianaMethod"]
