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

    def requires_layers(self) -> list[str]:
        return ["cellquorum_normalized"]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        cell_type_col = config.get("cell_type_col", "cell_type")
        sample_col = config.get("sample_col", "sample_id")
        layer = config.get("layer", "cellquorum_normalized")
        seed = int(config.get("seed", 42))

        # Eligibility: need ≥2 cell types to have any inter-type communication.
        n_types = int(adata.obs[cell_type_col].nunique())
        if n_types < 2:
            return MethodSkip(
                reason=f"liana skipped: need >=2 cell types, found {n_types}",
                details={"method": self.name, "n_cell_types": n_types},
            )

        try:
            import liana as li
        except Exception as exc:  # pragma: no cover - env dependent
            return MethodSkip(
                reason="liana skipped: liana unavailable",
                details={"method": self.name, "error": str(exc)[:300]},
            )

        try:
            li.mt.rank_aggregate.by_sample(
                adata,
                sample_key=sample_col,
                groupby=cell_type_col,
                resource_name=config.get("resource_name", "consensus"),
                expr_prop=float(config.get("expr_prop", 0.1)),
                min_cells=int(config.get("min_cells", 5)),
                use_raw=False,
                layer=layer if layer != "X" else None,
                n_perms=int(config.get("n_perms", 100)),
                seed=seed,
                key_added="liana_res",
                verbose=False,
                inplace=True,
            )
        except Exception as exc:
            return MethodSkip(
                reason="liana skipped: rank_aggregate failed",
                details={"method": self.name, "error": str(exc)[:300]},
            )

        res = adata.uns.get("liana_res")
        if res is None or len(res) == 0:
            return MethodSkip(
                reason="liana skipped: no interactions returned",
                details={"method": self.name},
            )

        # Rename sample_key to "sample" for consistency across stages
        if sample_col in res.columns:
            res = res.rename(columns={sample_col: "sample"})
            adata.uns["liana_res"] = res

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

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"LIANA per-sample consensus over {int(res['sample'].nunique())} samples."],
            metrics={
                "method": self.name,
                "n_interactions": int(len(res)),
                "n_samples": int(res["sample"].nunique()) if "sample" in res.columns else 0,
            },
            backend="python",
        )


__all__ = ["LianaMethod"]
