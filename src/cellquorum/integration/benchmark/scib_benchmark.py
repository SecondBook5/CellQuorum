"""scib-metrics integration-quality benchmark method."""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class ScibBenchmarkMethod(AnalysisMethod):
    """Evaluate integration quality via scib-metrics suite.

    Computes batch-correction metrics (iLISI/kBET/pcr_comparison) and
    biological-preservation metrics (cLISI/silhouette/graph-connectivity/NMI)
    for each configured embedding. Aggregates via weighted average and ranks
    embeddings by aggregate score. READ-ONLY: never modifies adata.

    Attributes:
        name: Method registry name.
        stage_category: Stage category (integration_benchmark).
        backend: Execution backend (python).
    """

    name = "scib_benchmark"
    stage_category = "integration_benchmark"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """Return contract: requires pre_embedding in obsm + batch_key in obs."""
        pre_embedding = config.get("pre_embedding", "X_pca")
        batch_key = config.get("batch_key", "batch")
        return DataContract(
            required_obsm=[pre_embedding],
            required_obs=[batch_key],
        )

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult:
        """Compute scib-metrics for each embedding; return ranking + metrics."""
        # Suppress JAX CPU fallback warning from scib_metrics.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                import scib_metrics.nearest_neighbors  # noqa: F401
            except ImportError:
                return self._fallback_backend(adata, config)

        # Extract config.
        batch_key = config.get("batch_key", "batch")
        label_key = config.get("label_key", "cell_type")
        label_key_fallback = config.get("label_key_fallback")
        pre_embedding = config.get("pre_embedding", "X_pca")
        embeddings = config.get("embeddings", [])
        n_neighbors = config.get("n_neighbors", 90)
        mode = config.get("mode", "full")
        batch_weight = config.get("batch_weight", 0.4)
        bio_weight = config.get("bio_weight", 0.6)

        # Resolve label column (label_key → fallback → None = batch-only).
        label_col = None
        if label_key in adata.obs.columns:
            label_col = label_key
        elif label_key_fallback and label_key_fallback in adata.obs.columns:
            label_col = label_key_fallback

        # Prepare batch/label arrays.
        batches = adata.obs[batch_key].to_numpy()
        labels = adata.obs[label_col].to_numpy() if label_col else None

        # Pre-embedding for pcr_comparison.
        X_pre = adata.obsm[pre_embedding]

        # Compute metrics for each embedding.
        notes = []
        embedding_metrics = {}
        for emb_key in embeddings:
            if emb_key not in adata.obsm:
                notes.append(f"Embedding '{emb_key}' not in obsm; skipped.")
                continue

            X_post = adata.obsm[emb_key]
            batch_metrics, bio_metrics, emb_notes = self._compute_metrics(
                X_pre=X_pre,
                X_post=X_post,
                batches=batches,
                labels=labels,
                n_neighbors=n_neighbors,
                mode=mode,
            )
            notes.extend(emb_notes)

            # Aggregate score.
            aggregate = self._aggregate_score(batch_metrics, bio_metrics, batch_weight, bio_weight)

            embedding_metrics[emb_key] = {
                "batch": batch_metrics,
                "bio": bio_metrics,
                "aggregate": aggregate,
            }

        # Rank embeddings by aggregate score (descending).
        ranking = sorted(
            embedding_metrics.keys(),
            key=lambda k: embedding_metrics[k]["aggregate"],
            reverse=True,
        )

        # Build metrics dict.
        metrics = {
            "embeddings": embedding_metrics,
            "ranking": ranking,
            "label_used": label_col,
            "mode": mode,
        }

        # READ-ONLY: return the SAME adata.
        return StageResult(adata=adata, metrics=metrics, notes=notes)

    def _compute_metrics(
        self,
        X_pre: np.ndarray,
        X_post: np.ndarray,
        batches: np.ndarray,
        labels: np.ndarray | None,
        n_neighbors: int,
        mode: str,
    ) -> tuple[dict[str, float], dict[str, float], list[str]]:
        """Compute batch and bio metrics for one embedding."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Batch metrics (label-free).
            # Bio metrics (require labels).
            from scib_metrics import (
                clisi_knn,
                graph_connectivity,
                ilisi_knn,
                isolated_labels,
                kbet,
                nmi_ari_cluster_labels_leiden,
                pcr_comparison,
                silhouette_batch,
                silhouette_label,
            )
            from scib_metrics.nearest_neighbors import pynndescent

        notes = []
        batch_metrics = {}
        bio_metrics = {}

        # Build kNN neighbors.
        nbrs = pynndescent(X_post, n_neighbors=n_neighbors)

        # --- Batch metrics (label-free) --- #
        batch_metrics["ilisi"] = self._safe_metric(
            ilisi_knn, (nbrs, batches), {"scale": True}, notes, "ilisi"
        )
        batch_metrics["kbet"] = self._safe_metric(
            kbet, (nbrs, batches), {"alpha": 0.05}, notes, "kbet"
        )
        # pcr_comparison needs integer covariate codes.
        import pandas as pd

        batch_codes = pd.factorize(batches)[0]
        batch_metrics["pcr"] = self._safe_metric(
            pcr_comparison,
            (X_pre, X_post, batch_codes),
            {"scale": True},
            notes,
            "pcr_comparison",
        )

        # graph_connectivity + silhouette_batch need labels.
        if labels is not None:
            batch_metrics["graph_connectivity"] = self._safe_metric(
                graph_connectivity, (nbrs, labels), {}, notes, "graph_connectivity"
            )
            batch_metrics["silhouette_batch"] = self._safe_metric(
                silhouette_batch, (X_post, labels, batches), {}, notes, "silhouette_batch"
            )

        # --- Bio metrics (require labels + full mode) --- #
        if labels is not None and mode == "full":
            bio_metrics["clisi"] = self._safe_metric(
                clisi_knn, (nbrs, labels), {"scale": True}, notes, "clisi"
            )
            bio_metrics["silhouette_label"] = self._safe_metric(
                silhouette_label, (X_post, labels), {}, notes, "silhouette_label"
            )
            bio_metrics["isolated_labels"] = self._safe_metric(
                isolated_labels, (X_post, labels, batches), {}, notes, "isolated_labels"
            )
            # nmi_ari_cluster_labels_leiden returns dict {"nmi": ..., "ari": ...}.
            nmi_ari = self._safe_metric(
                nmi_ari_cluster_labels_leiden, (nbrs, labels), {}, notes, "nmi_ari_leiden"
            )
            if isinstance(nmi_ari, dict):
                bio_metrics["nmi"] = nmi_ari.get("nmi", np.nan)
            else:
                bio_metrics["nmi"] = np.nan

        return batch_metrics, bio_metrics, notes

    def _safe_metric(
        self,
        func: object,
        args: tuple,
        kwargs: dict,
        notes: list[str],
        metric_name: str,
    ) -> float:
        """Call a metric function; record nan + note on failure."""
        try:
            result = func(*args, **kwargs)
            # Handle dict return (nmi_ari_leiden).
            if isinstance(result, dict):
                return result
            # Handle tuple/list return (kbet returns (acceptance_rate, stat,
            # pval); we use acceptance_rate).
            if isinstance(result, tuple | list):
                return float(result[0])
            return float(result)
        except Exception as e:
            notes.append(f"{metric_name} failed: {e}")
            return float("nan")

    def _aggregate_score(
        self, batch_metrics: dict, bio_metrics: dict, batch_weight: float, bio_weight: float
    ) -> float:
        """Aggregate finite metric values via weighted average."""
        batch_vals = [v for v in batch_metrics.values() if np.isfinite(v)]
        bio_vals = [v for v in bio_metrics.values() if np.isfinite(v)]

        batch_mean = np.nanmean(batch_vals) if batch_vals else np.nan
        bio_mean = np.nanmean(bio_vals) if bio_vals else np.nan

        # If both are nan, aggregate is nan.
        if np.isnan(batch_mean) and np.isnan(bio_mean):
            return np.nan

        # Replace nan with 0 for aggregation (only one family has values).
        batch_mean = 0.0 if np.isnan(batch_mean) else batch_mean
        bio_mean = 0.0 if np.isnan(bio_mean) else bio_mean

        return batch_weight * batch_mean + bio_weight * bio_mean

    def _fallback_backend(self, adata: ad.AnnData, config: dict) -> StageResult | MethodSkip:
        """Fallback to harmonypy.compute_lisi for ilisi/clisi; skip if unavailable."""
        try:
            from harmonypy import compute_lisi
        except ImportError:
            return MethodSkip(
                reason="scib_metrics not available and harmonypy fallback also missing.",
                details={"method": self.name},
            )

        # Extract config.
        batch_key = config.get("batch_key", "batch")
        label_key = config.get("label_key", "cell_type")
        label_key_fallback = config.get("label_key_fallback")
        embeddings = config.get("embeddings", [])

        # Resolve label column.
        label_col = None
        if label_key in adata.obs.columns:
            label_col = label_key
        elif label_key_fallback and label_key_fallback in adata.obs.columns:
            label_col = label_key_fallback

        # Compute iLISI (batch-mixing) only in harmonypy fallback.
        notes = [
            "Using harmonypy fallback: batch-mixing (iLISI) only; "
            "bio-conservation metrics unavailable."
        ]
        embedding_metrics = {}
        for emb_key in embeddings:
            if emb_key not in adata.obsm:
                notes.append(f"Embedding '{emb_key}' not in obsm; skipped.")
                continue

            X = adata.obsm[emb_key]
            metadata_df = adata.obs[[batch_key]]
            if label_col:
                metadata_df = adata.obs[[batch_key, label_col]]

            try:
                # iLISI: batch diversity per cell (higher=better batch mixing).
                lisi_batch = compute_lisi(X, metadata_df, [batch_key], perplexity=30)
                ilisi = float(np.mean(lisi_batch))
            except Exception as e:
                notes.append(f"iLISI failed for {emb_key}: {e}")
                ilisi = np.nan

            # cLISI: harmonypy fallback does not support bio-conservation metrics.
            # harmonypy compute_lisi returns raw LISI (lower=better, unbounded) on
            # the wrong scale vs scib_metrics clisi_knn (higher=better, 0-1 scaled).
            # Exclude bio metrics in fallback to avoid inverted/meaningless scores.

            # Aggregate: batch only (no bio metrics in harmonypy fallback).
            batch_metrics = {"ilisi": ilisi}
            bio_metrics = {}
            aggregate = self._aggregate_score(
                batch_metrics,
                bio_metrics,
                config.get("batch_weight", 0.4),
                config.get("bio_weight", 0.6),
            )

            embedding_metrics[emb_key] = {
                "batch": batch_metrics,
                "bio": bio_metrics,
                "aggregate": aggregate,
            }

        # Rank embeddings.
        ranking = sorted(
            embedding_metrics.keys(),
            key=lambda k: embedding_metrics[k]["aggregate"],
            reverse=True,
        )

        metrics = {
            "embeddings": embedding_metrics,
            "ranking": ranking,
            "label_used": label_col,
            "mode": "fallback",
        }

        return StageResult(adata=adata, metrics=metrics, notes=notes)


__all__ = ["ScibBenchmarkMethod"]
