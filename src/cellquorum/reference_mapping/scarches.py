"""ScArchesMethod: scVI→scANVI→surgery label transfer with multi-seed consensus."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy.stats import entropy
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from cellquorum.compute.router import gpu_compute_available
from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class ScArchesMethod(AnalysisMethod):
    """scArches reference mapping: multi-seed scVI→scANVI→surgery with kNN uncertainty."""

    name = "scarches"
    stage_category = "reference_mapping"
    backend = "gpu"

    def input_contract(self, config: dict) -> DataContract:
        """Require the counts layer on the query."""
        counts_layer = config.get("counts_layer", "counts")
        return DataContract(
            required_layers=[counts_layer],
            expression_layer=counts_layer,
            expected_kind="counts",
        )

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        """Execute scVI→scANVI→query surgery with multi-seed consensus."""
        # Extract config params.
        atlas_h5ad = config.get("atlas_h5ad")
        if atlas_h5ad is None:
            return MethodSkip(
                reason="atlas_h5ad is None; reference mapping cannot proceed.",
                details={"method": self.name},
            )

        atlas_path = Path(atlas_h5ad)
        if not atlas_path.exists():
            return MethodSkip(
                reason=f"Atlas file not found: {atlas_path}",
                details={"method": self.name, "atlas_h5ad": str(atlas_path)},
            )

        label_key = config.get("label_key", "cell_type")
        atlas_batch_key = config.get("atlas_batch_key", "batch")
        query_batch_value = config.get("query_batch_value", "query")
        counts_layer = config.get("counts_layer", "counts")
        n_top_genes = int(config.get("n_top_genes", 3000))
        hvg_flavor = config.get("hvg_flavor", "seurat_v3")
        force_genes = config.get("force_genes", [])
        n_latent = int(config.get("n_latent", 30))
        n_layers = int(config.get("n_layers", 2))
        dropout_rate = float(config.get("dropout_rate", 0.2))
        gene_likelihood = config.get("gene_likelihood", "zinb")
        max_epochs_scvi = int(config.get("max_epochs_scvi", 400))
        max_epochs_scanvi = int(config.get("max_epochs_scanvi", 20))
        max_epochs_query = int(config.get("max_epochs_query", 100))
        early_stopping = bool(config.get("early_stopping", True))
        query_early_stopping_patience = int(config.get("query_early_stopping_patience", 10))
        query_early_stopping_monitor = config.get(
            "query_early_stopping_monitor", "reconstruction_loss_train"
        )
        unlabeled_category = config.get("unlabeled_category", "Unknown")
        seeds = config.get("seeds", [0, 1, 2, 3, 4])
        knn_k = int(config.get("knn_k", 30))
        key_added = config.get("key_added", "ref_state")
        compute_backend = config.get("compute_backend", "auto")
        write_loss_curves = bool(config.get("write_loss_curves", True))
        compartment_filter = config.get("compartment_filter")
        reference_filters = config.get("reference_filters", [])
        min_label_prob = config.get("min_label_prob")
        label_prob_col = config.get("label_prob_col")

        # Load atlas.
        atlas = sc.read_h5ad(atlas_path)

        # Apply generic filters to the reference atlas.
        if compartment_filter is not None:
            col = compartment_filter.get("column")
            keep = compartment_filter.get("keep")
            if col and keep and col in atlas.obs.columns:
                atlas = atlas[atlas.obs[col].isin(keep)].copy()

        for filt in reference_filters:
            col = filt.get("column")
            keep = filt.get("keep", [])
            if col and col in atlas.obs.columns:
                atlas = atlas[atlas.obs[col].isin(keep)].copy()

        if min_label_prob is not None and label_prob_col is not None:
            if label_prob_col in atlas.obs.columns:
                atlas = atlas[atlas.obs[label_prob_col] >= min_label_prob].copy()

        # Set atlas X to counts layer + copy labels.
        atlas.X = atlas.layers[counts_layer]
        atlas.obs["_labels"] = atlas.obs[label_key].astype(str).copy()

        # Gene intersection.
        shared = list(set(atlas.var_names) & set(adata.var_names))
        atlas = atlas[:, shared].copy()
        query = adata[:, shared].copy()

        # HVG selection.
        sc.pp.highly_variable_genes(
            atlas,
            n_top_genes=n_top_genes,
            flavor=hvg_flavor,
            layer=counts_layer,
        )
        hvg_mask = atlas.var["highly_variable"].values
        hvg_set = set(atlas.var_names[hvg_mask])

        # Add force_genes to HVG set.
        for gene in force_genes:
            if gene in atlas.var_names:
                hvg_set.add(gene)

        # Preserve order.
        hvg_list = [g for g in atlas.var_names if g in hvg_set]
        atlas = atlas[:, hvg_list].copy()
        query = query[:, hvg_list].copy()

        # GPU gate.
        if compute_backend == "cpu":
            use_gpu = False
        elif compute_backend == "gpu":
            use_gpu = True
        else:
            use_gpu = gpu_compute_available()

        accelerator = "gpu" if use_gpu else "cpu"

        # Per-seed training.
        seed_predictions = {}
        seed_latents = {}
        seed_loss_history = {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            for seed in seeds:
                scvi.settings.seed = seed

                # Setup scVI on atlas.
                scvi.model.SCVI.setup_anndata(
                    atlas,
                    batch_key=atlas_batch_key,
                    labels_key="_labels",
                )

                # Train scVI.
                vae = scvi.model.SCVI(
                    atlas,
                    n_latent=n_latent,
                    n_layers=n_layers,
                    dropout_rate=dropout_rate,
                    gene_likelihood=gene_likelihood,
                    encode_covariates=True,
                    deeply_inject_covariates=False,
                )
                vae.train(
                    max_epochs=max_epochs_scvi,
                    early_stopping=early_stopping,
                    accelerator=accelerator,
                )

                # Train scANVI from scVI.
                scanvi = scvi.model.SCANVI.from_scvi_model(
                    vae, unlabeled_category=unlabeled_category
                )
                scanvi.train(max_epochs=max_epochs_scanvi, accelerator=accelerator)

                # Prepare query.
                q = query.copy()
                q.X = q.layers[counts_layer]
                q.obs["_labels"] = unlabeled_category
                q.obs[atlas_batch_key] = query_batch_value

                # Query surgery.
                scvi.model.SCANVI.prepare_query_anndata(q, scanvi)
                qmodel = scvi.model.SCANVI.load_query_data(q, scanvi)
                qmodel.train(
                    max_epochs=max_epochs_query,
                    plan_kwargs={"weight_decay": 0.0},
                    early_stopping=early_stopping,
                    early_stopping_patience=query_early_stopping_patience,
                    early_stopping_monitor=query_early_stopping_monitor,
                    accelerator=accelerator,
                )

                # Predict labels.
                pred_hard = qmodel.predict(q)
                pred_soft = qmodel.predict(q, soft=True)

                # Get latents.
                q_latent = qmodel.get_latent_representation(q)
                ref_latent = scanvi.get_latent_representation(atlas)

                # kNN uncertainty (NOT softmax).
                nn = NearestNeighbors(n_neighbors=knn_k)
                nn.fit(ref_latent)
                _, idx = nn.kneighbors(q_latent)

                ref_labels = atlas.obs["_labels"].to_numpy()
                knn_entropy_vals = np.zeros(len(q))
                knn_agreement_vals = np.zeros(len(q))

                for i in range(len(q)):
                    neigh = ref_labels[idx[i]]
                    uniq, counts = np.unique(neigh, return_counts=True)
                    p = counts / knn_k
                    knn_entropy_vals[i] = entropy(p)
                    knn_agreement_vals[i] = (neigh == pred_hard[i]).mean()

                # Store per-seed results.
                seed_predictions[seed] = {
                    "hard": pred_hard,
                    "soft": pred_soft,
                    "knn_entropy": knn_entropy_vals,
                    "knn_agreement": knn_agreement_vals,
                }
                seed_latents[seed] = {"query": q_latent, "ref": ref_latent}

                # Loss history.
                seed_loss_history[seed] = {
                    "scvi": _serialize_history(vae),
                    "scanvi": _serialize_history(scanvi),
                    "query_surgery": _serialize_history(qmodel),
                }

        # Consensus across seeds.
        n_cells = len(query)
        consensus_labels = []
        consensus_fracs = []

        for i in range(n_cells):
            votes = [seed_predictions[s]["hard"][i] for s in seeds]
            counter = Counter(votes)
            majority, count = counter.most_common(1)[0]
            consensus_labels.append(majority)
            consensus_fracs.append(count / len(seeds))

        # Choose the best-agreeing seed for latent/kNN metrics.
        seed_mean_agreement = {s: seed_predictions[s]["knn_agreement"].mean() for s in seeds}
        best_seed = max(seed_mean_agreement, key=seed_mean_agreement.get)

        # Write results onto query.
        result_query = query.copy()
        result_query.obs[key_added] = pd.Categorical(consensus_labels)
        result_query.obs[f"{key_added}_consensus_frac"] = consensus_fracs
        result_query.obs[f"{key_added}_knn_entropy"] = seed_predictions[best_seed]["knn_entropy"]
        result_query.obs[f"{key_added}_knn_agreement"] = seed_predictions[best_seed][
            "knn_agreement"
        ]

        # Per-seed columns.
        for s in seeds:
            result_query.obs[f"{key_added}_seed{s}"] = pd.Categorical(seed_predictions[s]["hard"])

        # Softmax probs from best seed.
        soft_df = seed_predictions[best_seed]["soft"]
        for col in soft_df.columns:
            result_query.obs[f"refprob_{col}"] = soft_df[col].values

        # Latent embedding from best seed.
        result_query.obsm["X_scANVI"] = seed_latents[best_seed]["query"]

        # uns metadata.
        ref_states = list(atlas.obs["_labels"].unique())
        result_query.uns["reference_mapping"] = {
            "atlas_h5ad": str(atlas_path),
            "label_key": label_key,
            "n_latent": n_latent,
            "n_hvg": len(hvg_list),
            "n_shared": len(shared),
            "ref_states": ref_states,
            "seeds": seeds,
            "uncertainty_note": "kNN entropy from k-NN in reference latent space",
        }

        # Diagnostics: kNN accuracy.
        ref_latent_best = seed_latents[best_seed]["ref"]
        knn_clf = KNeighborsClassifier(n_neighbors=knn_k)
        cv_scores = cross_val_score(knn_clf, ref_latent_best, atlas.obs["_labels"], cv=3)
        knn_accuracy = cv_scores.mean()

        # Metrics.
        metrics = {
            "knn_accuracy": float(knn_accuracy),
            "median_knn_entropy": float(np.median(seed_predictions[best_seed]["knn_entropy"])),
            "median_knn_agreement": float(np.median(seed_predictions[best_seed]["knn_agreement"])),
            "median_consensus_frac": float(np.median(consensus_fracs)),
            "frac_unanimous": float((np.array(consensus_fracs) == 1.0).mean()),
            "n_ref_cells": int(len(atlas)),
            "n_hvg": len(hvg_list),
            "key_added": key_added,
        }

        # Write artifacts if figures path exists.
        artifacts = []
        if write_loss_curves and hasattr(context.paths, "figures"):
            figures_path = Path(context.paths.figures)
            if figures_path.exists():
                from cellquorum.reference_mapping.diagnostics import (
                    plot_loss_curves,
                    plot_uncertainty,
                )

                # Loss curves for best seed.
                loss_path = figures_path / f"{key_added}_loss_curves.png"
                plot_loss_curves(seed_loss_history[best_seed], loss_path)
                artifacts.append(
                    StageArtifact(
                        name="loss_curves",
                        path=loss_path,
                        kind="figure",
                        description="scVI/scANVI/query loss curves",
                    )
                )

                # Uncertainty histograms.
                uncertainty_path = figures_path / f"{key_added}_uncertainty.png"
                plot_uncertainty(result_query.obs, key_added, uncertainty_path)
                artifacts.append(
                    StageArtifact(
                        name="uncertainty",
                        path=uncertainty_path,
                        kind="figure",
                        description="kNN entropy + agreement + consensus distributions",
                    )
                )

        # Write loss history JSON to objects dir.
        if hasattr(context.paths, "objects"):
            objects_path = Path(context.paths.objects)
            if objects_path.exists():
                for s in seeds:
                    loss_json_path = objects_path / f"{key_added}_seed{s}_loss_history.json"
                    with open(loss_json_path, "w") as f:
                        json.dump(seed_loss_history[s], f, indent=2)

        notes = [
            f"Mapped {len(result_query)} cells to {len(ref_states)} reference states.",
            f"Multi-seed consensus across {len(seeds)} seeds.",
            f"Median kNN entropy: {metrics['median_knn_entropy']:.3f}",
            f"Median consensus fraction: {metrics['median_consensus_frac']:.3f}",
            f"kNN accuracy (CV): {knn_accuracy:.3f}",
        ]

        return StageResult(adata=result_query, metrics=metrics, artifacts=artifacts, notes=notes)


def _serialize_history(model: object) -> dict[str, list[float]]:
    """
    Serialize scvi model.history to {metric: [values]}.

    Args:
        model: scvi model with .history attribute.

    Returns:
        Dictionary of metric names to value lists.
    """
    hist = {}
    if hasattr(model, "history") and model.history is not None:
        for metric_name in model.history:
            df = model.history[metric_name]
            if metric_name in df.columns:
                hist[metric_name] = df[metric_name].tolist()
    return hist


__all__ = ["ScArchesMethod"]
