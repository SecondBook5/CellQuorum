"""ScArchesMethod: scVI→scANVI→surgery label transfer with multi-seed consensus."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy.stats import entropy
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

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
        resume = bool(config.get("resume", True))
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

        # Guard: atlas must have >= knn_k cells after filtering.
        if len(atlas) < knn_k:
            return MethodSkip(
                reason=f"reference_mapping skipped: filtered atlas has {len(atlas)} "
                f"cells < knn_k={knn_k}",
                details={"method": self.name, "n_atlas_cells": len(atlas), "knn_k": knn_k},
            )

        # Set atlas X to counts layer + copy labels.
        atlas.X = atlas.layers[counts_layer]
        atlas.obs["_labels"] = atlas.obs[label_key].astype(str).copy()

        # Gene intersection.
        shared = sorted(set(atlas.var_names) & set(adata.var_names))

        # Guard: must have shared genes.
        if len(shared) == 0:
            return MethodSkip(
                reason="reference_mapping skipped: no shared genes between atlas and query "
                "(check gene ID types: symbols vs Ensembl)",
                details={
                    "method": self.name,
                    "n_atlas_genes": len(atlas.var_names),
                    "n_query_genes": len(adata.var_names),
                },
            )

        atlas_train = atlas[:, shared].copy()
        query_full = adata.copy()  # Keep the full input for return.
        query_train = adata[:, shared].copy()

        # HVG selection.
        sc.pp.highly_variable_genes(
            atlas_train,
            n_top_genes=n_top_genes,
            flavor=hvg_flavor,
            layer=counts_layer,
        )
        hvg_mask = atlas_train.var["highly_variable"].values
        hvg_set = set(atlas_train.var_names[hvg_mask])

        # Add force_genes to HVG set.
        for gene in force_genes:
            if gene in atlas_train.var_names:
                hvg_set.add(gene)

        # Preserve order.
        hvg_list = [g for g in atlas_train.var_names if g in hvg_set]
        atlas_train = atlas_train[:, hvg_list].copy()
        query_train = query_train[:, hvg_list].copy()

        # GPU gate. scVI uses PyTorch/Lightning, not RAPIDS/CuPy.
        if compute_backend == "cpu":
            use_gpu = False
        elif compute_backend == "gpu":
            if not _scvi_gpu_available():
                raise RuntimeError(
                    "reference_mapping.compute_backend='gpu' was requested, but "
                    "PyTorch/Lightning does not report a supported CUDA accelerator."
                )
            use_gpu = True
        else:
            use_gpu = _scvi_gpu_available()

        accelerator = "gpu" if use_gpu else "cpu"

        # Resolve the per-seed checkpoint directory. These checkpoints are
        # intentionally stored under objects because they are machine-readable
        # intermediate state, not final results.
        objects_path = None
        if hasattr(context.paths, "objects"):
            objects_path = Path(context.paths.objects)
            objects_path.mkdir(parents=True, exist_ok=True)

        checkpoint_meta = _build_seed_checkpoint_meta(
            atlas_path=atlas_path,
            label_key=label_key,
            counts_layer=counts_layer,
            key_added=key_added,
            query_obs_names=query_train.obs_names,
            atlas_obs_names=atlas_train.obs_names,
            hvg_list=hvg_list,
            n_shared=len(shared),
            knn_k=knn_k,
            n_latent=n_latent,
        )

        # Per-seed training.
        seed_predictions = {}
        seed_latents = {}
        seed_loss_history = {}
        resumed_seeds: list[int] = []
        trained_seeds: list[int] = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            for seed in seeds:
                if resume and objects_path is not None:
                    checkpoint = _load_seed_checkpoint(
                        objects_path=objects_path,
                        key_added=key_added,
                        seed=int(seed),
                        expected_meta=checkpoint_meta,
                    )
                    if checkpoint is not None:
                        seed_predictions[seed] = checkpoint["prediction"]
                        seed_latents[seed] = checkpoint["latent"]
                        seed_loss_history[seed] = checkpoint["loss_history"]
                        resumed_seeds.append(int(seed))
                        continue

                scvi.settings.seed = seed

                # Setup scVI on atlas.
                scvi.model.SCVI.setup_anndata(
                    atlas_train,
                    batch_key=atlas_batch_key,
                    labels_key="_labels",
                )

                # Train scVI.
                vae = scvi.model.SCVI(
                    atlas_train,
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

                # Persist the trained reference models (scVI + scANVI) so a run is
                # reproducible and the reference can be reused across queries or
                # recovered after a crash without retraining. Best-effort: a save
                # failure must not abort mapping.
                if objects_path is not None:
                    _save_reference_models(
                        objects_path=objects_path,
                        key_added=key_added,
                        seed=int(seed),
                        scvi_model=vae,
                        scanvi_model=scanvi,
                    )

                # Prepare query.
                q = query_train.copy()
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
                ref_latent = scanvi.get_latent_representation(atlas_train)

                # kNN uncertainty (NOT softmax).
                nn = NearestNeighbors(n_neighbors=knn_k)
                nn.fit(ref_latent)
                _, idx = nn.kneighbors(q_latent)

                ref_labels = atlas_train.obs["_labels"].to_numpy()
                knn_entropy_vals = np.zeros(len(q))
                knn_agreement_vals = np.zeros(len(q))

                for i in range(len(q)):
                    neigh = ref_labels[idx[i]]
                    _, counts = np.unique(neigh, return_counts=True)
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
                trained_seeds.append(int(seed))

                if objects_path is not None:
                    _save_seed_checkpoint(
                        objects_path=objects_path,
                        key_added=key_added,
                        seed=int(seed),
                        meta=checkpoint_meta,
                        prediction=seed_predictions[seed],
                        latent=seed_latents[seed],
                        loss_history=seed_loss_history[seed],
                    )

        # Consensus across seeds.
        n_cells = len(query_train)

        # Mean-posterior consensus: average each seed's soft probabilities over a
        # shared, union label-column ordering, then derive the label and
        # confidence from the AVERAGED distribution. This is the scientifically
        # correct multi-seed consensus — the reported probabilities, label, and
        # confidence are mutually consistent, unlike picking one "best" seed's
        # probs alongside a separate hard vote.
        mean_soft_df = _mean_soft_probabilities(seed_predictions, seeds)
        prob_columns = list(mean_soft_df.columns)
        prob_matrix = mean_soft_df.to_numpy()
        consensus_labels = [prob_columns[i] for i in prob_matrix.argmax(axis=1)]
        # Confidence = the averaged posterior mass on the consensus label.
        consensus_confidence = prob_matrix.max(axis=1)

        # Hard-vote agreement is retained as a secondary robustness signal: the
        # fraction of seeds whose hard call matched the mean-posterior label.
        consensus_fracs = []
        for i in range(n_cells):
            votes = [seed_predictions[s]["hard"][i] for s in seeds]
            consensus_fracs.append(sum(v == consensus_labels[i] for v in votes) / len(seeds))

        # Choose the best-agreeing seed for the (single-seed) latent + kNN metrics.
        # NOTE: X_scANVI stays single-seed because independent scANVI latent spaces
        # are not rotation/sign-aligned and cannot be averaged; it is labeled as
        # such in uns. Only the probabilities are a true cross-seed consensus.
        seed_mean_agreement = {s: seed_predictions[s]["knn_agreement"].mean() for s in seeds}
        best_seed = max(seed_mean_agreement, key=seed_mean_agreement.get)

        # Write results onto the FULL input query (all genes, not HVG subset).
        # Cells are NOT subset (only genes were), so obs_names order is preserved.
        result_query = query_full.copy()
        assert len(result_query) == n_cells, "Cell count mismatch after gene subset."
        result_query.obs[key_added] = pd.Categorical(consensus_labels)
        result_query.obs[f"{key_added}_consensus_frac"] = consensus_fracs
        result_query.obs[f"{key_added}_confidence"] = consensus_confidence
        result_query.obs[f"{key_added}_knn_entropy"] = seed_predictions[best_seed]["knn_entropy"]
        result_query.obs[f"{key_added}_knn_agreement"] = seed_predictions[best_seed][
            "knn_agreement"
        ]

        # Per-seed columns.
        for s in seeds:
            result_query.obs[f"{key_added}_seed{s}"] = pd.Categorical(seed_predictions[s]["hard"])

        # Mean-posterior consensus probabilities (consistent with the label above).
        for col in prob_columns:
            result_query.obs[f"refprob_{col}"] = mean_soft_df[col].to_numpy()
        result_query.obsm[f"{key_added}_probabilities"] = prob_matrix

        # Latent embedding from the best-agreeing seed (single-seed; see note).
        result_query.obsm["X_scANVI"] = seed_latents[best_seed]["query"]

        # uns metadata.
        ref_states = list(atlas_train.obs["_labels"].unique())
        result_query.uns["reference_mapping"] = {
            "atlas_h5ad": str(atlas_path),
            "label_key": label_key,
            "n_latent": n_latent,
            "n_hvg": len(hvg_list),
            "n_shared": len(shared),
            "ref_states": ref_states,
            "seeds": seeds,
            "probability_obsm": f"{key_added}_probabilities",
            "probability_columns": [str(col) for col in prob_columns],
            "probability_consensus": "mean_posterior_across_seeds",
            "embedding_note": (
                "X_scANVI is from a single (best-agreeing) seed; scANVI latent "
                "spaces across seeds are not aligned and are not averaged."
            ),
            "uncertainty_note": "kNN entropy from k-NN in reference latent space",
        }

        # Diagnostics: kNN accuracy (guard against rare class / small fold NaN).
        ref_latent_best = seed_latents[best_seed]["ref"]
        n_ref = len(atlas_train)
        cv_folds = 3
        safe_k = max(1, min(knn_k, n_ref // (cv_folds + 1)))
        knn_clf = KNeighborsClassifier(n_neighbors=safe_k)
        cv_scores = cross_val_score(
            knn_clf, ref_latent_best, atlas_train.obs["_labels"], cv=cv_folds
        )
        knn_accuracy = cv_scores.mean()
        if np.isnan(knn_accuracy):
            knn_accuracy = None

        # Metrics.
        metrics = {
            "knn_accuracy": float(knn_accuracy) if knn_accuracy is not None else None,
            "median_knn_entropy": float(np.median(seed_predictions[best_seed]["knn_entropy"])),
            "median_knn_agreement": float(np.median(seed_predictions[best_seed]["knn_agreement"])),
            "median_consensus_frac": float(np.median(consensus_fracs)),
            "median_consensus_confidence": float(np.median(consensus_confidence)),
            "frac_unanimous": float((np.array(consensus_fracs) == 1.0).mean()),
            "n_ref_cells": int(len(atlas_train)),
            "n_hvg": len(hvg_list),
            "key_added": key_added,
            "resumed_seeds": resumed_seeds,
            "trained_seeds": trained_seeds,
            "resume_enabled": resume,
        }

        # Write final reference-mapping artifacts.
        artifacts = []
        if hasattr(context.paths, "results"):
            results_path = Path(context.paths.results)
            if results_path.exists():
                assignment_cols = [
                    col
                    for col in result_query.obs.columns
                    if col == key_added
                    or col.startswith(f"{key_added}_")
                    or col.startswith("refprob_")
                ]
                assignments = result_query.obs[assignment_cols].copy()
                assignments.insert(0, "cell_id", result_query.obs_names)
                assignments_path = results_path / f"{key_added}_assignments.csv"
                assignments.to_csv(assignments_path, index=False)
                artifacts.append(
                    StageArtifact(
                        name="reference_mapping_assignments",
                        path=assignments_path,
                        kind="csv",
                        description="Per-cell transferred labels and uncertainty scores.",
                    )
                )

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

        notes = [
            f"Mapped {len(result_query)} cells to {len(ref_states)} reference states.",
            f"Multi-seed consensus across {len(seeds)} seeds.",
            f"Reference mapping seeds trained={trained_seeds}, resumed={resumed_seeds}.",
            f"Median kNN entropy: {metrics['median_knn_entropy']:.3f}",
            f"Median consensus fraction: {metrics['median_consensus_frac']:.3f}",
        ]
        if knn_accuracy is not None:
            notes.append(f"kNN accuracy (CV): {knn_accuracy:.3f}")
        else:
            notes.append("kNN accuracy: N/A (too few cells per class for k-fold CV)")

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


def _mean_soft_probabilities(seed_predictions: dict, seeds: list) -> pd.DataFrame:
    """
    Average per-seed soft label probabilities into a single consensus matrix.

    Each seed's ``prediction["soft"]`` is a (cells x labels) DataFrame. Seeds may
    in principle order or include label columns differently, so this aligns them
    to the union of label columns (missing columns treated as zero probability),
    sums, divides by the seed count, and renormalizes each row to sum to 1. The
    result is a mean-posterior distribution per cell — the basis for a consistent
    consensus label + confidence.

    Args:
        seed_predictions: {seed: {"soft": DataFrame, ...}}.
        seeds: Ordered list of seeds to average over.

    Returns:
        (cells x labels) DataFrame of mean-posterior probabilities.
    """

    # Union of label columns across seeds, in first-seen order for determinism.
    columns: list[str] = []
    for seed in seeds:
        for col in seed_predictions[seed]["soft"].columns:
            if col not in columns:
                columns.append(col)

    # Row index (cells) comes from the first seed's soft frame.
    index = seed_predictions[seeds[0]]["soft"].index

    # Sum each seed's soft frame reindexed to the shared column set.
    accumulator = np.zeros((len(index), len(columns)), dtype=float)
    for seed in seeds:
        soft = seed_predictions[seed]["soft"]
        aligned = soft.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=float)
        accumulator += aligned

    mean = accumulator / len(seeds)

    # Renormalize rows so each cell's posterior sums to 1 (guards tiny drift and
    # the fill-zero case). Zero-sum rows (should not occur) stay zero.
    row_sums = mean.sum(axis=1, keepdims=True)
    mean = np.divide(mean, row_sums, out=np.zeros_like(mean), where=row_sums > 0)

    return pd.DataFrame(mean, index=index, columns=columns)


def _save_reference_models(
    *,
    objects_path: Path,
    key_added: str,
    seed: int,
    scvi_model: object,
    scanvi_model: object,
) -> None:
    """
    Persist the trained reference scVI + scANVI models under ``objects/``.

    Saving the reference models makes a run reproducible and lets the trained
    reference be reused across query datasets or recovered after a crash without
    retraining. This is best-effort: scvi's ``save`` writes a directory, and any
    failure here is swallowed so it never aborts label transfer.

    Args:
        objects_path: The run's objects directory.
        key_added: Output key prefix (namespaces the saved model dirs).
        seed: Seed whose reference models are being saved.
        scvi_model: Trained scVI model.
        scanvi_model: Trained scANVI model.
    """

    try:
        scvi_dir = objects_path / f"{key_added}_seed{seed}_scvi_reference"
        scanvi_dir = objects_path / f"{key_added}_seed{seed}_scanvi_reference"
        scvi_model.save(str(scvi_dir), overwrite=True)
        scanvi_model.save(str(scanvi_dir), overwrite=True)
    except Exception:  # pragma: no cover - persistence is best-effort
        pass


def _build_seed_checkpoint_meta(
    *,
    atlas_path: Path,
    label_key: str,
    counts_layer: str,
    key_added: str,
    query_obs_names: Sequence[str],
    atlas_obs_names: Sequence[str],
    hvg_list: Sequence[str],
    n_shared: int,
    knn_k: int,
    n_latent: int,
) -> dict[str, object]:
    """Build metadata used to validate per-seed resume checkpoints."""

    return {
        "version": 1,
        "atlas_h5ad": str(atlas_path),
        "label_key": label_key,
        "counts_layer": counts_layer,
        "key_added": key_added,
        "n_query_cells": len(query_obs_names),
        "n_ref_cells": len(atlas_obs_names),
        "n_hvg": len(hvg_list),
        "n_shared": int(n_shared),
        "knn_k": int(knn_k),
        "n_latent": int(n_latent),
        "query_obs_digest": _digest_strings(query_obs_names),
        "atlas_obs_digest": _digest_strings(atlas_obs_names),
        "hvg_digest": _digest_strings(hvg_list),
    }


def _digest_strings(values: Sequence[str]) -> str:
    """Return a stable digest for an ordered string sequence."""

    digest = sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _seed_checkpoint_paths(
    *,
    objects_path: Path,
    key_added: str,
    seed: int,
) -> tuple[Path, Path]:
    """Return the NPZ and JSON paths for one seed checkpoint."""

    stem = f"{key_added}_seed{seed}_checkpoint"
    return objects_path / f"{stem}.npz", objects_path / f"{stem}.json"


def _save_seed_checkpoint(
    *,
    objects_path: Path,
    key_added: str,
    seed: int,
    meta: dict[str, object],
    prediction: dict[str, object],
    latent: dict[str, np.ndarray],
    loss_history: dict[str, dict[str, list[float]]],
) -> None:
    """Persist one completed seed so reruns can skip retraining it."""

    checkpoint_path, metadata_path = _seed_checkpoint_paths(
        objects_path=objects_path,
        key_added=key_added,
        seed=seed,
    )
    soft = prediction["soft"]
    if not isinstance(soft, pd.DataFrame):
        raise TypeError("Seed checkpoint prediction['soft'] must be a pandas DataFrame.")

    metadata = {
        **meta,
        "seed": int(seed),
        "soft_columns": [str(col) for col in soft.columns],
    }

    tmp_checkpoint = checkpoint_path.with_suffix(".npz.tmp")
    tmp_metadata = metadata_path.with_suffix(".json.tmp")

    with open(tmp_metadata, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    with open(tmp_checkpoint, "wb") as f:
        np.savez_compressed(
            f,
            hard=np.asarray(prediction["hard"]).astype(str),
            soft=soft.to_numpy(),
            knn_entropy=np.asarray(prediction["knn_entropy"], dtype=float),
            knn_agreement=np.asarray(prediction["knn_agreement"], dtype=float),
            query_latent=np.asarray(latent["query"], dtype=float),
            ref_latent=np.asarray(latent["ref"], dtype=float),
        )
    tmp_metadata.replace(metadata_path)
    tmp_checkpoint.replace(checkpoint_path)

    loss_path = objects_path / f"{key_added}_seed{seed}_loss_history.json"
    tmp_loss_path = loss_path.with_suffix(".json.tmp")
    with open(tmp_loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    tmp_loss_path.replace(loss_path)


def _load_seed_checkpoint(
    *,
    objects_path: Path,
    key_added: str,
    seed: int,
    expected_meta: dict[str, object],
) -> dict[str, object] | None:
    """Load a seed checkpoint when it matches the current run inputs."""

    checkpoint_path, metadata_path = _seed_checkpoint_paths(
        objects_path=objects_path,
        key_added=key_added,
        seed=seed,
    )
    if not checkpoint_path.exists() or not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        return None

    expected = {**expected_meta, "seed": int(seed)}
    for key, value in expected.items():
        if metadata.get(key) != value:
            return None

    soft_columns = metadata.get("soft_columns")
    if not isinstance(soft_columns, list) or not all(isinstance(c, str) for c in soft_columns):
        return None

    try:
        with np.load(checkpoint_path, allow_pickle=False) as data:
            hard = data["hard"].astype(str)
            soft = pd.DataFrame(data["soft"], columns=soft_columns)
            knn_entropy = data["knn_entropy"].astype(float)
            knn_agreement = data["knn_agreement"].astype(float)
            query_latent = data["query_latent"].astype(float)
            ref_latent = data["ref_latent"].astype(float)
    except Exception:
        return None

    n_query = int(expected_meta["n_query_cells"])
    n_ref = int(expected_meta["n_ref_cells"])
    n_latent = int(expected_meta["n_latent"])
    if hard.shape[0] != n_query:
        return None
    if soft.shape[0] != n_query:
        return None
    if knn_entropy.shape[0] != n_query or knn_agreement.shape[0] != n_query:
        return None
    if query_latent.shape != (n_query, n_latent):
        return None
    if ref_latent.shape != (n_ref, n_latent):
        return None

    loss_path = objects_path / f"{key_added}_seed{seed}_loss_history.json"
    loss_history: dict[str, dict[str, list[float]]] = {}
    if loss_path.exists():
        try:
            loss_history = json.loads(loss_path.read_text())
        except Exception:
            loss_history = {}

    return {
        "prediction": {
            "hard": hard,
            "soft": soft,
            "knn_entropy": knn_entropy,
            "knn_agreement": knn_agreement,
        },
        "latent": {"query": query_latent, "ref": ref_latent},
        "loss_history": loss_history,
    }


def _scvi_gpu_available() -> bool:
    """
    Return whether scVI can use a CUDA accelerator in this process.

    scVI trains through PyTorch Lightning. This check intentionally does not use
    the RAPIDS/CuPy router because RAPIDS availability is unrelated to whether
    scVI can train on CUDA.
    """

    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return False
    except Exception:
        return False

    try:
        from lightning.pytorch.accelerators import CUDAAccelerator

        return bool(CUDAAccelerator.is_available())
    except Exception:
        return True


__all__ = ["ScArchesMethod"]
