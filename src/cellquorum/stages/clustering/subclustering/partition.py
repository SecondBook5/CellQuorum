"""Partition methods for subclustering (CHOIR + fallback grid)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.backends.script_paths import r_script_path
from cellquorum.core.exceptions import CellQuorumBackendError
from cellquorum.core.h5ad_io import write_h5ad
from cellquorum.methods.base import MethodSkip

if TYPE_CHECKING:
    from cellquorum.stages.clustering.subclustering.config import (
        PartitionConfig,
        SubclusteringConfig,
    )

# Path to bundled R scripts.
_CHOIR_R = r_script_path("choir.R")
_SCSHC_TEST_R = r_script_path("scshc_test.R")


def run_choir(
    adata: ad.AnnData,
    config: SubclusteringConfig,
    backend: object | None,
    scratch_dir: Path,
    reduction_key: str | None = None,
) -> ad.AnnData | MethodSkip:
    """
    Run CHOIR permutation-tested clustering on focus subset.

    CHOIR does its own normalization + dim-reduction + batch correction from
    raw counts. Feed it adata.layers['counts'] + optional batch labels.

    Args:
        adata: Focus subset with counts layer.
        config: Subclustering configuration.
        backend: Rscript backend from context registry (or None).
        scratch_dir: Scratch directory for temp files.
        reduction_key: obsm key of a precomputed (ideally batch-corrected)
            embedding to cluster on. Requires var['highly_variable']; when either
            is missing CHOIR computes its own uncorrected reduction instead.

    Returns:
        adata with obs[key_added] cluster labels, or MethodSkip if unavailable.

    Raises:
        CellQuorumBackendError: If CHOIR fails after starting.
    """
    # Skip if backend unavailable.
    if backend is None or shutil.which("Rscript") is None:
        return MethodSkip(
            reason="CHOIR partition skipped: Rscript unavailable",
            details={"method": "choir"},
        )

    # Check CHOIR R package availability.
    if not backend._r_package_available("CHOIR"):
        return MethodSkip(
            reason="CHOIR partition skipped: CHOIR R package unavailable",
            details={"method": "choir"},
        )

    # Validate counts layer exists.
    counts_layer = config.counts_layer
    if counts_layer not in adata.layers:
        raise CellQuorumBackendError(
            f"CHOIR requires counts layer '{counts_layer}'; " f"not found in adata.layers"
        )

    # Resolve CHOIR config.
    partition_config: PartitionConfig = config.partition
    choir_config = partition_config.choir
    alpha = choir_config.get("alpha", 0.05)
    n_iterations = choir_config.get("n_iterations", 100)
    n_trees = choir_config.get("n_trees", 50)
    # CHOIR requires positive integer seed (not 0).
    raw_seed = partition_config.seeds[0] if partition_config.seeds else 1
    seed = max(1, raw_seed)

    # Resolve batch labels (use donor_gate.group_key if available).
    batch_key = config.donor_gate.group_key
    batch_arg = batch_key if batch_key and batch_key in adata.obs.columns else "NONE"

    # Prepare scratch files.
    scratch_dir.mkdir(parents=True, exist_ok=True)
    in_h5ad = scratch_dir / "choir_input.h5ad"
    out_csv = scratch_dir / "choir_output.csv"

    # Write adata with counts as X (zellkonverter reads X → first assay → counts).
    choir_adata = ad.AnnData(X=adata.layers[counts_layer].copy())
    # Plain-object string indices: a pandas nullable StringArray index (which real
    # objects carry) is refused by the h5ad writer, and zellkonverter reads it as
    # a categorical group rather than as cell names.
    choir_adata.obs_names = pd.Index([str(n) for n in adata.obs_names], dtype=object)
    choir_adata.var_names = pd.Index([str(n) for n in adata.var_names], dtype=object)

    # Add batch column if batch correction requested.
    if batch_arg != "NONE":
        choir_adata.obs[batch_key] = adata.obs[batch_key].astype(str).to_numpy()

    # Ship a precomputed batch-corrected embedding when one is available. This is
    # what makes the cluster count trustworthy: CHOIR's internal Harmony path is
    # unusable against harmony >= 1.0 (it calls the removed HarmonyMatrix()), so
    # without an embedding CHOIR silently clusters uncorrected data and can certify
    # single-donor clusters as significant. CHOIR also requires var_features
    # whenever a reduction is supplied, hence the highly_variable flag.
    reduction_arg = "NONE"
    if (
        reduction_key
        and reduction_key in adata.obsm
        and "highly_variable" in adata.var
        and bool(np.asarray(adata.var["highly_variable"]).sum() >= 2)
    ):
        choir_adata.obsm[reduction_key] = np.asarray(adata.obsm[reduction_key])
        choir_adata.var["highly_variable"] = np.asarray(adata.var["highly_variable"]).astype(bool)
        reduction_arg = reduction_key

    # Shared writer: it opts in to nullable strings and coerces the columns h5py
    # refuses, so a poisoned column fails here with a clear error instead of
    # reaching CHOIR as a half-written file. See cellquorum.core.h5ad_io.
    write_h5ad(choir_adata, in_h5ad)

    # Build R script args.
    args = [
        str(in_h5ad),
        str(out_csv),
        config.key_added,
        str(alpha),
        str(n_iterations),
        str(n_trees),
        batch_arg,
        str(seed),
        reduction_arg,
    ]

    # Run choir.R.
    timeout = choir_config.get("timeout_seconds", 1800)
    result = backend.run_script(_CHOIR_R, args, timeout=timeout)

    if result.returncode != 0:
        return MethodSkip(
            reason="CHOIR partition skipped: choir.R script failed",
            details={"method": "choir", "stderr": result.stderr.strip()[:500]},
        )

    # Read cluster labels CSV (barcode, subcluster).
    labels_df = _read_barcode_csv(out_csv, value_col="subcluster")

    # Join labels onto adata.obs by barcode (fail-loud on misalignment).
    result_adata = _join_labels_by_barcode(
        adata, labels_df, key_added=config.key_added, method="choir"
    )

    return result_adata


def run_scshc_test(
    adata: ad.AnnData,
    cluster_key: str,
    config: SubclusteringConfig,
    backend: object | None,
    scratch_dir: Path,
    batch_key: str | None = None,
) -> dict | MethodSkip:
    """
    Run sc-SHC formal significance test on supplied cluster labels.

    sc-SHC tests whether each split in a hierarchical clustering is
    statistically significant (permutation test).

    The interesting output is not the per-split p-values but the *reconciled*
    labels: sc-SHC merges every pair of input clusters whose split it cannot
    support, so the number of surviving labels is the partition the test is
    willing to defend. Reporting only "0 of 1 splits significant" hides the
    consequence — that a lineage came out of the test as one undivided
    population — behind a fraction, and a downstream stage that keys a headline
    table on the input labels has no way to notice.

    Args:
        adata: Focus subset with counts layer + cluster labels.
        cluster_key: obs column with cluster labels to test.
        config: Subclustering configuration.
        backend: Rscript backend from context registry (or None).
        scratch_dir: Scratch directory for temp files.
        batch_key: obs column sc-SHC should condition its null model on, so a
            batch effect is not reported as a supported split. ``None`` falls
            back to ``config.donor_gate.group_key`` for callers that predate
            this argument; the stage resolves it through the cohort block and
            passes it explicitly, because a field named for the donor gate is
            not where a *batch* key belongs.

    Returns:
        dict with per-split significance results, or MethodSkip if unavailable.

    Raises:
        CellQuorumBackendError: If sc-SHC fails after starting.
    """
    # Skip if backend unavailable.
    if backend is None or shutil.which("Rscript") is None:
        return MethodSkip(
            reason="sc-SHC test skipped: Rscript unavailable",
            details={"method": "scshc"},
        )

    # Check scSHC R package availability.
    if not backend._r_package_available("scSHC"):
        return MethodSkip(
            reason="sc-SHC test skipped: scSHC R package unavailable",
            details={"method": "scshc"},
        )

    # Validate counts layer exists.
    counts_layer = config.counts_layer
    if counts_layer not in adata.layers:
        raise CellQuorumBackendError(
            f"sc-SHC requires counts layer '{counts_layer}'; not found in adata.layers"
        )

    # Validate cluster_key exists.
    if cluster_key not in adata.obs.columns:
        raise CellQuorumBackendError(
            f"sc-SHC test requires cluster column '{cluster_key}'; " f"not found in adata.obs"
        )

    # Resolve sc-SHC config.
    formal_test_config = config.formal_test
    alpha = formal_test_config.alpha

    # Resolve batch labels.
    if batch_key is None:
        batch_key = config.donor_gate.group_key
    batch_arg = batch_key if batch_key and batch_key in adata.obs.columns else "NONE"

    # Prepare scratch files.
    scratch_dir.mkdir(parents=True, exist_ok=True)
    in_h5ad = scratch_dir / "scshc_input.h5ad"
    clusters_csv = scratch_dir / "scshc_clusters.csv"
    out_csv = scratch_dir / "scshc_output.csv"

    # Write adata with counts as X.
    scshc_adata = ad.AnnData(X=adata.layers[counts_layer].copy())
    scshc_adata.obs_names = adata.obs_names
    scshc_adata.var_names = adata.var_names

    # Add batch column if batch correction requested.
    if batch_arg != "NONE":
        scshc_adata.obs[batch_key] = adata.obs[batch_key].values

    write_h5ad(scshc_adata, in_h5ad)

    # Write cluster labels CSV (barcode, cluster).
    cluster_df = pd.DataFrame(
        {"barcode": adata.obs_names, "cluster": adata.obs[cluster_key].values}
    )
    cluster_df.to_csv(clusters_csv, index=False)

    # Build R script args.
    args = [str(in_h5ad), str(clusters_csv), str(out_csv), str(alpha), batch_arg]

    # Run scshc_test.R.
    timeout = 1800
    result = backend.run_script(_SCSHC_TEST_R, args, timeout=timeout)

    if result.returncode != 0:
        return MethodSkip(
            reason="sc-SHC test skipped: scshc_test.R script failed",
            details={"method": "scshc", "stderr": result.stderr.strip()[:500]},
        )

    # Read per-split significance CSV.
    sig_df = pd.read_csv(out_csv)

    # Build significance dict.
    n_significant = int(sig_df["significant"].sum()) if not sig_df.empty else 0
    significance = {
        "method": "scshc",
        "alpha": alpha,
        "batch_key": batch_arg,
        "n_clusters_in": int(pd.Series(adata.obs[cluster_key]).nunique()),
        "n_splits_tested": len(sig_df),
        "n_significant": n_significant,
        "per_split": sig_df.to_dict(orient="records"),
    }

    # Read the reconciled labels the R script wrote alongside the split table and
    # attach them to obs. Without this the merge decision lives only in a scratch
    # file: a run whose eight clusters all collapsed to one looked, from every
    # persisted artifact, exactly like a run whose eight clusters were upheld.
    labels_csv = out_csv.with_name(out_csv.name.replace(".csv", "_labels.csv"))
    if labels_csv.exists():
        labels = pd.read_csv(labels_csv).set_index("barcode")["scshc_label"]
        aligned = labels.reindex(adata.obs_names)
        adata.obs[f"{cluster_key}_scshc"] = pd.Categorical(aligned)
        surviving = int(aligned.dropna().nunique())
        significance["n_labels_surviving"] = surviving
        significance["labels_key"] = f"{cluster_key}_scshc"
        # The headline reading, stated rather than left to be derived: this is
        # the difference between "the partition survived" and "there is no
        # partition", and it is the number a write-up has to quote.
        significance["merged_to_one"] = surviving == 1 and significance["n_clusters_in"] > 1

    return significance


def _read_barcode_csv(path: Path, value_col: str) -> pd.DataFrame:
    """
    Read barcode-indexed CSV (barcode, value_col).

    Args:
        path: CSV file path.
        value_col: Column name to extract (e.g., 'subcluster').

    Returns:
        DataFrame indexed by barcode with value_col.
    """
    # Find the barcode column (case-insensitive).
    temp_df = pd.read_csv(path, nrows=0)
    barcode_col = None
    for col in temp_df.columns:
        if col.lower() in ("barcode", "cell"):
            barcode_col = col
            break

    if barcode_col is None:
        raise ValueError(f"CSV missing barcode column: {path}")

    # Read CSV with barcode as string (matching adata.obs_names).
    df = pd.read_csv(path, dtype={barcode_col: str})

    if value_col not in df.columns:
        raise ValueError(f"CSV missing value column '{value_col}': {path}")

    # Set barcode as index and return value_col.
    df = df.set_index(barcode_col)[[value_col]]
    return df


def _join_labels_by_barcode(
    adata: ad.AnnData,
    labels_df: pd.DataFrame,
    key_added: str,
    method: str,
) -> ad.AnnData:
    """
    Join barcode-indexed labels onto adata.obs (fail-loud on misalignment).

    Args:
        adata: Input AnnData.
        labels_df: DataFrame indexed by barcode with label column.
        key_added: obs column name to add.
        method: Method name (for error messages).

    Returns:
        adata with obs[key_added] set.

    Raises:
        CellQuorumBackendError: If barcode sets do not match.
    """
    # Reindex labels to match adata.obs_names order.
    labels_df = labels_df.reindex(adata.obs_names)

    # Validate barcode alignment (all cells must have labels).
    n_missing = labels_df.isnull().all(axis=1).sum()
    if n_missing > 0:
        raise CellQuorumBackendError(
            f"{method} barcode misalignment: {n_missing} cells missing labels "
            f"after reindex. R script barcodes do not match adata.obs_names."
        )

    # Assign labels to obs.
    result_adata = adata.copy()
    result_adata.obs[key_added] = labels_df.iloc[:, 0].to_numpy()

    return result_adata


__all__ = ["run_choir", "run_scshc_test"]
