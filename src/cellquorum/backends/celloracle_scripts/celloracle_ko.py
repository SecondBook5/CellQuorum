#!/usr/bin/env python
"""CellQuorum in-env CellOracle in-silico KO backend.

Runs CellOracle's three-phase workflow in an isolated frozen env: (1) infer a
simulation-ready GRN from counts + the built-in promoter base GRN, (2) simulate
each TF knockout by zeroing the TF and propagating the signal, (3) score/rank the
knockouts by disease->healthy shift.

Two failure regimes, deliberately distinct (mirrors the pySCENIC backend):
  - NOT CONFIGURED (celloracle absent, base GRN unavailable, nothing fit): writes
    an empty perturbation_ranking.csv + perturbation_SKIPPED_{tag}.txt and exits 0
    -> harmless skip, nothing else affected.
  - REAL FAILURE (celloracle present but the GRN fit / simulation itself raises):
    writes perturbation_FAILED_{tag}.txt and exits NON-ZERO.

Scoring: with --condition-key AND --healthy-label, each TF's score is the mean
projection of its per-cell shift vectors onto the unit diseased->healthy centroid
axis in the embedding (direction="directional"); otherwise the mean shift
magnitude (direction="magnitude").
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_RANKING_HEADER = ["tf", "score", "n_cells", "direction"]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the CellOracle KO script."""
    p = argparse.ArgumentParser(description="CellOracle in-silico TF knockout")
    p.add_argument("--h5ad", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--organism", default="human")
    p.add_argument("--cluster-key", default="")
    p.add_argument("--rep-key", default="")
    p.add_argument("--embedding-key", default="")
    p.add_argument("--condition-key", default="")
    p.add_argument("--healthy-label", default="")
    p.add_argument("--tf-list", default="", help="space/comma-separated; empty -> screen all")
    p.add_argument("--n-top-targets", type=int, default=20)
    p.add_argument("--knn-n-neighbors", type=int, default=200)
    p.add_argument("--n-propagation", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p


def write_skip(out_dir: Path, tag: str, reason: str) -> None:
    """Write the empty ranking schema + a skip marker (harmless configuration skip)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "perturbation_ranking.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_RANKING_HEADER)
    (out_dir / f"perturbation_SKIPPED_{tag}.txt").write_text(
        f"CellOracle skipped (isolated backend, no downstream effect): {reason}\n",
        encoding="utf-8",
    )
    print(f"[celloracle] SKIPPED gracefully: {reason}")


def write_ranking(out_dir: Path, tag: str, rows: list[dict]) -> Path:
    """Write the ranked-target table. Returns the written path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "perturbation_ranking.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_RANKING_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _RANKING_HEADER})
    return path


def _parse_tf_list(raw: str) -> list[str]:
    """Parse a space/comma-separated TF list; empty -> [] (means screen all)."""
    if not raw:
        return []
    return [t for t in raw.replace(",", " ").split() if t]


def _run_celloracle(args: argparse.Namespace) -> None:  # pragma: no cover
    """CellOracle-dependent workflow: GRN inference → KO simulation → scoring.

    This function is not covered by CI tests as it requires a real CellOracle environment.
    """
    import anndata
    import celloracle as co
    import numpy as np
    import pandas as pd

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data and check base GRN availability
    print(f"[celloracle] Loading {args.h5ad}")
    adata = anndata.read_h5ad(args.h5ad)

    # Resolve keys (use args if provided, else CellOracle defaults)
    cluster_key = args.cluster_key if args.cluster_key else "louvain"
    rep_key = args.rep_key if args.rep_key else "X_pca"
    embedding_key = args.embedding_key if args.embedding_key else "X_umap"

    # Check if base GRN is available for the organism
    try:
        base_grn = co.data.load_TF_info_ref(args.organism)
    except Exception as e:
        write_skip(out_dir, args.tag, f"base GRN unavailable for organism {args.organism}: {e}")
        return

    if base_grn is None or len(base_grn) == 0:
        write_skip(out_dir, args.tag, f"base GRN unavailable for organism {args.organism}")
        return

    # Step 2: Build Oracle, fit GRN
    print(f"[celloracle] Building Oracle with cluster_key={cluster_key}, rep_key={rep_key}")
    oracle = co.Oracle()

    # Import the AnnData
    oracle.import_anndata_as_raw_count(
        adata=adata, cluster_column_name=cluster_key, embedding_name=embedding_key
    )

    # Import reference base GRN
    oracle.import_TF_data(TF_info_matrix=base_grn)

    # Perform PCA
    oracle.perform_PCA()

    # kNN imputation with seeded reproducibility
    print(f"[celloracle] kNN imputation (n_neighbors={args.knn_n_neighbors}, seed={args.seed})")
    oracle.knn_imputation(n_neighbors=args.knn_n_neighbors, random_state=args.seed)

    # Get cluster-specific links
    print(f"[celloracle] Fitting cluster-specific GRN links (n_top={args.n_top_targets})")
    links = oracle.get_cluster_specific_TFdict_from_Links(
        oracle_object=oracle, n_top=args.n_top_targets
    )

    # Write GRN summary
    grn_rows = []
    for cluster, tf_dict in links.items():
        for tf, targets in tf_dict.items():
            grn_rows.append(
                {"cluster": cluster, "tf": tf, "n_targets": len(targets) if targets else 0}
            )

    if grn_rows:
        grn_df = pd.DataFrame(grn_rows)
        grn_df.to_csv(out_dir / "grn_summary.csv", index=False)
        print(f"[celloracle] Wrote grn_summary.csv ({len(grn_rows)} rows)")

    # Write per-cluster links
    for cluster, tf_dict in links.items():
        rows = []
        for tf, targets in tf_dict.items():
            if targets:
                for target in targets:
                    rows.append({"tf": tf, "target": target})
        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(out_dir / f"links_{cluster}.parquet")

    # Step 3: Determine TF set to screen
    tf_list = _parse_tf_list(args.tf_list)

    # Get all TFs present in fitted GRN
    fitted_tfs = set()
    for tf_dict in links.values():
        fitted_tfs.update(tf_dict.keys())

    # Filter to TFs present in both GRN and gene expression
    available_tfs = fitted_tfs & set(adata.var_names)

    if tf_list:
        # User specified TFs - filter to available ones
        tfs_to_screen = [tf for tf in tf_list if tf in available_tfs]
        if not tfs_to_screen:
            write_skip(out_dir, args.tag, f"no requested TFs available (requested: {tf_list})")
            return
    else:
        # Screen all available TFs
        tfs_to_screen = sorted(available_tfs)
        if not tfs_to_screen:
            write_skip(out_dir, args.tag, "no TFs to screen")
            return

    print(f"[celloracle] Screening {len(tfs_to_screen)} TFs")

    # Step 4: Simulate KO for each TF and compute shift vectors
    # Determine if we're doing directional or magnitude scoring
    directional = bool(args.condition_key and args.healthy_label)
    direction_label = "directional" if directional else "magnitude"

    # Get embedding coordinates
    if embedding_key not in adata.obsm:
        write_skip(out_dir, args.tag, f"embedding {embedding_key} not found in adata.obsm")
        return

    embedding = adata.obsm[embedding_key]

    # Compute healthy centroid if directional
    if directional:
        if args.condition_key not in adata.obs:
            write_skip(
                out_dir, args.tag, f"condition key {args.condition_key} not found in adata.obs"
            )
            return

        healthy_mask = adata.obs[args.condition_key] == args.healthy_label
        diseased_mask = ~healthy_mask

        if not healthy_mask.any():
            write_skip(out_dir, args.tag, f"no cells with healthy label {args.healthy_label}")
            return

        if not diseased_mask.any():
            write_skip(out_dir, args.tag, f"all cells are healthy (label {args.healthy_label})")
            return

        healthy_centroid = embedding[healthy_mask].mean(axis=0)
        diseased_centroid = embedding[diseased_mask].mean(axis=0)

        # Unit direction vector from diseased to healthy
        direction_vec = healthy_centroid - diseased_centroid
        direction_norm = np.linalg.norm(direction_vec)
        if direction_norm == 0:
            write_skip(out_dir, args.tag, "healthy and diseased centroids coincide")
            return

        direction_unit = direction_vec / direction_norm

    # Simulate each TF KO
    tf_scores = []
    for tf in tfs_to_screen:
        print(f"[celloracle] Simulating KO: {tf}")

        # Simulate shift by setting TF to 0
        oracle.simulate_shift(
            perturb_condition={tf: 0.0}, n_propagation=args.n_propagation, random_state=args.seed
        )

        # Get simulated embedding (oracle stores this in oracle.delta_embedding or similar)
        # The shift vectors are the difference between perturbed and original embeddings
        # CellOracle stores these in oracle.delta_embedding
        if not hasattr(oracle, "delta_embedding") or oracle.delta_embedding is None:
            print(f"[celloracle] WARNING: No delta_embedding for {tf}, skipping")
            continue

        shift_vectors = oracle.delta_embedding

        # Write shift vectors
        shift_df = pd.DataFrame(
            shift_vectors,
            index=adata.obs_names,
            columns=[f"d{i}" for i in range(shift_vectors.shape[1])],
        )
        shift_df.to_parquet(out_dir / f"shift_vectors_{tf}.parquet")

        # Score the TF
        if directional:
            # Project shift onto disease->healthy axis
            projections = shift_vectors @ direction_unit
            score = float(np.mean(projections))
        else:
            # Mean magnitude
            magnitudes = np.linalg.norm(shift_vectors, axis=1)
            score = float(np.mean(magnitudes))

        tf_scores.append(
            {"tf": tf, "score": score, "n_cells": len(adata), "direction": direction_label}
        )

    if not tf_scores:
        write_skip(out_dir, args.tag, "no TF simulations produced results")
        return

    # Step 5: Sort by score descending and write ranking
    tf_scores.sort(key=lambda x: x["score"], reverse=True)

    write_ranking(out_dir, args.tag, tf_scores)
    print(f"[celloracle] Wrote perturbation_ranking.csv ({len(tf_scores)} TFs)")


def main() -> None:
    """Run the CellOracle KO workflow, or skip gracefully."""
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)

    # Heavy import gate: absence of CellOracle is a harmless skip, not a failure.
    try:
        import celloracle as co  # noqa: F401
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import scanpy as sc  # noqa: F401
    except Exception as e:  # pragma: no cover - exercised only where celloracle absent
        write_skip(out_dir, args.tag, f"celloracle import failed: {type(e).__name__}: {e}")
        sys.exit(0)

    try:
        _run_celloracle(args)  # pragma: no cover - requires a real celloracle env
    except Exception as e:  # pragma: no cover
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"perturbation_FAILED_{args.tag}.txt").write_text(
            f"celloracle KO failed: {type(e).__name__}: {e}\n", encoding="utf-8"
        )
        print(f"[celloracle] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
