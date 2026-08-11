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


def _load_base_grn(co, organism: str):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202 -- pragma: no cover
    """Load CellOracle's built-in promoter base GRN for the organism.

    Returns the base-GRN DataFrame, or None if no loader matches the organism.
    Loader names and default genome versions are taken verbatim from
    ``celloracle.data.load_promoter_base_GRN`` (hg19 human / mm10 mouse).
    """
    org = (organism or "").strip().lower()
    loaders = {
        "human": "load_human_promoter_base_GRN",
        "hsapiens": "load_human_promoter_base_GRN",
        "hs": "load_human_promoter_base_GRN",
        "mouse": "load_mouse_promoter_base_GRN",
        "mmusculus": "load_mouse_promoter_base_GRN",
        "mm": "load_mouse_promoter_base_GRN",
        "rat": "load_rat_promoter_base_GRN",
        "zebrafish": "load_zebrafish_promoter_base_GRN",
        "drosophila": "load_drosophila_promoter_base_GRN",
        "chicken": "load_chicken_promoter_base_GRN",
    }
    loader_name = loaders.get(org)
    if loader_name is None:
        return None
    loader = getattr(co.data, loader_name, None)
    if loader is None:
        return None
    return loader()


def _run_celloracle(args: argparse.Namespace) -> None:  # pragma: no cover
    """CellOracle-dependent workflow: GRN inference → KO simulation → scoring.

    Mirrors the maintainers' canonical Network-analysis + Gata1-KO tutorials:
    Oracle → import counts + base GRN → PCA → kNN imputation → get_links →
    filter_links → get_cluster_specific_TFdict_from_Links → fit_GRN_for_simulation,
    then per TF: simulate_shift → estimate_transition_prob → calculate_embedding_shift,
    reading the per-cell shift field from ``oracle.delta_embedding``.

    Not covered by CI: requires a live CellOracle environment.
    """
    import anndata
    import celloracle as co
    import numpy as np
    import pandas as pd

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data and the built-in base GRN for this organism.
    print(f"[celloracle] Loading {args.h5ad}")
    adata = anndata.read_h5ad(args.h5ad)

    cluster_key = args.cluster_key if args.cluster_key else "cluster"
    embedding_key = args.embedding_key if args.embedding_key else "X_umap"

    if embedding_key not in adata.obsm:
        write_skip(out_dir, args.tag, f"embedding {embedding_key} not found in adata.obsm")
        return

    # CellOracle needs a real cluster column; synthesize a single group if the
    # resolved key is absent (the "all"/no-clustering generic fallback).
    if cluster_key not in adata.obs.columns:
        adata.obs[cluster_key] = "all"
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

    try:
        base_grn = _load_base_grn(co, args.organism)
    except Exception as e:
        write_skip(out_dir, args.tag, f"base GRN unavailable for organism {args.organism}: {e}")
        return
    if base_grn is None or len(base_grn) == 0:
        write_skip(out_dir, args.tag, f"no base GRN for organism {args.organism!r}")
        return

    # Step 2: Build Oracle and import counts + base GRN.
    print(f"[celloracle] Building Oracle (cluster_key={cluster_key}, embedding={embedding_key})")
    oracle = co.Oracle()
    oracle.import_anndata_as_raw_count(
        adata=adata, cluster_column_name=cluster_key, embedding_name=embedding_key
    )
    oracle.import_TF_data(TF_info_matrix=base_grn)

    # Step 3: PCA + kNN imputation (tutorial-style component/k auto-selection).
    oracle.perform_PCA()
    evr = np.cumsum(oracle.pca.explained_variance_ratio_)
    cand = np.where(np.diff(np.diff(evr) > 0.002))[0]
    n_comps = int(cand[0]) if len(cand) else 50
    n_comps = max(1, min(n_comps, 50))
    n_cell = oracle.adata.shape[0]
    k = max(1, int(0.025 * n_cell))
    print(f"[celloracle] kNN imputation (k={k}, n_pca_dims={n_comps}, n_cells={n_cell})")
    oracle.knn_imputation(
        n_pca_dims=n_comps,
        k=k,
        balanced=True,
        b_sight=min(k * 8, n_cell - 1),
        b_maxl=min(k * 4, n_cell - 1),
        n_jobs=4,
    )

    # Step 4: Cluster-specific GRN → filter → simulation-ready model.
    print(f"[celloracle] Inferring cluster-specific GRN (get_links, cluster={cluster_key})")
    links = oracle.get_links(cluster_name_for_GRN_unit=cluster_key, alpha=10, verbose_level=0)
    links.filter_links(p=0.001, weight="coef_abs", threshold_number=args.n_top_targets)
    oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
    oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True, verbose_level=0)

    # GRN summary: per (cluster, TF) target count, from the filtered edge lists.
    # Each filtered_links[cluster] is a DataFrame with source/target/coef columns.
    grn_rows = []
    for cluster, df in links.filtered_links.items():
        if df is None or len(df) == 0:
            continue
        counts = df.groupby("source").size()
        for tf, n_targets in counts.items():
            grn_rows.append({"cluster": cluster, "tf": str(tf), "n_targets": int(n_targets)})
    if grn_rows:
        pd.DataFrame(grn_rows).to_csv(out_dir / "grn_summary.csv", index=False)
        print(f"[celloracle] Wrote grn_summary.csv ({len(grn_rows)} rows)")

    # Step 5: TF set to screen — regulators in the fitted GRN present as genes.
    fitted_tfs: set[str] = set()
    for df in links.filtered_links.values():
        if df is not None and len(df) > 0:
            fitted_tfs.update(str(s) for s in df["source"].unique())
    available_tfs = fitted_tfs & set(map(str, oracle.adata.var_names))

    tf_list = _parse_tf_list(args.tf_list)
    if tf_list:
        tfs_to_screen = [tf for tf in tf_list if tf in available_tfs]
        if not tfs_to_screen:
            write_skip(out_dir, args.tag, f"no requested TFs available (requested: {tf_list})")
            return
    else:
        tfs_to_screen = sorted(available_tfs)
        if not tfs_to_screen:
            write_skip(out_dir, args.tag, "no TFs to screen in fitted GRN")
            return
    print(f"[celloracle] Screening {len(tfs_to_screen)} TFs")

    # Step 6: Scoring geometry. delta_embedding aligns with oracle.adata cells.
    directional = bool(args.condition_key and args.healthy_label)
    direction_label = "directional" if directional else "magnitude"
    obs = oracle.adata.obs
    emb = np.asarray(oracle.adata.obsm[embedding_key])

    direction_unit = None
    if directional:
        if args.condition_key not in obs.columns:
            write_skip(out_dir, args.tag, f"condition key {args.condition_key} not in adata.obs")
            return
        healthy_mask = (obs[args.condition_key].astype(str) == args.healthy_label).to_numpy()
        if not healthy_mask.any():
            write_skip(out_dir, args.tag, f"no cells with healthy label {args.healthy_label}")
            return
        if healthy_mask.all():
            write_skip(out_dir, args.tag, f"all cells are healthy (label {args.healthy_label})")
            return
        # 2D embedding axis from diseased centroid -> healthy centroid.
        healthy_centroid = emb[healthy_mask, :2].mean(axis=0)
        diseased_centroid = emb[~healthy_mask, :2].mean(axis=0)
        direction_vec = healthy_centroid - diseased_centroid
        direction_norm = float(np.linalg.norm(direction_vec))
        if direction_norm == 0:
            write_skip(out_dir, args.tag, "healthy and diseased centroids coincide")
            return
        direction_unit = direction_vec / direction_norm

    # Step 7: Simulate each TF KO -> per-cell shift field -> score.
    n_neighbors = min(int(args.knn_n_neighbors), n_cell - 1)
    tf_scores = []
    for tf in tfs_to_screen:
        print(f"[celloracle] Simulating KO: {tf}")
        try:
            oracle.simulate_shift(
                perturb_condition={tf: 0.0}, n_propagation=int(args.n_propagation)
            )
            oracle.estimate_transition_prob(
                n_neighbors=n_neighbors,
                knn_random=True,
                sampled_fraction=1,
                random_seed=int(args.seed),
            )
            oracle.calculate_embedding_shift(sigma_corr=0.05)
        except Exception as e:  # one infeasible TF must not sink the screen
            print(f"[celloracle] skip {tf}: {type(e).__name__}: {e}")
            continue

        shift_vectors = np.asarray(oracle.delta_embedding)
        if shift_vectors.ndim != 2 or shift_vectors.shape[1] < 2:
            print(f"[celloracle] skip {tf}: unexpected delta_embedding shape")
            continue

        pd.DataFrame(
            shift_vectors,
            index=oracle.adata.obs_names,
            columns=[f"d{i}" for i in range(shift_vectors.shape[1])],
        ).to_parquet(out_dir / f"shift_vectors_{tf}.parquet")

        if directional:
            score = float(np.mean(shift_vectors[:, :2] @ direction_unit))
        else:
            score = float(np.mean(np.linalg.norm(shift_vectors[:, :2], axis=1)))
        tf_scores.append(
            {"tf": tf, "score": score, "n_cells": int(n_cell), "direction": direction_label}
        )

    if not tf_scores:
        write_skip(out_dir, args.tag, "no TF simulations produced results")
        return

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
