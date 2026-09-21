#!/usr/bin/env python
"""CellQuorum in-env MetaCell2 backend.

Runs the standard MetaCell2 (Baran & Tanay) pipeline inside its isolated
environment: exclude cell-cycle/stress genes, exclude low-quality cells,
extract the clean sub-object, run divide_and_conquer_pipeline to partition it
into metacells, and collect the aggregated (pseudobulk) metacell object.
Cells MetaCell2 could not confidently place are written back with the
literal metacell_id "Outliers" -- never silently dropped.

Argument parsing and every skip-gate check happen BEFORE `import metacells`,
so the graceful-skip path is exercisable without the metacells package
installed (see tests/test_metacell2_script.py, which runs this script with
the caller's own bare interpreter).

Positional args:
    h5ad                   Path to the input AnnData (raw counts + rep_key obsm).
    out_dir                Directory for metacell_assignments.csv + metacell_adata.h5ad.
    rep_key                obsm key MetaCell2 uses for its similarity graph.
    layer                  Layer holding raw counts.
    group_by               obs column used as the per-cell group label (for the
                            assignment table's "group" column; MetaCell2 itself
                            does not need this -- it is carried through purely
                            for CellQuorum's own downstream QC-scatter figures).
    target_metacell_size    MetaCell2's target cells/metacell (int).
    min_cells              Minimum total cells required to attempt partitioning (int).
    seed                   Random seed (int).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _write_empty(out_dir: Path, reason: str) -> None:
    """Write the expected empty outputs for a harmless configuration/data skip."""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metacell_assignments.csv").write_text(
        "cell_id,metacell_id,group,n_umis\n", encoding="utf-8"
    )
    (out_dir / "metacells_SKIPPED.txt").write_text(
        f"MetaCell2 skipped (isolated backend, no downstream effect): {reason}\n",
        encoding="utf-8",
    )
    print(f"[metacells] SKIPPED gracefully: {reason}")


def main(argv: list[str]) -> int:
    if len(argv) != 8:
        print(
            "usage: run_metacell2.py <h5ad> <out_dir> <rep_key> <layer> <group_by> "
            "<target_metacell_size> <min_cells> <seed>",
            file=sys.stderr,
        )
        return 2

    h5ad_path, out_dir_s, rep_key, layer, group_by, target_size_s, min_cells_s, seed_s = argv
    out_dir = Path(out_dir_s)

    try:
        target_metacell_size = int(target_size_s)
        min_cells = int(min_cells_s)
        seed = int(seed_s)
    except ValueError as exc:
        _write_empty(out_dir, f"non-integer numeric argument: {exc}")
        return 0

    # ---- Skip-gate 1: metacells package not importable ---- #
    try:
        import metacells as mc
    except ImportError as exc:
        _write_empty(out_dir, f"metacells package not importable: {exc}")
        return 0

    # ---- Skip-gate 2: input file missing/unreadable ---- #
    try:
        import anndata as ad

        full = ad.read_h5ad(h5ad_path)
    except Exception as exc:  # noqa: BLE001 -- any read failure is a graceful skip
        _write_empty(out_dir, f"could not read input h5ad: {exc}")
        return 0

    # ---- Skip-gate 3: too few cells ---- #
    if full.n_obs < min_cells:
        _write_empty(out_dir, f"too few cells ({full.n_obs} < {min_cells})")
        return 0

    # ---- Skip-gate 4: required obsm/obs/layer missing ---- #
    if rep_key not in full.obsm:
        _write_empty(out_dir, f"obsm key '{rep_key}' absent")
        return 0
    if group_by not in full.obs.columns:
        _write_empty(out_dir, f"obs column '{group_by}' absent")
        return 0
    if layer != "X" and layer not in full.layers:
        _write_empty(out_dir, f"layer '{layer}' absent")
        return 0

    # ---- Run the MetaCell2 pipeline ---- #
    import numpy as np

    if layer != "X":
        full.X = full.layers[layer]

    try:
        mc.ut.set_name(full, "full")

        # Generic cell-cycle/stress gene exclusion -- these patterns are biology-
        # agnostic housekeeping/stress markers common to the standard MetaCell2
        # tutorial, not study-specific to any one project.
        mc.pl.exclude_genes(
            full,
            excluded_gene_patterns=["MT-.*", "RP[LS].*"],
            random_seed=seed,
        )
        mc.pl.exclude_cells(full, properly_sampled_min_cell_total=None)
        clean = mc.pl.extract_clean_data(full, name="clean")

        mc.pl.divide_and_conquer_pipeline(
            clean,
            target_metacell_size=target_metacell_size,
            random_seed=seed,
        )
        metacell_adata = mc.pl.collect_metacells(clean, name="metacells")

        # ---- Build the per-cell assignment table, including Outliers ---- #
        n_umis = np.asarray(full.X.sum(axis=1)).ravel()
        groups = full.obs[group_by].astype(str).to_numpy()
        cell_ids = full.obs_names.to_numpy()

        clean_metacell_of_cell = clean.obs.get("metacell_name")
        metacell_by_cell_id = (
            dict(zip(clean.obs_names, clean_metacell_of_cell, strict=False))
            if clean_metacell_of_cell is not None
            else {}
        )
        metacell_ids = [metacell_by_cell_id.get(cid, "Outliers") or "Outliers" for cid in cell_ids]

        import pandas as pd

        assignments = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "metacell_id": metacell_ids,
                "group": groups,
                "n_umis": n_umis,
            }
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(out_dir / "metacell_assignments.csv", index=False)
        metacell_adata.write_h5ad(out_dir / "metacell_adata.h5ad")
    except Exception as exc:  # noqa: BLE001 -- a real pipeline failure, not a graceful skip
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metacells_FAILED.txt").write_text(
            f"MetaCell2 pipeline run failed: {exc}\n", encoding="utf-8"
        )
        print(f"[metacells] FAILED: {exc}", file=sys.stderr)
        return 1

    print(
        f"[metacells] wrote {assignments['metacell_id'].nunique()} metacell groups "
        f"({(assignments['metacell_id'] == 'Outliers').sum()} outlier cells) to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
