#!/usr/bin/env python3
"""Prepare the GSE140819 tutorial object: four donors, each with whole-cell and nuclei libraries.

GSE140819 (Slyper et al. 2020, Nat Med) compares single-cell and single-nucleus RNA-seq on the
same tumour samples. Eight libraries come from four donors that each have both preparations,
which makes a paired design: the donor is the unit, "cells" is the control and "nuclei" is the
case. The question the tutorial asks is a technical one: which cell types does each preparation
capture, and does gene expression shift with it?

Steps: load the eight libraries with the authors' annotations, name the columns the config
expects, keep the raw counts in a layer, and (optionally) subsample so a first run finishes in
minutes.

Usage:
    python scripts/prepare_gse140819_tutorial.py                 # ~15,000 cells, fast
    python scripts/prepare_gse140819_tutorial.py --max-cells 0   # every annotated cell (37,024)

The first call downloads the 1.2 GB GEO archive (set CELLQUORUM_DATA_HOME to choose where).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np

from cellquorum import datasets
from cellquorum.datasets import _gse140819 as gse

#: Four donors, each with a whole-cell library and a nuclei library from the same sample.
LIBRARIES = [
    "CLL-pilot1_fresh",
    "CLL-pilot1_TST",
    "HTAPP-656-SMP-3481_fresh-T1",
    "HTAPP-656-SMP-3481_TST",
    "HTAPP-963-SMP-4741_fresh",
    "HTAPP-963-SMP-4741_TST",
    "HTAPP-StJude-SMP-PDX1_cell",
    "HTAPP-StJude-SMP-PDX1_nuclei",
]
CONDITION = {"scRNA": "cells", "snRNA": "nuclei"}


def prepare(max_cells: int = 15_000, seed: int = 7) -> ad.AnnData:
    """Return the tutorial AnnData (raw counts in ``X`` and in ``layers['counts']``)."""
    adata = datasets.gse140819(LIBRARIES, barcodes="annotated")
    adata.obs_names_make_unique()
    adata.obs["sample"] = adata.obs["sample_id"].astype(str)
    adata.obs["donor"] = adata.obs["donor_id"].astype(str)
    adata.obs["condition"] = adata.obs["assay"].map(CONDITION).astype(str)
    adata.obs["cell_type"] = adata.obs["truth_annotate"].astype(str)

    if max_cells and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        keep = []
        for _, idx in adata.obs.groupby("sample", observed=True).indices.items():
            n = max(1, round(max_cells * len(idx) / adata.n_obs))
            keep.extend(rng.choice(idx, size=min(n, len(idx)), replace=False))
        adata = adata[np.sort(keep)].copy()

    adata.layers["counts"] = adata.X.copy()
    return adata


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--max-cells",
        type=int,
        default=15_000,
        help="subsample to about this many cells; 0 keeps all",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .h5ad (default: <data home>/gse140819/tutorial.h5ad)",
    )
    a = ap.parse_args()

    adata = prepare(a.max_cells, a.seed)
    out = a.out or gse.dataset_dir() / "tutorial.h5ad"
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    print(f"wrote {out}: {adata.n_obs} cells x {adata.n_vars} genes")
    print(adata.obs.groupby(["donor", "condition"], observed=True).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
