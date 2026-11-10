"""Build an RNA-only input from the Hao three-prime GEO deposit (GSE164378)."""

import argparse
import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.source
    barcodes = pd.read_csv(root / "GSM5008737_RNA_3P-barcodes.tsv.gz", header=None)[0]
    features = pd.read_csv(root / "GSM5008737_RNA_3P-features.tsv.gz", sep="\t", header=None)
    design = pd.read_csv(
        root / "GSE164378_sc.meta.data_3P.csv.gz",
        usecols=["Unnamed: 0", "lane", "donor", "time", "Batch"],
        index_col=0,
    )
    if not design.index.is_unique or not barcodes.is_unique or set(barcodes) != set(design.index):
        raise ValueError("GEO design and count barcodes do not match one-to-one")
    with gzip.open(root / "GSM5008737_RNA_3P-matrix.mtx.gz", "rb") as handle:
        matrix = mmread(handle)
    if matrix.dtype.kind not in "iu" or matrix.data.min() < 0 or matrix.data.max() > 2**31 - 1:
        raise ValueError("Expected nonnegative integer UMI counts")
    counts = matrix.T.astype(np.int32).tocsr()
    del matrix
    obs = design.loc[barcodes].copy()
    obs["sample"] = obs["donor"].astype(str) + "_day" + obs["time"].astype(str)
    var = pd.DataFrame({"gene_id": features[0].values}, index=features[1].astype(str))
    var.index.name = None
    obj = ad.AnnData(counts, obs=obs, var=var)
    obj.var_names_make_unique()
    obj.write_h5ad(args.output)
    print(f"{obj.n_obs:,} barcodes, {obj.n_vars:,} features, {counts.nnz:,} nonzero counts")


if __name__ == "__main__":
    main()
