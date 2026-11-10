"""Prepare an independent SS2 query and inDrop reference from the scArches pancreas atlas."""

import argparse
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.atlas.open("rb") as handle:
        if (
            hashlib.file_digest(handle, "sha256").hexdigest()
            != "5b9eb06703cadfe0da712b338d52362d435b43c837e65d4ae0ea081a0b02cd83"
        ):
            raise ValueError("Unexpected checksum for the scArches pancreas reference input.")
    data = ad.read_h5ad(args.atlas)
    args.output.mkdir(parents=True, exist_ok=True)
    studies = []
    for name, study in [("reference", "Pancreas inDrop"), ("query", "Pancreas SS2")]:
        cohort = data[data.obs.study.eq(study)]
        data_counts = cohort.raw.to_adata()
        data_counts.X = sparse.csr_matrix(data_counts.X)
        values = data_counts.X.data
        if (
            not len(values)
            or not (np.isfinite(values) & (values >= 0) & (values == np.floor(values))).all()
        ):
            raise ValueError(f"{study} needs verified integer counts.")
        clean = ad.AnnData(
            data_counts.X,
            obs=cohort.obs[["study", "batch", "cell_type"]].copy(),
            var=data_counts.var.copy(),
        )
        studies.append(clean.copy())
        if name == "query":
            clean.obs[["cell_type"]].to_csv(
                args.output / "held_out_labels.csv", index_label="barcode"
            )
            del clean.obs["cell_type"]
        clean.write_h5ad(args.output / f"{name}.h5ad", compression="gzip")

    ad.concat(studies, merge="same").write_h5ad(
        args.output / "integration.h5ad", compression="gzip"
    )


if __name__ == "__main__":
    main()
