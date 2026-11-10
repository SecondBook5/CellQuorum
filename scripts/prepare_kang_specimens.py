"""Add a specimen ID for each donor-condition combination in the paired Kang study."""

import argparse
from pathlib import Path

import anndata as ad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotated", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.annotated.resolve():
        parser.error("Choose a new output path to preserve the study input.")
    study = ad.read_h5ad(args.annotated)
    meta = study.obs[["replicate", "label"]]
    if meta.isna().any().any() or set(meta.label) != {"ctrl", "stim"}:
        parser.error("Expected donor identities and the published ctrl/stim labels.")
    design = meta.drop_duplicates()
    if len(design) != 16 or design.replicate.nunique() != 8:
        parser.error("Expected eight donors with both specimens.")
    study.obs["specimen"] = meta.replicate.astype(str) + "_" + meta.label.astype(str)
    study.write_h5ad(args.output, compression="lzf")


if __name__ == "__main__":
    main()
