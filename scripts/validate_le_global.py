"""Validate the LE global annotated object against the deliverable contract.

Usage:
    python scripts/validate_le_global.py <final_h5ad>
"""

from __future__ import annotations

import sys

import anndata as ad


def main(path: str) -> int:
    a = ad.read_h5ad(path)
    problems: list[str] = []

    # Patient traceability: non-null for every cell.
    for col in ("sample_id", "donor_id", "condition"):
        if col not in a.obs.columns:
            problems.append(f"missing obs column: {col}")
        elif a.obs[col].isna().any():
            problems.append(f"{col} has {int(a.obs[col].isna().sum())} null value(s)")

    # Final annotation columns present.
    for col in ("cell_type", "annotation_confidence", "needs_review"):
        if col not in a.obs.columns:
            problems.append(f"missing annotation column: {col}")

    # Both integration embeddings present.
    for rep in ("X_pca_harmony", "X_scvi"):
        if rep not in a.obsm:
            problems.append(f"missing embedding: {rep}")

    # Layers.
    for layer in ("counts", "cellquorum_normalized"):
        if layer not in a.layers:
            problems.append(f"missing layer: {layer}")

    # cell_type has no NaN (every cell annotated).
    if "cell_type" in a.obs.columns and a.obs["cell_type"].isna().any():
        problems.append(f"cell_type has {int(a.obs['cell_type'].isna().sum())} unannotated cell(s)")

    print(f"object: {a.shape[0]} cells x {a.shape[1]} genes")
    if "annotation_confidence" in a.obs.columns:
        print("confidence tiers:", a.obs["annotation_confidence"].value_counts().to_dict())
    if "cell_type" in a.obs.columns:
        print("cell types:", a.obs["cell_type"].value_counts().to_dict())
    if "predicted_doublet" in a.obs.columns:
        print("remaining predicted_doublet:", int(a.obs["predicted_doublet"].sum()))
    if "donor_id" in a.obs.columns:
        print("cells per patient:", a.obs.groupby("donor_id", observed=True).size().to_dict())

    if problems:
        print("\nVALIDATION FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nVALIDATION PASSED: object meets the deliverable contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
