"""Export the shareable LE global deliverable + patient summary + README.

Usage:
    python scripts/export_le_global.py <final_h5ad> <run_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad


def main(final_h5ad: str, run_dir: str) -> int:
    run = Path(run_dir)
    objects = run / "objects"
    results = run / "results"
    objects.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(final_h5ad)

    # Patient x condition cell counts.
    summary = (
        a.obs.groupby(["donor_id", "condition"], observed=True).size().reset_index(name="n_cells")
    )
    summary_path = results / "patients_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Write the deliverable under a stable name.
    out = objects / "le_global_annotated.h5ad"
    a.write_h5ad(out)

    # Plain-language README.
    tiers = (
        a.obs["annotation_confidence"].value_counts().to_dict()
        if "annotation_confidence" in a.obs.columns
        else {}
    )
    types = a.obs["cell_type"].value_counts().to_dict() if "cell_type" in a.obs.columns else {}
    readme = run / "README_for_collaborator.md"
    lines = [
        "# Lymphedema global single-cell atlas",
        "",
        f"- Cells: {a.n_obs:,}  Genes: {a.n_vars:,}",
        f"- Patients: {a.obs['donor_id'].nunique()} (paired Normal/Lymphedema)",
        "",
        "## How to use",
        "- `adata.obs['cell_type']`: final cell-type label (consensus of marker genes,",
        "  CellTypist, and scANVI/scArches transfer from the normal skin atlas).",
        "- `adata.obs['annotation_confidence']`: high / medium / low (agreement across the",
        "  three methods). `adata.obs['needs_review']`: True where methods disagreed.",
        "- `adata.obs['cell_type_granular']`: finer state for high-confidence cells.",
        "- `adata.obs['donor_id']` / `condition` / `sample_id`: per-cell patient identity.",
        "- Embeddings in `adata.obsm`: `X_scvi`, `X_pca_harmony`, `X_umap`.",
        "- Layers: `counts` (raw), `cellquorum_normalized` (log-normalized).",
        "",
        "## Confidence tiers",
        *[f"- {k}: {v:,}" for k, v in tiers.items()],
        "",
        "## Cell types",
        *[f"- {k}: {v:,}" for k, v in types.items()],
        "",
        "## Notes",
        "- Threshold QC flagged (did not delete) marginal cells; only consensus doublets",
        "  (flagged by BOTH Scrublet and scDblFinder) were removed.",
        "- To exclude a patient: filter `adata[adata.obs.donor_id != 'P<i>']`, or set that",
        "  patient's `include=FALSE` in configs/manifests/le_global_cohort.csv and re-run.",
    ]
    readme.write_text("\n".join(lines) + "\n")

    print(f"wrote {out}")
    print(f"wrote {summary_path}")
    print(f"wrote {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
