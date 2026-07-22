"""Export the shareable LE global deliverable + patient summary + README.

Usage:
    python scripts/export_le_global.py <final_h5ad> <run_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad


def _score_cell_cycle(a: ad.AnnData) -> None:
    """
    Add S_score / G2M_score / phase to obs, scored on the normalized layer.

    Uses CellQuorum's own scorer and Tirosh gene lists so the result matches the
    engine. No-op (with a printed note) when the normalized layer is absent.

    Args:
        a: The final annotated AnnData (expects a cellquorum_normalized layer).
    """

    layer = "cellquorum_normalized"
    if layer not in a.layers:
        print(f"cell-cycle: skipped ('{layer}' layer absent)")
        return
    try:
        from cellquorum.qc.cell_cycle import (
            TIROSH_G2M_GENES,
            TIROSH_S_GENES,
            score_cell_cycle,
        )
        from cellquorum.qc.config import QCCellCycleConfig

        cc = QCCellCycleConfig(
            enabled=True,
            score_layer=layer,
            s_genes=TIROSH_S_GENES,
            g2m_genes=TIROSH_G2M_GENES,
        )
        score_cell_cycle(a, cc)
        print("cell-cycle: scored S_score / G2M_score / phase on", layer)
    except Exception as e:  # diagnostic only — never fail export on it
        print(f"cell-cycle: skipped (scoring error: {e!r})")


def main(final_h5ad: str, run_dir: str) -> int:
    run = Path(run_dir)
    objects = run / "objects"
    results = run / "results"
    objects.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(final_h5ad)

    # Cell-cycle scoring, done HERE (post-normalization) rather than in QC:
    # QC runs before normalization exists, so the accurate place to score the
    # cell cycle is on the real log-normalized layer. Reuse the engine's own
    # scorer + Tirosh gene lists so the phase call matches what CellQuorum would
    # produce. Diagnostic only (not used for annotation); skipped if the layer
    # is absent so export never fails on it.
    _score_cell_cycle(a)

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
        "- `adata.obs['phase']` (+ S_score/G2M_score): cell-cycle phase, scored on the",
        "  normalized layer. Diagnostic only — not used for the cell-type calls.",
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
