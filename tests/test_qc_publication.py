"""Tests for publication-style QC figures."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.qc.publication import write_publication_qc_figures


def test_write_publication_qc_figures(tmp_path):
    """Publication QC writer should emit compact PNG/PDF panel sets."""

    obs = pd.DataFrame(
        {
            "patient_id": ["P1"] * 20 + ["P2"] * 20,
            "condition": (["Normal"] * 10 + ["Lymphedema"] * 10) * 2,
            "pct_counts_mito": np.linspace(1, 12, 40),
            "pct_counts_ribo": np.linspace(10, 40, 40),
            "pct_counts_hemoglobin": np.linspace(0, 2, 40),
            "log1p_total_counts": np.linspace(7, 11, 40),
            "log1p_n_genes_by_counts": np.linspace(6.5, 9, 40),
            "pct_counts_in_top_20_genes": np.linspace(10, 35, 40),
            "doublet_score": np.linspace(0.01, 0.30, 40),
            "n_genes_by_counts": np.linspace(500, 4000, 40),
            "total_counts": np.linspace(1000, 30000, 40),
            "cellquorum_qc_keep": [True, False] * 20,
            "cellquorum_qc_mad_log1p_total_counts": [False] * 38 + [True, True],
        },
        index=[f"cell_{i}" for i in range(40)],
    )
    adata = ad.AnnData(X=np.ones((40, 3)), obs=obs)
    thresholds = pd.DataFrame(
        {
            "axis": ["cell", "cell", "cell", "cell"],
            "metric": [
                "log1p_total_counts",
                "log1p_n_genes_by_counts",
                "pct_counts_in_top_20_genes",
                "pct_counts_mito",
            ],
            "rule_name": [
                "mad_log1p_total_counts",
                "mad_log1p_n_genes_by_counts",
                "mad_pct_counts_in_top_20_genes",
                "mad_mito_pct_counts_mito",
            ],
            "lower": [7.1, 6.6, 10.5, 1.5],
            "upper": [10.8, 8.9, 32.0, 10.0],
            "source": ["mad", "mad", "mad", "mad_mito"],
        }
    )

    paths = write_publication_qc_figures(adata, tmp_path, thresholds=thresholds, dpi=60)

    names = {path.name for path in paths}
    assert "qc_panel_A_mitochondrial_content.png" in names
    assert "qc_panel_A_mitochondrial_content.pdf" in names
    assert "qc_panel_I_mad_thresholds.png" in names
    assert "qc_panel_I_mad_thresholds.svg" in names
    assert "qc_panel_J_umi_detected_genes_normal.png" in names
    assert "qc_panel_K_umi_detected_genes_le.png" in names
    assert "qc_panel_L_by_condition_publication.png" in names
    assert "qc_panel_M_cells_per_sample.png" in names
    assert "supp_figure_qc_visual_qa_sheet.png" in names
    assert all(path.exists() for path in paths)
