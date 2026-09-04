"""QC figures must render from the pre-filter population, and the publication
suite must actually run on a real cohort schema.

Two bugs these lock down, both found on the 2026-09-01 lec_mechanotransduction
run and both silent — the stage reported success either way:

1. Figures were drawn from the stage's OUTPUT object. Under ``mode="filter"``
   that object has already lost the failing cells, so the keep/fail barplot read
   "3294 Pass / 0 Fail / Pass Rate 100.0%" on a run that dropped 503 of 3797
   cells, and the pass/fail scatter had a one-entry legend. Every figure whose
   subject is the filter showed the opposite of what happened.

2. ``write_publication_qc_figures`` defaults ``patient_key="patient_id"``. No
   CellQuorum cohort schema uses that name — cohorts declare ``donor_key`` — so
   the writer raised, the caller swallowed the exception into a warning string
   in qc_summary.json, and 14 publication panels silently never rendered on any
   run. The fallback diagnostic plots shipped in their place.
"""

from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.qc._annotate import (
    annotate_adata_with_qc_metrics,
    build_qc_figure_adata,
    build_qc_output_adata,
)
from cellquorum.stages.qc._report import (
    resolve_publication_qc_keys,
)
from cellquorum.stages.qc.artifacts import write_qc_artifacts
from cellquorum.stages.qc.config import QCConfig
from cellquorum.stages.qc.floors import apply_floors
from cellquorum.stages.qc.metrics import calculate_qc_metrics


def _cohort_adata(n_cells=90, n_genes=40):
    """A paired two-condition cohort where a known subset must fail QC.

    The last 15 cells are given almost no counts, so ``min_genes_per_cell``
    removes them and pre- vs post-filter figure sources are distinguishable.
    """
    rng = np.random.default_rng(0)
    x = rng.poisson(5.0, size=(n_cells, n_genes)).astype(np.float32)
    # Drive the final 15 cells below any sane detected-gene floor.
    x[-15:, :] = 0.0
    x[-15:, 0] = 1.0

    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    a = ad.AnnData(X=x, var=var)
    a.obs_names = [f"cell_{i}" for i in range(n_cells)]
    a.layers["counts"] = x.copy()
    # Paired design: 3 donors x 2 conditions, one sample each.
    donors = np.repeat(["D1", "D2", "D3"], n_cells // 3)
    conditions = np.tile(["Normal", "Lymphedema"], n_cells // 2)
    a.obs["donor_id"] = pd.Categorical(donors)
    a.obs["condition"] = pd.Categorical(conditions)
    a.obs["sample_id"] = pd.Categorical(
        [f"{d}_{c}" for d, c in zip(donors, conditions, strict=True)]
    )
    return a


def _qc_products(adata):
    """Run the QC computation chain the stage runs, without stage plumbing.

    Floors are stated rather than inherited: the shipped default is 100 detected genes, which a
    40-gene fixture cannot clear, so every cell would fall below it. A floor of 10 straddles the
    fixture instead — the zeroed cells fail, the rest pass.

    There is no ``mode`` any more. Floors always filter, because a barcode below the detection
    limit is not a cell; every judgement belongs to graded adjudication, which never deletes.
    """
    config = QCConfig(basic={"min_genes_per_cell": 10, "min_cells_per_gene": 1})
    metrics = calculate_qc_metrics(adata, config)
    floors = apply_floors(
        adata.X,
        adata.obs_names,
        adata.var_names,
        min_genes_per_cell=10,
        min_cells_per_gene=1,
    )
    output = build_qc_output_adata(adata=adata, floors=floors)
    annotate_adata_with_qc_metrics(adata=output, metrics_result=metrics)
    return metrics, floors, output


def test_figure_adata_keeps_the_cells_qc_removed():
    """The figure source must carry every input cell, flagged, not the survivors."""
    parent = _cohort_adata()
    metrics, floors, output = _qc_products(parent)

    # Precondition: filtering actually removed cells, or this proves nothing.
    assert output.n_obs < parent.n_obs, "fixture must produce QC failures"

    figure_adata = build_qc_figure_adata(
        adata=parent,
        output_adata=output,
        metrics_result=metrics,
        floors=floors,
    )

    # Every input cell is present, so a keep/fail panel has both categories.
    assert figure_adata.n_obs == parent.n_obs
    keep = figure_adata.obs["cellquorum_qc_keep"].astype(bool)
    assert int(keep.sum()) == output.n_obs
    assert int((~keep).sum()) == parent.n_obs - output.n_obs

    # The metrics the figures plot are present for the REMOVED cells too —
    # otherwise a pre-filter histogram would still only show survivors.
    failed_metrics = figure_adata.obs.loc[~keep, "n_genes_by_counts"]
    assert failed_metrics.notna().all()


def test_figure_adata_carries_no_expression_matrix():
    """Figures read obs/var/obsm only; a second copy of X is pure cost at QC."""
    parent = _cohort_adata()
    metrics, floors, output = _qc_products(parent)

    figure_adata = build_qc_figure_adata(
        adata=parent,
        output_adata=output,
        metrics_result=metrics,
        floors=floors,
    )

    assert figure_adata.X is None
    # var still travels: the gene-detection histogram reads n_cells_by_counts.
    assert figure_adata.n_vars == parent.n_vars


def test_figure_adata_marks_post_filter_only_columns_as_missing():
    """Doublet scores are computed after filtering; removed cells get NaN.

    NaN is the accurate value here — those cells were never scored. Carrying a
    fabricated 0.0 would put them at the confident-singlet end of the ECDF.
    """
    parent = _cohort_adata()
    metrics, floors, output = _qc_products(parent)
    # Simulate the doublet detector, which the stage runs on the filtered object.
    output.obs["doublet_score"] = np.linspace(0.0, 0.4, output.n_obs)

    figure_adata = build_qc_figure_adata(
        adata=parent,
        output_adata=output,
        metrics_result=metrics,
        floors=floors,
    )

    keep = figure_adata.obs["cellquorum_qc_keep"].astype(bool)
    scores = figure_adata.obs["doublet_score"]
    assert scores.loc[keep].notna().all()
    assert scores.loc[~keep].isna().all()


def test_cell_labels_table_covers_the_cells_qc_removed(tmp_path):
    """The run directory must be able to re-render by-cell-type QC on its own.

    Under ``mode="filter"`` qc.h5ad holds survivors only, so a removed cell's
    cell type is unrecoverable from the run directory — a re-rendered attrition
    figure then reports every cell type as losing nothing, which is the exact
    opposite of the panel's subject. cell_labels.csv is the pre-filter labels.
    """
    parent = _cohort_adata()
    # Two resolutions of label, and the removed cells (the last 15) are labelled
    # too — they are the rows the table exists for.
    lineage = np.where(np.arange(parent.n_obs) % 3 == 0, "LEC", "Fibroblast")
    parent.obs["cell_type"] = pd.Categorical(lineage)
    parent.obs["cell_type_granular"] = pd.Categorical(
        [f"{name} {i % 2}" for i, name in enumerate(lineage)]
    )
    metrics, floors, output = _qc_products(parent)
    assert output.n_obs < parent.n_obs, "fixture must produce QC failures"

    figure_adata = build_qc_figure_adata(
        adata=parent,
        output_adata=output,
        metrics_result=metrics,
        floors=floors,
    )
    config = QCConfig(
        mode="filter",
        threshold_strategy="fixed",
        basic={"min_genes_per_cell": 10, "min_cells_per_gene": 1, "max_mito_percent": None},
        outputs={"write_figures": False, "write_h5ad": False, "publication_tables": False},
    )

    write_qc_artifacts(
        output_dir=tmp_path,
        metrics_result=metrics,
        floors=floors,
        config=config,
        adata=output,
        figure_adata=figure_adata,
        publication_keys={
            "sample_key": "sample_id",
            "patient_key": "donor_id",
            "condition_key": "condition",
        },
    )

    labels = pd.read_csv(tmp_path / "cell_labels.csv", index_col=0)
    # Pre-filter index, in the same order as the decision table it is joined to.
    assert labels.index.tolist() == floors.cell_table().index.tolist()
    assert {"sample", "donor", "condition", "cell_type", "cell_type_granular"} <= set(
        labels.columns
    )

    removed = ~floors.cell_table()["keep"].to_numpy(dtype=bool)
    assert removed.sum() == parent.n_obs - output.n_obs
    assert labels.loc[removed].notna().all().all()
    # And the labels are the input's, not something rebuilt from survivors.
    pd.testing.assert_series_equal(
        labels["cell_type_granular"],
        parent.obs["cell_type_granular"].astype(str),
        check_names=False,
        check_index=False,
    )


def test_publication_keys_resolve_donor_key_not_patient_id():
    """A cohort declaring donor_key must not fall through to `patient_id`.

    This is the whole bug: the writer's default names a column that does not
    exist, so it raised and 14 panels were skipped behind a warning.
    """
    adata = _cohort_adata()
    cohort = SimpleNamespace(
        donor_key="donor_id",
        sample_key="sample_id",
        condition_key="condition",
        condition_levels=["Normal", "Lymphedema"],
    )
    design = SimpleNamespace(
        donor_col="donor_id",
        condition_col="condition",
        case="Lymphedema",
        control="Normal",
    )

    keys = resolve_publication_qc_keys(adata=adata, cohort=cohort, design=design)

    assert keys["patient_key"] == "donor_id"
    assert keys["sample_key"] == "sample_id"
    assert keys["condition_key"] == "condition"
    assert keys["normal_label"] == "Normal"
    assert keys["disease_label"] == "Lymphedema"


def test_publication_keys_omit_columns_the_object_lacks():
    """Unresolvable keys are left out so the writer keeps its own defaults."""
    adata = _cohort_adata()
    del adata.obs["sample_id"]
    cohort = SimpleNamespace(
        donor_key="donor_id", sample_key="sample_id", condition_key="condition"
    )

    keys = resolve_publication_qc_keys(adata=adata, cohort=cohort, design=None)

    assert keys["patient_key"] == "donor_id"
    assert "sample_key" not in keys


def test_publication_keys_fall_back_to_condition_levels_for_labels():
    """Without an explicit design, ordered cohort levels supply the labels."""
    adata = _cohort_adata()
    cohort = SimpleNamespace(
        donor_key="donor_id",
        sample_key="sample_id",
        condition_key="condition",
        condition_levels=["Normal", "Lymphedema"],
    )

    keys = resolve_publication_qc_keys(adata=adata, cohort=cohort, design=None)

    assert keys["normal_label"] == "Normal"
    assert keys["disease_label"] == "Lymphedema"
