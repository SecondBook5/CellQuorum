"""Physically removed vs inference-excluded: two tables, mutually exclusive categories."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.qc.eligibility import build_eligibility_masks
from cellquorum.stages.qc.exclusion_audit import (
    BORDERLINE_NONMULTIPLET,
    MULTIPLET_PLUS_DAMAGE,
    PROBABLE_MULTIPLET_ONLY,
    QUARANTINE_CAPTURE,
    QUARANTINE_MULTI_FAMILY,
    exclusion_category,
    inference_excluded_table,
    physically_removed_table,
)


def _obs() -> pd.DataFrame:
    state = ["core", "core", "borderline", "borderline", "quarantine", "quarantine", "core"]
    multiplet = [False, True, False, True, False, False, False]
    reason = [
        "no_concern", "probable_multiplet", "single_family_concern", "single_family_concern",
        "uninformative_barcode", "concordant_severe_damage", "no_concern",
    ]  # fmt: skip
    obs = pd.DataFrame(
        {"qc_state_initial": state, "qc_probable_multiplet": multiplet, "qc_state_reason": reason},
        index=[f"c{i}" for i in range(7)],
    )
    masks = build_eligibility_masks(
        obs["qc_state_initial"], probable_multiplet=obs["qc_probable_multiplet"]
    )
    return obs.join(masks.to_obs_frame())


def test_every_excluded_cell_gets_exactly_one_category():
    cat = exclusion_category(_obs())
    assert cat.tolist() == [
        "eligible",
        PROBABLE_MULTIPLET_ONLY,
        BORDERLINE_NONMULTIPLET,
        MULTIPLET_PLUS_DAMAGE,
        QUARANTINE_CAPTURE,
        QUARANTINE_MULTI_FAMILY,
        "eligible",
    ]


def test_inference_excluded_table_lists_retained_cells_and_counts_denied_permissions():
    obs = _obs()
    adata = ad.AnnData(X=np.zeros((7, 2), dtype=np.float32), obs=obs)
    table = inference_excluded_table(adata)
    assert list(table.index) == ["c1", "c2", "c3", "c4", "c5"]  # core singlets are not listed
    assert (table["n_permissions_denied"] > 0).all()
    assert table.loc["c1", "qc_inference_de"] is np.False_ or not table.loc["c1", "qc_inference_de"]
    assert set(table["exclusion_category"]) == set(exclusion_category(obs)) - {"eligible"}


def test_inference_excluded_table_is_empty_when_qc_wrote_no_verdict():
    adata = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32), obs=pd.DataFrame(index=["a", "b"]))
    assert inference_excluded_table(adata).empty


def test_physically_removed_table_keeps_only_barcodes_absent_from_the_object():
    accounting = pd.DataFrame(
        {
            "removal_stage": ["floor", "probable_multiplet"],
            "removal_reason": ["fewer_than_100_genes", "probable_multiplet"],
        },
        index=["gone", "kept"],
    )
    adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32), obs=pd.DataFrame(index=["kept"]))
    removed = physically_removed_table(accounting, adata)
    assert list(removed.index) == ["gone"]
