"""One QC run can hold whole cells and nuclei: the nuclear axis applies per library.

MALAT1 and the intronic fraction are evidence of damage in a whole cell but the expected signal in a
nucleus. The run-level switch made a mixed run give the axis up for everyone, so these tests pin the
per-library behaviour: the axis is not applicable for the named nuclei, it still scores the whole
cells, and a nucleus is not penalised in evidence coverage for evidence it could never carry.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.qc.config import QCGradedConfig
from cellquorum.stages.qc.evidence import (
    EvidenceAvailability,
    EvidenceFamily,
    build_evidence_table,
)

N = 40


def _mixed_run() -> tuple[ad.AnnData, pd.DataFrame, pd.Series]:
    """Two libraries: whole cells (low MALAT1) and nuclei (high MALAT1), same genes."""
    rng = np.random.default_rng(0)
    genes = ["MALAT1", "ACTB", "GAPDH", "FOS", "MT-CO1"]
    cell_rows = rng.poisson([2, 30, 25, 1, 4], size=(N, 5))
    nuclei_rows = rng.poisson([60, 20, 15, 1, 0.2], size=(N, 5))
    counts = np.vstack([cell_rows, nuclei_rows]).astype(np.float32)
    obs = pd.DataFrame(
        {
            "sample": ["cells_lib"] * N + ["nuclei_lib"] * N,
            "assay": ["scRNA"] * N + ["snRNA"] * N,
        },
        index=[f"c{i}" for i in range(2 * N)],
    )
    adata = ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=genes))
    total = counts.sum(axis=1)
    metrics = pd.DataFrame(
        {
            "total_counts": total,
            "n_genes_by_counts": (counts > 0).sum(axis=1),
            "pct_counts_mito": 100.0 * counts[:, 4] / np.maximum(total, 1),
        },
        index=adata.obs_names,
    )
    mask = pd.Series(adata.obs["assay"] != "snRNA", index=adata.obs_names)
    return adata, metrics, mask


def _axis(evidence, name):
    return next(axis for axis in evidence.axes if axis.name == name)


def test_nuclei_are_marked_not_applicable_and_cells_are_still_scored() -> None:
    adata, metrics, mask = _mixed_run()
    evidence = build_evidence_table(adata, metrics, group_key="sample", nuclear_axis_mask=mask)
    axis = _axis(evidence, "malat1_fraction")
    nuclei = adata.obs["assay"] == "snRNA"
    assert (axis.availability[nuclei] == str(EvidenceAvailability.NOT_APPLICABLE)).all()
    assert axis.severity[nuclei].isna().all()
    assert (axis.availability[~nuclei] == str(EvidenceAvailability.AVAILABLE_VALID)).all()
    assert axis.severity[~nuclei].notna().all()


def test_nuclear_axes_do_not_apply_to_nuclei_and_coverage_follows_the_frozen_rule() -> None:
    """Not applicable lowers coverage, as the design prescribes; whole cells are unaffected."""
    adata, metrics, mask = _mixed_run()
    mixed = build_evidence_table(adata, metrics, group_key="sample", nuclear_axis_mask=mask)
    plain = build_evidence_table(adata, metrics, group_key="sample")
    nuclei = adata.obs["assay"] == "snRNA"
    n_families = len(mixed.families_present())
    # Whole cells keep full coverage, and nuclei lose exactly the one family that cannot apply.
    assert (mixed.evidence_coverage()[~nuclei] == plain.evidence_coverage()[~nuclei]).all()
    np.testing.assert_allclose(
        mixed.evidence_coverage()[nuclei].to_numpy(), (n_families - 1) / n_families
    )


def test_high_nuclear_signal_in_nuclei_cannot_raise_damage_severity() -> None:
    adata, metrics, mask = _mixed_run()
    with_axis = build_evidence_table(adata, metrics, group_key="sample")
    per_library = build_evidence_table(adata, metrics, group_key="sample", nuclear_axis_mask=mask)
    nuclei = adata.obs["assay"] == "snRNA"
    nuclear = str(EvidenceFamily.NUCLEAR_INTEGRITY)
    assert per_library.family_severity().loc[nuclei, nuclear].isna().all()
    # The run-level axis scored these nuclei as if MALAT1 meant damage; per library it does not.
    assert with_axis.family_severity().loc[nuclei, nuclear].notna().any()


def test_coverage_is_unchanged_when_no_axis_is_marked_not_applicable() -> None:
    adata, metrics, _ = _mixed_run()
    plain = build_evidence_table(adata, metrics, group_key="sample")
    all_cells = pd.Series(True, index=adata.obs_names)
    masked = build_evidence_table(adata, metrics, group_key="sample", nuclear_axis_mask=all_cells)
    np.testing.assert_allclose(
        plain.evidence_coverage().to_numpy(), masked.evidence_coverage().to_numpy()
    )


def test_nuclei_key_and_values_must_be_set_together() -> None:
    with pytest.raises(ValueError, match="set together"):
        QCGradedConfig(nuclei_key="assay")
    with pytest.raises(ValueError, match="set together"):
        QCGradedConfig(nuclei_values=["snRNA"])
    config = QCGradedConfig(nuclei_key="assay", nuclei_values=["snRNA"])
    assert config.nuclei_values == ["snRNA"]


def test_naming_nuclei_while_switching_the_axis_off_is_one_decision_spelled_twice() -> None:
    with pytest.raises(ValueError, match="nuclear_axis_applicable"):
        QCGradedConfig(nuclear_axis_applicable=False, nuclei_key="assay", nuclei_values=["snRNA"])
