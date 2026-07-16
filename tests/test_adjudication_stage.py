"""Tests for the adjudication pipeline stage."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.adjudication.config import AdjudicationConfig
from cellquorum.adjudication.evidence import build_cluster_evidence_table
from cellquorum.adjudication.stage import AdjudicationStage
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.executor import build_default_stage_registry
from cellquorum.core.pipeline import build_pipeline_context


def _adata_with_metadata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "leiden": ["0", "0", "0", "1", "1", "1"],
            "patient_id": ["d1", "d2", "d3", "d1", "d1", "d2"],
            "condition": ["case", "case", "control", "case", "case", "control"],
            "cell_type_conf": [0.8, 0.7, 0.9, 0.4, 0.5, 0.6],
            "predicted_doublet": [False, False, False, True, True, False],
        },
        index=[f"cell_{i}" for i in range(6)],
    )
    return ad.AnnData(X=np.ones((6, 3)), obs=obs)


def test_build_cluster_evidence_table_from_obs():
    adata = _adata_with_metadata()

    evidence = build_cluster_evidence_table(
        adata,
        config=AdjudicationConfig(),
        donor_key="patient_id",
        condition_key="condition",
    )

    assert {row.cluster_id for row in evidence} == {"0", "1"}
    cluster_0 = next(row for row in evidence if row.cluster_id == "0")
    assert cluster_0.n_cells == 3
    assert cluster_0.n_donors == 3
    assert cluster_0.marker_support == pytest.approx(0.8)


def test_adjudication_stage_writes_artifacts_and_uns_payload(tmp_path):
    config = CellQuorumConfig(
        compute={"prefer_gpu": False},
        r={"enabled": False},
        adjudication={"cluster_key": "leiden"},
    )
    context = build_pipeline_context(config, output_dir=tmp_path / "run").with_adata(
        _adata_with_metadata()
    )

    result = AdjudicationStage().run(context)

    assert result.status == "success"
    assert result.metrics["n_clusters"] == 2
    assert "adjudication" in result.adata.uns["cellquorum"]
    artifact_names = {artifact.name for artifact in result.artifacts}
    assert artifact_names == {
        "adjudication_results",
        "adjudication_evidence",
        "adjudication_results_json",
        "adjudication_summary",
    }
    for artifact in result.artifacts:
        assert artifact.path.exists()


def test_adjudication_stage_skips_when_required_metadata_missing(tmp_path):
    config = CellQuorumConfig(
        compute={"prefer_gpu": False},
        r={"enabled": False},
        adjudication={"cluster_key": "leiden"},
    )
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame({"leiden": ["0", "1"]}, index=["cell_1", "cell_2"]),
    )
    context = build_pipeline_context(config, output_dir=tmp_path / "run").with_adata(adata)

    result = AdjudicationStage().run(context)

    assert result.status == "skipped"
    assert "patient_id" in result.skip_reason
    assert "condition" in result.skip_reason


def test_default_stage_registry_includes_adjudication():
    registry = build_default_stage_registry()

    assert registry.get("adjudication") is not None
