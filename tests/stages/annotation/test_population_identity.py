"""Tests for generic population identity evidence outputs."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.backends.base import BaseBackend
from cellquorum.backends.registry import BackendRegistry
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.pipeline import build_pipeline_context
from cellquorum.stages.annotation.population_identity.stage import (
    PopulationIdentityStage,
    add_qc_columns,
)


def build_backend_registry() -> BackendRegistry:
    """Build deterministic backend registry for tests."""

    registry = BackendRegistry()
    registry.register(BaseBackend(name="python", kind="python"))
    return registry


def make_identity_adata(*, include_reference: bool) -> ad.AnnData:
    """Build a synthetic object with native clusters and optional atlas labels."""

    n_cells = 48
    obs = pd.DataFrame(
        {
            "leiden": ["0"] * 24 + ["1"] * 24,
            "patient_id": ["P1"] * 12 + ["P2"] * 12 + ["P1"] * 12 + ["P2"] * 12,
            "sample_id": ["S1"] * 12 + ["S2"] * 12 + ["S3"] * 12 + ["S4"] * 12,
            "condition": ["Normal"] * 24 + ["Lymphedema"] * 24,
            "cellquorum_qc_keep": [True] * n_cells,
            "predicted_doublet": [False] * n_cells,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    if include_reference:
        obs["ref_state"] = ["KC basal"] * 24 + ["KC spinous"] * 24
        obs["ref_state_consensus_frac"] = [0.92] * n_cells
        obs["ref_state_knn_entropy"] = [0.15] * n_cells

    adata = ad.AnnData(X=np.ones((n_cells, 5)), obs=obs)
    x = np.r_[
        np.random.default_rng(1).normal(0, 0.2, 24), np.random.default_rng(2).normal(2, 0.2, 24)
    ]
    y = np.r_[
        np.random.default_rng(3).normal(0, 0.2, 24), np.random.default_rng(4).normal(2, 0.2, 24)
    ]
    adata.obsm["X_umap"] = np.c_[x, y]
    return adata


def build_context(tmp_path: Path, adata: ad.AnnData, *, write_figures: bool = True):
    """Build a PipelineContext carrying a synthetic AnnData."""

    config = CellQuorumConfig(
        project={"name": "population_identity_test"},
        compute={"backend": "cpu", "prefer_gpu": False},
        r={"enabled": False},
        population_identity={
            "min_cells": 5,
            "min_samples": 2,
            "min_donors": 2,
            "write_figures": write_figures,
        },
    )
    context = build_pipeline_context(
        config,
        output_dir=tmp_path / "run",
        backend_registry=build_backend_registry(),
    )
    return context.with_adata(adata)


def test_population_identity_uses_reference_when_available(tmp_path: Path) -> None:
    """Atlas/reference labels should be used as identity evidence when present."""

    context = build_context(tmp_path, make_identity_adata(include_reference=True))
    result = PopulationIdentityStage().run(context)

    assert result.status == "success"
    assert result.metrics["candidate_key"] == "ref_state"
    assert result.metrics["candidate_source"] == "reference"
    assert result.metrics["n_populations"] == 2
    assert result.metrics["status_counts"]["atlas_supported_state"] == 2

    summary_path = (
        tmp_path / "run" / "results" / "population_identity" / "tables" / "population_summary.csv"
    )
    evidence_path = tmp_path / "run" / "results" / "population_identity" / "evidence.md"
    reference_plot = (
        tmp_path / "run" / "figures" / "population_identity" / "embedding_by_population.png"
    )
    assert summary_path.exists()
    assert evidence_path.exists()
    assert reference_plot.exists()

    summary = pd.read_csv(summary_path)
    assert set(summary["population_id"]) == {"KC basal", "KC spinous"}
    assert set(summary["evidence_status"]) == {"atlas_supported_state"}


def test_population_identity_falls_back_to_native_clusters_without_atlas(tmp_path: Path) -> None:
    """No-atlas datasets should still get native cluster candidate evidence."""

    context = build_context(
        tmp_path, make_identity_adata(include_reference=False), write_figures=False
    )
    result = PopulationIdentityStage().run(context)

    assert result.status == "success"
    assert result.metrics["candidate_key"] == "leiden"
    assert result.metrics["candidate_source"] == "cluster"
    assert result.metrics["status_counts"]["cluster_native_candidate"] == 2
    assert "No atlas/reference or annotation labels were available" in result.warnings[0]

    summary_path = (
        tmp_path / "run" / "results" / "population_identity" / "tables" / "population_summary.csv"
    )
    audit_path = tmp_path / "run" / "results" / "population_identity" / "audit.json"
    assert summary_path.exists()
    assert audit_path.exists()

    summary = pd.read_csv(summary_path)
    assert set(summary["population_id"].astype(str)) == {"0", "1"}
    assert set(summary["population_source"]) == {"cluster"}
    assert set(summary["evidence_status"]) == {"cluster_native_candidate"}


def test_population_identity_skips_when_no_identity_signal(tmp_path: Path) -> None:
    """Stage should skip cleanly instead of inventing populations from nothing."""

    adata = ad.AnnData(X=np.ones((8, 3)), obs=pd.DataFrame(index=[f"cell_{i}" for i in range(8)]))
    context = build_context(tmp_path, adata, write_figures=False)
    result = PopulationIdentityStage().run(context)

    assert result.status == "skipped"
    assert "no population identity column" in result.skip_reason


def test_population_identity_handles_empty_summary_without_crash(tmp_path: Path) -> None:
    """A candidate column but zero cells must skip, not KeyError on evidence_status.

    Regression: a 0-cell object (e.g. an upstream stage that emptied the working
    object) produces an empty population_summary with no 'evidence_status' column;
    the metrics block indexed that column and crashed the whole stage.
    """

    obs = pd.DataFrame(
        {"leiden": pd.Series([], dtype="object")},
        index=pd.Index([], dtype="object"),
    )
    adata = ad.AnnData(X=np.empty((0, 3), dtype=np.float32), obs=obs)
    context = build_context(tmp_path, adata, write_figures=False)
    result = PopulationIdentityStage().run(context)

    assert result.status == "skipped"
    assert result.metrics["n_populations"] == 0


def test_add_qc_columns_prefers_qc_state_final_when_cellquorum_qc_keep_absent() -> None:
    """A pipeline running qc_finalization (order=135) writes qc_state_final, not the
    older cellquorum_qc_keep boolean. Confirmed on the real lymphedema atlas run:
    it has qc_state_final (832 quarantine + 16,572 unresolved_borderline of 201,871
    cells) and NO cellquorum_qc_keep column at all. Falling through to the 0.0
    default made every population look QC-clean there regardless of real
    quarantine/borderline content -- exactly the gap already closed once for
    adjudication's technical_score (see adjudication/evidence.py).
    """

    group = pd.DataFrame(
        {"qc_state_final": ["core"] * 6 + ["quarantine"] * 2 + ["unresolved_borderline"] * 2}
    )
    row: dict = {}
    add_qc_columns(row, group)
    assert row["qc_fail_fraction"] == pytest.approx(0.4)


def test_add_qc_columns_falls_back_to_cellquorum_qc_keep_when_qc_state_final_absent() -> None:
    """Older configs (no qc_finalization) keep working via the original signal."""

    group = pd.DataFrame({"cellquorum_qc_keep": [True, True, False, False]})
    row: dict = {}
    add_qc_columns(row, group)
    assert row["qc_fail_fraction"] == pytest.approx(0.5)


def test_add_qc_columns_prefers_qc_state_final_over_cellquorum_qc_keep_when_both_present() -> None:
    """qc_state_final is the newer, richer verdict and should win when both exist."""

    group = pd.DataFrame(
        {
            "qc_state_final": ["quarantine"] * 3 + ["core"] * 7,
            "cellquorum_qc_keep": [True] * 10,  # would say 0% fail if used instead
        }
    )
    row: dict = {}
    add_qc_columns(row, group)
    assert row["qc_fail_fraction"] == pytest.approx(0.3)
