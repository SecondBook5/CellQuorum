"""Tests for the differential-state stage (order=215)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.pipeline import build_pipeline_context
from cellquorum.stages.comparative.differential_state.stage import DifferentialStateStage


def _adata(n_per_donor: int = 25) -> ad.AnnData:
    """8 donors, paired Normal/LE, state_up genuinely shifted, state_flat untouched."""
    rng = np.random.default_rng(0)
    donors = [f"d{i}" for i in range(8)]
    rows_condition, rows_donor, rows_up, rows_flat = [], [], [], []
    for donor in donors:
        for condition in ("Normal", "LE"):
            base = rng.normal(0, 1, n_per_donor)
            shift = 3.0 if condition == "LE" else 0.0
            rows_up.append(base + shift)
            rows_flat.append(rng.normal(0, 1, n_per_donor))
            rows_condition.extend([condition] * n_per_donor)
            rows_donor.extend([donor] * n_per_donor)
    obs = pd.DataFrame(
        {
            "condition": rows_condition,
            "donor_id": rows_donor,
            "state_up": np.concatenate(rows_up),
            "state_flat": np.concatenate(rows_flat),
            "state_label": ["x"] * len(rows_condition),  # non-numeric, must be excluded
        }
    )
    n = len(obs)
    return ad.AnnData(X=np.ones((n, 2)), obs=obs)


def _context(tmp_path, adata, **cfg):
    config = CellQuorumConfig(
        compute={"prefer_gpu": False},
        r={"enabled": False},
        design={
            "donor_col": "donor_id",
            "condition_col": "condition",
            "case": "LE",
            "control": "Normal",
        },
        differential_state=cfg,
    )
    return build_pipeline_context(config, output_dir=tmp_path / "run").with_adata(adata)


def test_detects_the_genuinely_shifted_program_as_significant(tmp_path):
    context = _context(tmp_path, _adata())
    result = DifferentialStateStage().run(context)
    assert result.status == "success"
    csv_path = next(a.path for a in result.artifacts if a.name == "differential_state_results")
    table = pd.read_csv(csv_path).set_index("program")
    assert table.loc["state_up", "significant"]
    assert table.loc["state_up", "effect"] > 0
    assert not table.loc["state_flat", "significant"]


def test_skips_when_case_control_not_configured(tmp_path):
    config = CellQuorumConfig(
        compute={"prefer_gpu": False},
        r={"enabled": False},
    )
    context = build_pipeline_context(config, output_dir=tmp_path / "run").with_adata(_adata())
    result = DifferentialStateStage().run(context)
    assert result.status == "skipped"


def test_skips_when_no_score_columns_resolved(tmp_path):
    adata = _adata()
    del adata.obs["state_up"]
    del adata.obs["state_flat"]
    context = _context(tmp_path, adata)
    result = DifferentialStateStage().run(context)
    assert result.status == "skipped"


def test_explicit_obs_columns_overrides_auto_detection(tmp_path):
    context = _context(tmp_path, _adata(), obs_columns=["state_flat"])
    result = DifferentialStateStage().run(context)
    csv_path = next(a.path for a in result.artifacts if a.name == "differential_state_results")
    table = pd.read_csv(csv_path)
    assert list(table["program"]) == ["state_flat"]


def test_min_cells_per_donor_excludes_a_thin_donor_without_crashing(tmp_path):
    adata = _adata()
    # Thin one donor's LE arm below the default floor (20).
    mask = (adata.obs["donor_id"] == "d0") & (adata.obs["condition"] == "LE")
    keep = np.ones(adata.n_obs, dtype=bool)
    thin_idx = np.where(mask.to_numpy())[0][5:]  # drop all but 5 of d0's LE cells
    keep[thin_idx] = False
    adata = adata[keep].copy()
    context = _context(tmp_path, adata)
    result = DifferentialStateStage().run(context)
    assert result.status == "success"  # other 7 donors still carry the test


def test_metrics_report_n_programs_tested_and_significant(tmp_path):
    context = _context(tmp_path, _adata())
    result = DifferentialStateStage().run(context)
    assert result.metrics["n_programs_tested"] == 2
    assert result.metrics["n_significant"] == 1
