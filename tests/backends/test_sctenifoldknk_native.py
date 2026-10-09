"""Opt-in integration test that invokes the actual scTenifoldpy package."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellquorum.backends.sctenifoldknk_backend import ScTenifoldKnkBackend
from cellquorum.stages.gene_regulation.perturbation.config import ScTenifoldKnkConfig


@pytest.mark.integration
def test_native_non_tf_gene_knockout(tmp_path: Path) -> None:
    python = os.environ.get("CELLQUORUM_SCTENIFOLDKNK_PYTHON")
    if not python:
        pytest.skip("Set CELLQUORUM_SCTENIFOLDKNK_PYTHON to opt into native package test")
    rng = np.random.default_rng(42)
    latent = rng.uniform(0.3, 4, 60)
    x = rng.poisson(latent[None, :] * rng.uniform(0.5, 2, (24, 1)))
    matrix = pd.DataFrame(
        x,
        index=["non_TF_gene"] + [f"G{i}" for i in range(23)],
        columns=[f"c{i}" for i in range(60)],
    )
    input_path = tmp_path / "counts.csv"
    matrix.to_csv(input_path)
    cfg = ScTenifoldKnkConfig(
        ko_genes=["non_TF_gene"],
        n_nets=2,
        n_samp_cells=40,
        tensor_max_iter=30,
        manifold_dim=3,
        network_quantile=0.5,
    ).model_dump()
    cfg["seed"] = 42
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    output = tmp_path / "native.csv"
    result = ScTenifoldKnkBackend(python_executable=python).run_script(
        ["--input", str(input_path), "--config", str(config_path), "--output", str(output)],
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    table = pd.read_csv(output)
    assert len(table) == len(matrix) and set(table.Gene) == set(matrix.index)
    assert np.isfinite(table.Distance).all()
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert metadata["package"] == "scTenifoldpy"
    assert metadata["n_knockout_outgoing_edges"] > 0
    assert metadata["n_fitted_cells"] == 60
    assert len(metadata["fitted_cell_ids"]) == 60
    assert (pd.read_csv(output.with_suffix(".outgoing_edges.csv")).source == "non_TF_gene").all()
