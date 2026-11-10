"""A researcher can configure their own counts and design through the installed CLI."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from typer.testing import CliRunner

from cellquorum.cli.app import app
from cellquorum.config.loader import load_config


@pytest.fixture
def study(tmp_path: Path) -> Path:
    obs = pd.DataFrame(
        {
            "identity": ["T", "B"] * 6,
            "animal": np.repeat([1, 2, 3], 4),
            "arm": ["A", "A", "B", "B"] * 3,
        },
        index=[f"c{i}" for i in range(12)],
    )
    data = ad.AnnData(csr_matrix(np.full((12, 3), 1.5)), obs=obs)
    data.layers["original"] = csr_matrix(np.ones((12, 3), dtype=np.int32))
    path = tmp_path / "my data.h5ad"
    data.write_h5ad(path)
    return path


def _init(study, target, *answers):
    return CliRunner().invoke(
        app, ["init", str(target), "--input", str(study)], input="\n".join(answers) + "\n"
    )


def test_annotated_study_sets_up_a_paired_test_in_real_columns(study, tmp_path):
    target = tmp_path / "paired study"
    result = _init(
        study,
        target,
        "abundance",
        "layers/original",
        "mouse",
        "mixed",
        "arm",
        "B",
        "",
        "1",
        "identity",
        "animal",
        "arm",
        "A",
        "B",
        "paired",
    )
    assert result.exit_code == 0, result.output
    config = load_config(target / "config.yaml")
    assert config.paths.output_dir == target / "run"
    assert config.input.counts_layer == config.qc_evidence.metrics.layer == "original"
    assert config.design.paired and config.design.donor_col == "animal"
    assert config.design.control == "A" and config.design.case == "B"
    assert config.cohort.sample_key is None
    assert config.qc_evidence.graded.nuclei_values == ["B"]
    assert config.project.species_id == 10090
    assert {s for s, on in config.stages.model_dump().items() if on} == {
        "qc_evidence",
        "composition",
        "differential_abundance",
    }
    assert "3 complete replicates" in result.output
    assert not (target / "run").exists()
    assert _init(study, target).exit_code == 1
    assert config == load_config(target / "config.yaml")


def test_explore_accepts_counts_in_x_and_preserves_other_layers(tmp_path):
    path = tmp_path / "data.h5ad"
    data = ad.AnnData(np.ones((5, 4)))
    data.layers["counts"] = np.full(data.shape, 1.7)
    data.layers["cellquorum_normalized"] = np.ones(data.shape)
    data.write_h5ad(path)
    target = tmp_path / "explore"
    result = _init(path, target, "explore", "X", "human", "nuclei", "2")
    assert result.exit_code == 0, result.output
    config = load_config(target / "config.yaml")
    assert config.input.counts_layer is None
    assert config.stages.embeddings and not config.stages.differential_abundance
    assert not config.qc_evidence.graded.nuclear_axis_applicable
    assert config.qc_evidence.floors.min_genes_per_cell == 2
    norm = config.preprocessing.normalization
    assert norm.preserve_counts_layer == config.feature_selection.counts_layer != "counts"
    assert norm.output_layer == config.dimensionality.input_layer != "cellquorum_normalized"


@pytest.mark.parametrize(
    "answers, message",
    [
        (["explore", "X"], "raw counts"),
        (
            [
                "abundance",
                "layers/original",
                "human",
                "cells",
                "",
                "1",
                "identity",
                "animal",
                "arm",
                "A",
                "B",
                "independent",
            ],
            "Replicate IDs occur in both arms",
        ),
    ],
)
def test_invalid_counts_or_design_leave_no_config(study, tmp_path, answers, message):
    target = tmp_path / "bad"
    result = _init(study, target, *answers)
    assert result.exit_code == 1
    assert message in result.output
    assert not target.exists()


def test_question_recipe_accepts_a_replicated_study_without_existing_cell_labels(study, tmp_path):
    target = tmp_path / "new-cohort"
    result = _init(
        study,
        target,
        "questions",
        "layers/original",
        "human",
        "cells",
        "",
        "1",
        "differences",
        "y",
        "",
        "animal",
        "arm",
        "A",
        "B",
        "paired",
        "",
        "y",
    )
    assert result.exit_code == 0, result.output
    config = load_config(target / "config.yaml")
    assert config.design.paired
    assert config.differential_expression.cell_type_col == "leiden"
    assert config.run.study_questions == ["differences"]
    assert config.run.sensitivity_analysis and config.stages.clustering
    assert config.run.require_selected_methods


def test_population_recipe_uses_independent_marker_evidence(study, tmp_path):
    target = tmp_path / "populations"
    markers = tmp_path / "markers.yaml"
    markers.write_text("T: [CD3D, CD3E]\nB: [CD79A, MS4A1]\n")
    result = _init(
        study,
        target,
        "questions",
        "layers/original",
        "human",
        "cells",
        "",
        "1",
        "populations",
        "",
        str(markers),
        "",
        "",
    )
    assert result.exit_code == 0, result.output
    config = load_config(target / "config.yaml")
    assert config.stages.annotation_viz and config.state_scoring_viz.descriptive
    assert config.discovery.cell_type_col == config.population_identity.candidate_key == "leiden"
    assert config.population_identity.annotation_keys == [config.annotation.key_added]
    assert config.annotation.key_added in config.embeddings.color_by
    assert "figures/annotation/marker_dotplot.pdf" in config.run.required_outputs["annotation_viz"]
