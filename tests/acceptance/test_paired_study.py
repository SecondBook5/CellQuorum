"""Run the installed pipeline on the public, paired Kang PBMC reference study."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.mark.integration
@pytest.mark.r
@pytest.mark.slow
def test_kang_paired_study(tmp_path):
    study_dir = os.environ.get("CELLQUORUM_PAIRED_STUDY_DIR")
    if not study_dir:
        pytest.skip("set CELLQUORUM_PAIRED_STUDY_DIR to the directory containing data/")
    study = Path(study_dir).resolve()
    recipe_name = "kang_paired_analysis.yaml"
    recipe = Path(sysconfig.get_path("data")) / "share" / "cellquorum" / "workflows" / recipe_name
    if not recipe.exists():  # editable source installation
        recipe = Path(__file__).resolve().parents[2] / "configs" / recipe_name
    run_dir = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cellquorum.cli.app",
            "run",
            "-c",
            str(recipe),
            "-o",
            str(run_dir),
            "--quiet",
            "--json",
        ],
        cwd=study,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["successful_stages"] == summary["enabled_stages"]
    assert len(summary["successful_stages"]) == 18
    assert not summary["unable_to_run_stages"]
    assert len(summary["disabled_stages"]) == 23
    assert not summary["failed_stages"]

    records = {
        row["stage_name"]: row
        for row in json.loads((run_dir / "provenance/stage_execution_records.json").read_text())
    }
    design = records["differential_expression"]["metrics"]
    assert design["paired"] and design["n_complete_donor_pairs"] == 8
    assert design["n_pseudosamples"] == 16 and design["tested_effect"] == "stim vs ctrl"

    report = (run_dir / "reports/report.html").read_text()
    settings = yaml.safe_load(recipe.read_text())["run"]
    assert "18 completed · 0 skipped · 0 failed · 23 not selected" in report
    for name in summary["disabled_stages"]:
        skip = records[name]["skip_reason"]
        assert skip["details"]["plan_status"] == "disabled"
        assert skip["details"]["implemented"]
        assert skip["reason"] == settings["disabled_stage_reasons"][name]
    assert {"integration_gate", "molecular_inference"}.isdisjoint(records)
    required = settings["required_outputs"]
    for stage, files in required.items():
        recorded = {Path(item["path"]) for item in records[stage]["output_artifacts"]}
        for rel in files:
            path = run_dir / rel
            assert path in recorded and path.stat().st_size > 0
            assert f'href="../{rel}"' in report
            if path.suffix == ".pdf":
                assert path.read_bytes().startswith(b"%PDF-")
    assert (run_dir / "objects/final_annotated.h5ad").stat().st_size > 0

    results = run_dir / "results"
    de = pd.read_csv(results / "de_pseudobulk_edger_by_subcluster.csv")
    da = pd.read_csv(results / "da_proportion_ttest.csv")
    assert len(da) == 8 and da["cell_type"].is_unique
    assert da["paired"].all() and (da[["n_case", "n_control"]] == 8).all().all()
    assert de["FDR"].between(0, 1).all() and da["fdr"].between(0, 1).all()
    assert set(de["cell_type"]) == set(da["cell_type"])

    states = pd.read_csv(results / "differential_state_results.csv").set_index("program")
    assert set(states.index) == {"state_interferon_isg", "state_stress_hsp", "state_hypoxia_hif"}
    assert (states["n_pairs"] == 8).all() and (states["test"] == "wilcoxon_signed_rank").all()
    assert states.loc["state_interferon_isg", "effect"] > 0
    assert states.loc["state_interferon_isg", "padj"] < 0.05
    populations = pd.read_csv(results / "population_identity/tables/population_summary.csv")
    votes = pd.read_csv(results / "population_identity/tables/population_by_annotation.csv")
    assert set(votes["population_id"]) == set(da["cell_type"])
    assert populations["population_source"].eq("annotation").all()
    assert votes["n_cells"].sum() == populations["n_cells"].sum()
    assert (
        votes.groupby("population_id")["fraction_within_population"]
        .sum()
        .between(0.999, 1.001)
        .all()
    )
    assert (
        votes.loc[
            votes.population_id.eq("B cells") & votes.marker_vote_label.eq("B cells"),
            "fraction_within_population",
        ].item()
        > 0.8
    )

    pooled = pd.read_csv(results / "enrichment_gsea_hallmark.csv")
    split = pd.read_csv(results / "enrichment_gsea_hallmark_by_cell_type.csv")
    assert pooled["source"].nunique() == 50
    eligible = de.groupby("cell_type")["gene"].nunique()
    assert len(eligible[eligible >= 1000]) == 6
    assert set(split["cell_type"]) == set(eligible[eligible >= 1000].index)
    assert (split["permutations"] == 1000).all()
    interferon = split[split["source"] == "HALLMARK_INTERFERON_ALPHA_RESPONSE"]
    assert len(interferon) == 6 and (interferon["score"] > 0).all()
    assert (interferon["padj"] < 0.05).all() and (interferon["leading_edge_size"] > 0).all()
