import json
from html.parser import HTMLParser
from types import SimpleNamespace

import pandas as pd

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.artifacts import ArtifactManager
from cellquorum.core.publication import (
    _communication,
    abundance_sensitivity,
    write_publication_bundle,
)
from cellquorum.core.study_preflight import study_blockers


def test_null_abundance_figure_has_source_and_resolvable_links(tmp_path):
    cfg = CellQuorumConfig()
    cfg.run.study_questions = ["differences"]
    manager = ArtifactManager.from_root(tmp_path)
    source = pd.DataFrame(
        {
            "cell_type": ["B"],
            "effect_pp": [0.0],
            "fdr": [1.0],
            "bootstrap_ci_low_pp": [-1.0],
            "bootstrap_ci_high_pp": [1.0],
            "n_case": [3],
            "n_control": [3],
        }
    )
    artifact = manager.write_dataframe(
        source,
        relative_path="results/da_proportion_ttest.csv",
        name="da",
        description="Replicate-level abundance",
        index=False,
    )
    record = SimpleNamespace(
        status="success", reused_from_checkpoint=False, output_artifacts=[artifact]
    )
    panels = write_publication_bundle(cfg, [record], manager, None)
    panel = panels[0]
    assert panel["panel"] == "differences.abundance"
    exported = pd.read_csv(tmp_path / panel["source_data"])
    assert exported.value.item() == 0 and exported.fdr.item() == 1
    gallery = tmp_path / "reports/publication"

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            fields = dict(attrs)
            target = fields.get({"a": "href", "img": "src"}.get(tag))
            if target and target != "../report.html":
                assert (gallery / target).is_file()

    Links().feed((gallery / "index.html").read_text())
    assert not json.loads((gallery / "index.json").read_text())["omissions"]


def test_sensitivity_uses_donors_with_repeated_libraries():
    cfg = CellQuorumConfig()
    cfg.design.case = "case"
    cfg.design.control = "ctrl"
    cfg.design.paired = True
    rows = []
    for donor in ["a", "b", "c"]:
        for arm, counts in [("case", (10, 30)), ("ctrl", (20, 20))]:
            for sample in ["first", "second"] if donor == "a" else ["first"]:
                for ct, k in zip(["T", "B"], counts, strict=True):
                    rows.append((donor, arm, sample, ct, k, k / 40))
    table = pd.DataFrame(
        rows, columns=["donor", "condition", "sample", "cell_type", "count", "proportion"]
    )
    result = abundance_sensitivity(table, cfg)
    assert len(result) == 6 and result.same_direction.all()
    assert set(result.effect_pp) == {25, -25} and (result["rank"] == result.full_rank).all()


def test_missing_selected_nichenet_resources_are_actionable(tmp_path):
    cfg = CellQuorumConfig()
    cfg.cell_cell_communication.nichenet_ligand_target_matrix = str(tmp_path / "missing.rds")
    blockers = study_blockers(cfg, {"cell_cell_communication": ["nichenet"]})
    assert any("ligand_target_matrix" in b for b in blockers)
    assert any("sender and receiver" in b for b in blockers)


def test_fully_confounded_batch_stops_preflight(tmp_path):
    import anndata as ad
    import numpy as np

    obs = pd.DataFrame(
        {
            "donor": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "condition": ["reference"] * 4 + ["treated"] * 4,
            "batch": ["run1"] * 4 + ["run2"] * 4,
        },
        index=[str(i) for i in range(8)],
    )
    path = tmp_path / "confounded.h5ad"
    ad.AnnData(np.ones((8, 3)), obs=obs).write_h5ad(path)
    cfg = CellQuorumConfig.model_validate(
        {
            "input": {"h5ad": str(path)},
            "cohort": {"batch_key": "batch"},
            "design": {
                "donor_col": "donor",
                "condition_col": "condition",
                "control": "reference",
                "case": "treated",
                "paired": False,
            },
        }
    )
    assert any("fully confounded" in item for item in study_blockers(cfg, {}))


def test_stochastic_velocity_preflight_explains_incompatible_numpy(monkeypatch):
    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.core import study_preflight

    cfg = CellQuorumConfig.model_validate(
        {"project": {"name": "splice"}, "trajectory": {"velocity": {"mode": "stochastic"}}}
    )
    versions = {"scvelo": "0.3.4", "numpy": "2.4.6"}
    monkeypatch.setattr(study_preflight.metadata, "version", versions.__getitem__)
    errors = study_preflight.study_blockers(cfg, {"trajectory": ["velocity"]})
    assert any("cellquorum[lineage]" in error for error in errors)
    versions["numpy"] = "2.3.5"
    assert not any(
        "cellquorum[lineage]" in error
        for error in study_preflight.study_blockers(cfg, {"trajectory": ["velocity"]})
    )


def test_aucell_preflight_checks_default_population_backend(monkeypatch):
    from cellquorum.core import study_preflight

    monkeypatch.setattr(study_preflight.importlib.util, "find_spec", lambda name: None)
    config = CellQuorumConfig()
    assert study_blockers(config, {"state_scoring": ["score_genes"]}) == []
    assert any("decoupler" in e for e in study_blockers(config, {"state_scoring": ["aucell"]}))


def test_single_library_regulation_includes_scored_regulons(tmp_path):
    from cellquorum.core.publication import _regulation

    cfg = CellQuorumConfig()
    cfg.grn.group_by = "leiden"
    activity = tmp_path / "enrichment_activity_collectri.csv"
    pd.DataFrame({"cell_type": [0, 0], "source": ["A", "B"], "mean_score": [-2, 1]}).to_csv(
        activity, index=False
    )
    auc = tmp_path / "scenic_auc_mtx_grn.parquet"
    pd.DataFrame({"A(+)": [0.8, 0.6, 0.2]}, index=["c", "b", "a"]).to_parquet(auc)
    obs = pd.DataFrame({"leiden": ["0", "1", "1"]}, index=["a", "b", "c"])
    panels = _regulation({p.name: p for p in (activity, auc)}, obs, cfg)
    assert [p.method for p in panels] == ["CollecTRI", "pySCENIC"]
    data = panels[1].values.set_index("group")
    assert list(data.n_cells) == [1, 2]
    assert data.value.to_dict() == {"0": 0.2, "1": 0.7}
    assert "donor" not in data


def test_post_qc_support_preserves_categories_for_a_lost_donor_arm(tmp_path):
    import anndata as ad
    import numpy as np

    from cellquorum.core.study_preflight import record_qc_support

    obs = pd.DataFrame(
        {
            "arm": pd.Categorical(["ctrl", "ctrl", "case", "case"]),
            "person": pd.Categorical(["A", "B", "A", "B"]),
        },
        index=["c1", "c2", "c3", "c4"],
    )
    before = ad.AnnData(np.ones((4, 1)), obs=obs)
    source = tmp_path / "study.h5ad"
    before.write_h5ad(source)
    cfg = CellQuorumConfig()
    cfg.input.h5ad = source
    cfg.run.study_questions = ["differences"]
    cfg.design.donor_col, cfg.design.condition_col = "person", "arm"
    cfg.design.control, cfg.design.case = "ctrl", "case"
    ctx = SimpleNamespace(config=cfg, paths=SimpleNamespace(provenance=tmp_path))
    artifact = record_qc_support(ctx, "qc_evidence", before[:3])
    df = pd.read_csv(artifact.path).set_index(["person", "arm"])
    assert df.loc[("B", "case"), ["input_cells", "eligible_cells", "excluded_cells"]].tolist() == [
        1,
        0,
        1,
    ]
    assert df.eligible_cells.sum() == 3


def test_liana_publication_keeps_the_native_ribbons_and_their_exact_source(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz.chord import (
        liana_receiver_panel,
        population_palette,
    )

    cfg = CellQuorumConfig()
    cfg.run.study_questions = ["communication"]
    manager = ArtifactManager.from_root(tmp_path)
    labels = {"0": "Mono (C0)", "1": "Mono (C1)", "2": "NK (C2)"}
    palette = population_palette(list(labels.values()))
    assert palette[labels["0"]] == palette[labels["1"]] != palette[labels["2"]]
    raw = pd.DataFrame(
        [
            ("0", "2", "L", "R", "A", 0.5),
            ("0", "2", "L", "R", "A", 0.9),
            ("0", "2", "L", "R", "B", 0.9),
            ("1", "2", "L", "R", "A", 0.7),
            ("1", "1", "X", "Y", "A", 1.0),
        ],
        columns=["source", "target", "ligand", "receptor", "sample", "weight"],
    )
    artifacts = liana_receiver_panel(
        raw,
        labels,
        figures=tmp_path / "figures",
        results=tmp_path / "results",
        formats=("svg", "pdf"),
        dpi=100,
        top_k=8,
        palette=palette,
    )
    record = SimpleNamespace(
        status="success", reused_from_checkpoint=False, output_artifacts=artifacts
    )
    (panel,) = write_publication_bundle(cfg, [record], manager, None)
    gallery = tmp_path / "reports/publication"
    assert panel["preview"] and panel["method"] == "LIANA"
    assert (gallery / "communication.svg").read_bytes() == artifacts[0].path.read_bytes()
    assert (gallery / "communication.pdf").read_bytes() == artifacts[1].path.read_bytes()
    source = pd.read_csv(tmp_path / panel["source_data"])
    assert source.n_libraries.tolist() == [2, 1]
    assert source.libraries.map(json.loads).tolist() == [["A", "B"], ["A"]]
    assert source.weight.round(3).tolist() == [0.8, 0.7]
    assert source.sender_label.tolist() == ["Mono (C0)", "Mono (C1)"]
    assert "Dimitrov" in (gallery / "citations.md").read_text()


def test_communication_joins_complete_complexes_when_liana_exports_component_genes(tmp_path):
    cfg = CellQuorumConfig()
    cfg.design.case = "Tumor"
    cfg.cell_cell_communication.nichenet_receiver = "B"
    cfg.cell_cell_communication.nichenet_sender = "A"
    inputs = {
        "mnn_prioritization.csv": (
            "sender,receiver,ligand,receptor,group,prioritization_score\nA,B,L,R1_R2,Tumor,0.9\n"
        ),
        "liana_ranks.csv": (
            "sample,source,target,ligand,receptor,ligand_complex,receptor_complex,magnitude_rank\n"
            "S1,A,B,L,R1,L,R1_R2,0.03\n"
        ),
        "nichenet_activities.csv": "test_ligand,rank\nL,1\n",
        "nichenet_ligand_receptor_weights.csv": "ligand,receptor,weight\nL,R1_R2,1\n",
        "nichenet_ligand_target_weights.csv": "ligand,target,weight\nL,T,0.4\n",
        "mnn_sample_prioritization.csv": (
            "sample,sender,receiver,ligand,receptor,scaled_LR_pb_prod\nS1,A,B,L,R1_R2,2.3\n"
        ),
    }
    files = {name: tmp_path / name for name in inputs}
    for name, data in inputs.items():
        files[name].write_text(data)
    panels = {panel.key: panel.values for panel in _communication(files, pd.DataFrame(), cfg)}
    assert panels["communication.liana"][["ligand", "receptor", "value"]].values.tolist() == [
        ["L", "R1_R2", 0.03]
    ]
    assert panels["communication.mnn"]["value"].tolist() == [2.3]
