"""Python-side orchestration of ``pseudobulk_dream`` with a mocked R backend.

No R is needed: a fake Rscript backend records exactly what the method sends (counts,
metadata, contrast matrix, formula) and writes the files the real script would. These
tests pin the workflow order and the fatal-vs-skip boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _dream_synth import SITES, SUBJECTS, design_dict, make_adata, make_config

from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.core.exceptions import CellQuorumConfigError, CellQuorumDataError
from cellquorum.stages.comparative.differential_expression.dream_design import (
    human_formula,
    unique_slugs,
)
from cellquorum.stages.comparative.differential_expression.stage import DifferentialExpressionStage

ALL_SPECIMENS = [f"{s}{c}" for s in SUBJECTS for c in SITES.values()]


class FakeRscript:
    """Stands in for the Rscript backend; records calls and fabricates outputs."""

    def __init__(
        self,
        *,
        has_variance_partition: bool = True,
        returncode: int = 0,
        failed_genes: tuple[str, ...] = (),
    ):
        self.calls: list[dict] = []
        self.failed_genes = failed_genes
        self.has_vp = has_variance_partition
        self.returncode = returncode

    def _rscript_available(self) -> bool:
        return True

    def _r_package_available(self, pkg: str) -> bool:
        return self.has_vp if pkg == "variancePartition" else True

    def run_script(self, script, args, *, timeout=None):
        counts_csv, meta_csv, l_csv, formula, out_dir, ddf = args[:6]
        counts = pd.read_csv(counts_csv)
        L = pd.read_csv(l_csv)
        self.calls.append(
            {
                "specimens": list(counts["sample"]),
                "n_genes": counts.shape[1] - 1,
                "meta": pd.read_csv(meta_csv, index_col=0),
                "L": L.set_index("coef"),
                "formula": formula,
                "ddf": ddf,
                "n_workers": args[8],
            }
        )
        if self.returncode:
            return type("P", (), {"returncode": self.returncode, "stderr": "boom: model failed"})()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "gene": g, "contrast": c, "logFC": 0.0, "AveExpr": 5.0, "standard_error": 0.1,
                "moderated_standard_error": 0.1, "t": 0.0, "df_residual": 4.0, "df_total": 19.0,
                "PValue": 0.5, "FDR": 0.9,
            }
            for c in L.columns[1:]
            for g in ("g0", "g1")
        ]  # fmt: skip
        pd.DataFrame(rows).to_csv(out / "results_long.csv", index=False)
        if self.failed_genes:
            pd.DataFrame(
                {
                    "gene": list(self.failed_genes),
                    "error": ["response variance < 1e-05"] * len(self.failed_genes),
                }
            ).to_csv(out / "failed_genes.csv", index=False)
        (out / "fit_info.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "variancePartition": "9.9.9",
                    "n_genes_after_filter": 2,
                    "n_genes_failed": len(self.failed_genes),
                    "bpparam": {"class": "SerialParam", "workers": 1},
                    "filter_design_columns": ["(Intercept)", "cq_fx1", "cq_fx2", "cq_fx3"],
                }
            )
        )
        return type("P", (), {"returncode": 0, "stderr": ""})()


class Registry:
    def __init__(self, backend):
        self._b = backend

    def get(self, name):
        return self._b


def _run(tmp_path, adata, *, backend=None, config=None):
    backend = backend or FakeRscript()
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    ctx = PipelineContext(
        config=config or make_config(), paths=paths, adata=adata, backend_registry=Registry(backend)
    )
    return DifferentialExpressionStage().run(ctx), backend, paths


def _b_cells(skip=(), low=None):
    cells = {("B", s): 30 for s in ALL_SPECIMENS if s not in skip}
    if low:
        cells[("B", low)] = 8
    return cells


# --- universe, support and the 12-specimen invariant ---------------------------------------


def test_universal_population_enters_r_with_twelve_specimens_one_call_per_cell_type(tmp_path):
    res, backend, _ = _run(tmp_path, make_adata())
    assert res.status == "success"
    assert len(backend.calls) == 1  # one fit per cell type, all five contrasts in it
    call = backend.calls[0]
    assert sorted(call["specimens"]) == sorted(ALL_SPECIMENS)
    assert call["L"].shape == (4, 5)  # (Intercept)+3 fixed coefficients x 5 contrasts


def test_machine_design_sent_to_r_uses_python_coding_and_subject_stays_random(tmp_path):
    _, backend, _ = _run(tmp_path, make_adata())
    call = backend.calls[0]
    assert call["formula"] == "~ cq_fx1 + cq_fx2 + cq_fx3 + (1 | cq_re1)"
    meta = call["meta"]
    # Subject is a random-effect grouping column, never a fixed numeric column.
    assert list(meta.columns) == ["cq_fx1", "cq_fx2", "cq_fx3", "cq_re1"]
    assert meta.loc["P2J"].tolist()[:3] == [1, 1, 1]  # trt, jej, interaction
    assert meta.loc["C1J"].tolist()[:3] == [0, 1, 0]
    assert meta.loc["P2J", "cq_re1"] == "P2" and meta.loc["P2C", "cq_re1"] == "P2"
    assert list(call["L"].index) == ["(Intercept)", "cq_fx1", "cq_fx2", "cq_fx3"]
    dod = call["L"]["cq_c5"].tolist()
    assert dod == [0, 0, 0, 1]
    assert call["L"]["cq_c1"].tolist() == [0, 1, 0, 0]  # trt vs ctl in colon


def test_population_absent_in_one_specimen_stays_visible_and_is_never_zero_profile(tmp_path):
    adata = make_adata(cells=_b_cells(skip=("C2C",)))
    res, backend, paths = _run(tmp_path, adata)
    assert res.status == "success"
    b_call = next(c for c in backend.calls if "C2C" not in c["specimens"])
    assert len(b_call["specimens"]) == 11
    support = pd.read_csv(
        paths.results / "differential_expression/pseudobulk_dream/b/support.csv"
    ).set_index("sample_id")
    assert len(support) == 12  # the complete universe, not just observed specimens
    assert support.loc["C2C", "exclusion_reason"] == "population_absent"
    assert support.loc["C2C", "n_cells"] == 0 and not support.loc["C2C", "included_for_expression"]
    counts = pd.read_csv(
        paths.results / "differential_expression/pseudobulk_dream/b/pseudobulk_counts.csv",
        index_col=0,
    )
    assert "C2C" not in counts.index and (counts.sum(axis=1) > 0).all()


def test_below_min_cells_specimen_is_excluded_with_reason(tmp_path):
    adata = make_adata(cells=_b_cells(low="P3J"))
    _, backend, paths = _run(tmp_path, adata)
    support = pd.read_csv(
        paths.results / "differential_expression/pseudobulk_dream/b/support.csv"
    ).set_index("sample_id")
    assert support.loc["P3J", "exclusion_reason"] == "below_min_cells"
    assert all("P3J" not in c["specimens"] or len(c["specimens"]) == 12 for c in backend.calls)


def test_unmodified_counts_layer(tmp_path):
    adata = make_adata()
    before = adata.layers["counts"].copy()
    _run(tmp_path, adata)
    assert (adata.layers["counts"] != before).nnz == 0


# --- support loss => per-cell-type skip; never a silently simpler model ----------------------


def test_loss_of_a_factorial_cell_skips_only_that_cell_type(tmp_path):
    # Cell type B has no trt/jej specimen at all (the interaction cell is empty).
    adata = make_adata(cells=_b_cells(skip=("P2J", "P3J", "P4J")))
    res, backend, paths = _run(tmp_path, adata)
    assert res.status == "success"  # A still fitted
    summary = pd.read_csv(
        paths.results / "differential_expression/pseudobulk_dream/cell_type_summary.csv"
    ).set_index("cell_type")
    assert summary.loc["A", "status"] == "fitted"
    assert summary.loc["B", "status"] == "skipped"
    assert "not estimable" in summary.loc["B", "reason"] or "empty" in summary.loc["B", "reason"]
    assert len(backend.calls) == 1  # R was never invoked for B
    assert any("B skipped" in w for w in res.warnings)


def test_a_between_level_with_too_few_subjects_skips(tmp_path):
    # B retains a single trt subject -> fails min_subjects_per_between_level (2).
    adata = make_adata(cells=_b_cells(skip=("P3C", "P3J", "P4C", "P4J")))
    _, backend, paths = _run(tmp_path, adata)
    summary = pd.read_csv(
        paths.results / "differential_expression/pseudobulk_dream/cell_type_summary.csv"
    ).set_index("cell_type")
    assert summary.loc["B", "status"] == "skipped"
    assert (
        "independent subjects" in summary.loc["B", "reason"]
        or "specimens in" in summary.loc["B", "reason"]
    )


def test_every_cell_type_skipped_gives_a_method_skip(tmp_path):
    cfg = make_config(de={"pseudobulk": {"min_cells_per_sample": 10_000}})
    res, backend, _ = _run(tmp_path, make_adata(), config=cfg)
    assert res.status == "skipped" and backend.calls == []


def test_r_failure_is_recorded_per_cell_type_with_stderr(tmp_path):
    res, _, paths = _run(tmp_path, make_adata(), backend=FakeRscript(returncode=1))
    summary = pd.read_csv(
        paths.results / "differential_expression/pseudobulk_dream/cell_type_summary.csv"
    )
    assert summary.loc[0, "status"] == "failed" and "boom: model failed" in summary.loc[0, "reason"]
    assert res.status == "skipped"


# --- fatal conditions -------------------------------------------------------------------------


def test_corrupt_specimen_metadata_is_fatal_not_a_skip(tmp_path):
    adata = make_adata()
    idx = np.flatnonzero((adata.obs["specimen"] == "C1C").to_numpy())[:5]
    adata.obs["arm"] = adata.obs["arm"].astype(object)
    adata.obs.iloc[idx, adata.obs.columns.get_loc("arm")] = "trt"
    with pytest.raises(CellQuorumDataError, match="C1C"):
        _run(tmp_path, adata)


def test_corrupt_metadata_confined_to_one_cell_type_is_still_fatal(tmp_path):
    # Only cell type B cells carry the conflicting value; the universe (all cells) sees it.
    adata = make_adata(cells=_b_cells())
    rows = np.flatnonzero(
        ((adata.obs["specimen"] == "C1C") & (adata.obs["cell_type"] == "B")).to_numpy()
    )[:3]
    adata.obs["site"] = adata.obs["site"].astype(object)
    adata.obs.iloc[rows, adata.obs.columns.get_loc("site")] = "jej"
    with pytest.raises(CellQuorumDataError):
        _run(tmp_path, adata)


def test_between_subject_violation_is_fatal(tmp_path):
    # Specimen C1J is relabelled trt while its sibling C1C (same subject) stays ctl.
    adata = make_adata()
    adata.obs["arm"] = adata.obs["arm"].astype(object)
    adata.obs.loc[adata.obs["specimen"] == "C1J", "arm"] = "trt"
    with pytest.raises(CellQuorumConfigError, match="between_subject.*'C1'"):
        _run(tmp_path, adata)


def test_globally_empty_interaction_cell_is_fatal(tmp_path):
    adata = make_adata()
    adata = adata[~((adata.obs["arm"] == "trt") & (adata.obs["site"] == "jej"))].copy()
    with pytest.raises(CellQuorumConfigError, match="not estimable|not valid"):
        _run(tmp_path, adata)


def test_invalid_contrast_is_fatal(tmp_path):
    bad = [
        {
            "kind": "simple_effect",
            "name": "c",
            "factor": "arm",
            "numerator": "trt",
            "denominator": "ctl",
        }
    ]
    with pytest.raises(CellQuorumConfigError, match="ambiguous"):
        _run(tmp_path, make_adata(), config=make_config(contrasts=bad))


def test_subject_must_be_a_random_effect_when_a_within_subject_factor_is_declared(tmp_path):
    cfg = make_config(factorial=design_dict(random_effects=["site"]))
    with pytest.raises(CellQuorumConfigError, match="random effect"):
        _run(tmp_path, make_adata(), config=cfg)


ARM_ONLY = {
    "subject_col": "subject",
    "sample_col": "specimen",
    "factors": {"arm": {"reference": "ctl", "scope": "between_subject"}},
    "interactions": [],
    "random_effects": [],
}
ARM_CONTRAST = [
    {"kind": "simple_effect", "name": "trt_vs_ctl", "factor": "arm",
     "numerator": "trt", "denominator": "ctl"}
]  # fmt: skip


def test_fixed_effect_design_runs_when_every_subject_has_one_specimen(tmp_path):
    adata = make_adata()
    adata = adata[adata.obs["site"] == "colon"].copy()
    cfg = make_config(factorial=ARM_ONLY, contrasts=ARM_CONTRAST)
    result, backend, _ = _run(tmp_path, adata, config=cfg)
    call = backend.calls[0]
    assert "(1 |" not in call["formula"]
    assert call["formula"].replace(" ", "") == "~cq_fx1"
    assert len(call["specimens"]) == 6
    assert result.metrics["human_formula"] == "~ arm"


def test_fixed_effect_design_refuses_repeated_specimens_per_subject(tmp_path):
    cfg = make_config(factorial=ARM_ONLY, contrasts=ARM_CONTRAST)
    with pytest.raises(CellQuorumConfigError, match="contribute several specimens"):
        _run(tmp_path, make_adata(), config=cfg)


def test_covariates_and_legacy_interactions_are_rejected_not_ignored(tmp_path):
    with pytest.raises(CellQuorumConfigError, match="covariates"):
        _run(tmp_path, make_adata(), config=make_config(de={"covariates": ["batch"]}))
    with pytest.raises(CellQuorumConfigError, match="interactions"):
        _run(tmp_path, make_adata(), config=make_config(de={"interactions": [["arm", "site"]]}))


def test_missing_design_or_contrasts_is_a_config_error(tmp_path):
    with pytest.raises(CellQuorumConfigError, match="contrasts.factorial"):
        _run(tmp_path, make_adata(), config=make_config(contrasts=[]))


def test_missing_source_column_is_fatal(tmp_path):
    adata = make_adata()
    del adata.obs["site"]
    with pytest.raises(Exception, match="site"):
        _run(tmp_path, adata)


def test_missing_variance_partition_is_a_clean_skip(tmp_path):
    res, backend, _ = _run(
        tmp_path, make_adata(), backend=FakeRscript(has_variance_partition=False)
    )
    assert res.status == "skipped"
    assert "variancePartition" in " ".join(res.warnings) and backend.calls == []


def test_config_errors_surface_even_without_r(tmp_path):
    with pytest.raises(CellQuorumConfigError):
        _run(
            tmp_path,
            make_adata(),
            backend=FakeRscript(has_variance_partition=False),
            config=make_config(de={"covariates": ["x"]}),
        )


# --- artifacts and provenance -------------------------------------------------------------------


def test_artifact_layout_is_unique_and_never_overwritten(tmp_path):
    adata = make_adata(cells=_b_cells())
    res, _, paths = _run(tmp_path, adata)
    root = paths.results / "differential_expression/pseudobulk_dream"
    for cell in ("a", "b"):
        for f in (
            "support.csv", "design_report.json", "contrast_matrix.csv", "model_provenance.json",
            "trt_vs_ctl_in_colon.csv", "trt_vs_ctl_in_jej.csv", "jej_vs_colon_in_ctl.csv",
            "jej_vs_colon_in_trt.csv", "arm_x_site.csv",
        ):  # fmt: skip
            assert (root / cell / f).exists(), (cell, f)
    assert (root / "de_pseudobulk_dream_all.csv").exists()
    paths_seen = [str(a.path) for a in res.artifacts]
    assert len(paths_seen) == len(set(paths_seen))
    names = [a.name for a in res.artifacts]
    assert len(names) == len(set(names))


def test_combined_table_carries_the_output_contract(tmp_path):
    res, _, _ = _run(tmp_path, make_adata(cells=_b_cells()))
    de = pd.read_csv(next(a.path for a in res.artifacts if a.name == "de_results"))
    for col in (
        "gene", "cell_type", "contrast", "tested_effect", "logFC", "AveExpr", "standard_error",
        "t", "df_residual", "PValue", "FDR", "n_samples", "n_subjects", "formula", "backend",
    ):  # fmt: skip
        assert col in de.columns, col
    assert set(de["cell_type"]) == {"A", "B"} and de["contrast"].nunique() == 5
    inter = de[de["contrast"] == "arm_x_site"].iloc[0]
    assert inter["tested_effect"] == "arm x site: (trt jej - trt colon) - (ctl jej - ctl colon)"
    assert inter["formula"] == "~ arm * site + (1 | subject)"
    assert inter["n_samples"] == 12 and inter["n_subjects"] == 6


def test_provenance_records_model_coding_support_and_contrasts(tmp_path):
    _, _, paths = _run(tmp_path, make_adata(cells=_b_cells(skip=("C2C",))))
    prov = json.loads(
        (
            paths.results / "differential_expression/pseudobulk_dream/b/model_provenance.json"
        ).read_text()
    )
    assert prov["human_formula"] == "~ arm * site + (1 | subject)"
    assert prov["machine_formula"] == "~ cq_fx1 + cq_fx2 + cq_fx3 + (1 | cq_re1)"
    assert prov["fixed_effect_column_mapping"] == {
        "arm_trt": "cq_fx1", "site_jej": "cq_fx2", "arm_trt:site_jej": "cq_fx3"
    }  # fmt: skip
    assert prov["reference_levels"] == {"arm": "ctl", "site": "colon"}
    assert prov["ddf_requested"] == "adaptive"
    assert prov["n_specimens_universe"] == 12 and prov["n_specimens_included"] == 11
    assert prov["excluded_specimens"] == [
        {"sample_id": "C2C", "exclusion_reason": "population_absent"}
    ]
    assert prov["counts_layer"] == "counts" and prov["r"]["variancePartition"] == "9.9.9"
    assert prov["contrasts"]["contrasts"][4]["name"] == "arm_x_site"
    assert prov["contrast_support"]["arm_x_site"]["components"][0]["n_specimens"] >= 2
    assert "cellquorum_version" in prov


def test_requested_ddf_is_passed_to_r(tmp_path):
    _, backend, _ = _run(
        tmp_path, make_adata(), config=make_config(de={"dream": {"ddf": "Satterthwaite"}})
    )
    assert backend.calls[0]["ddf"] == "Satterthwaite"


def test_without_cell_type_col_one_pooled_population_is_analysed(tmp_path):
    cfg = make_config(de={"cell_type_col": None})
    _, backend, paths = _run(tmp_path, make_adata(cells=_b_cells()), config=cfg)
    assert len(backend.calls) == 1 and len(backend.calls[0]["specimens"]) == 12
    assert (
        paths.results / "differential_expression/pseudobulk_dream/all_cells/support.csv"
    ).exists()


# --- pure helpers --------------------------------------------------------------------------


def test_slugs_are_deterministic_and_collision_safe():
    slugs = unique_slugs(["B cell", "B_cell", "T/NK"])
    assert len(set(slugs.values())) == 3
    assert slugs["T/NK"] == "t_nk"
    assert unique_slugs(["B cell", "B_cell"]) == unique_slugs(["B_cell", "B cell"])


def test_human_formula():
    from cellquorum.config.design import FactorialDesignConfig

    fac = FactorialDesignConfig.model_validate(design_dict())
    assert human_formula(fac) == "~ arm * site + (1 | subject)"
    three = FactorialDesignConfig.model_validate(
        design_dict(
            factors={
                "arm": {"reference": "ctl"},
                "site": {"reference": "colon"},
                "batch": {"reference": "b1"},
            },
            interactions=[["arm", "site"]],
            random_effects=["subject", "litter"],
        )
    )  # fmt: skip
    assert human_formula(three) == "~ arm + site + batch + arm:site + (1 | subject) + (1 | litter)"


# --- Stage 4C: parallel workers, stale outputs, subject-level gates, failed genes ---------------


def test_worker_count_inherits_compute_n_jobs_and_defaults_to_serial(tmp_path):
    _, backend, _ = _run(tmp_path, make_adata())
    assert backend.calls[0]["n_workers"] == "1"  # no compute block -> conservative serial
    _, backend, paths = _run(tmp_path / "p", make_adata(), config=make_config(n_jobs=4))
    assert backend.calls[0]["n_workers"] == "4"
    prov = json.loads(
        (
            paths.results / "differential_expression/pseudobulk_dream/a/model_provenance.json"
        ).read_text()
    )
    assert (
        prov["parallel"]["n_jobs"] == 4 and prov["parallel"]["class"] == "SerialParam"
    )  # echoed from R
    assert prov["filter_design_columns"][0] == "(Intercept)"


def _seed_stale(paths):
    root = paths.results / "differential_expression/pseudobulk_dream"
    (root / "old_cell_type").mkdir(parents=True)
    (root / "old_cell_type" / "arm_x_site.csv").write_text("stale")
    (root / "de_pseudobulk_dream_all.csv").write_text("stale")
    (paths.results / "differential_expression" / "de_pseudobulk_edger.csv").write_text("edger")
    (paths.results / "unrelated.csv").write_text("keep")
    return root


def test_rerun_discards_only_the_method_owned_directory(tmp_path):
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    root = _seed_stale(paths)
    ctx = PipelineContext(
        config=make_config(),
        paths=paths,
        adata=make_adata(),
        backend_registry=Registry(FakeRscript()),
    )
    res = DifferentialExpressionStage().run(ctx)
    assert res.status == "success"
    assert not (root / "old_cell_type").exists()
    assert "stale" not in (root / "de_pseudobulk_dream_all.csv").read_text()
    # Siblings outside the method directory are untouched.
    assert (
        paths.results / "differential_expression" / "de_pseudobulk_edger.csv"
    ).read_text() == "edger"
    assert (paths.results / "unrelated.csv").read_text() == "keep"
    # Everything under the method directory was written by this invocation.
    registered = {Path(a.path).resolve() for a in res.artifacts}
    on_disk = {p.resolve() for p in root.rglob("*") if p.is_file() and "r_output" not in p.parts}
    assert on_disk <= registered
    assert (root / "invocation.json").exists()


def test_fewer_cell_types_on_rerun_leave_no_stale_folder(tmp_path):
    adata_two = make_adata(cells=_b_cells())
    _run(tmp_path, adata_two)
    root = tmp_path / "results/differential_expression/pseudobulk_dream"
    assert (root / "b").exists()
    _run(tmp_path, make_adata())  # only cell type A this time
    assert not (root / "b").exists() and (root / "a").exists()
    assert set(pd.read_csv(root / "de_pseudobulk_dream_all.csv")["cell_type"]) == {"A"}


def test_symlinked_method_directory_is_refused_not_followed(tmp_path):
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("x")
    (paths.results / "differential_expression").mkdir()
    (paths.results / "differential_expression" / "pseudobulk_dream").symlink_to(outside)
    ctx = PipelineContext(
        config=make_config(),
        paths=paths,
        adata=make_adata(),
        backend_registry=Registry(FakeRscript()),
    )
    with pytest.raises(CellQuorumConfigError, match="symbolic link"):
        DifferentialExpressionStage().run(ctx)
    assert (outside / "keep.txt").exists()


def test_invalid_config_does_not_wipe_previous_results(tmp_path):
    _run(tmp_path, make_adata())
    root = tmp_path / "results/differential_expression/pseudobulk_dream"
    with pytest.raises(CellQuorumConfigError):
        _run(tmp_path, make_adata(), config=make_config(de={"covariates": ["x"]}))
    assert (root / "a" / "model_provenance.json").exists()


def _two_specimens_per_subject_report(n_trt_subjects: int):
    """Cells with plenty of specimens but few DISTINCT subjects (two specimens per subject/site)."""
    from cellquorum.config.design import FactorialDesignConfig
    from cellquorum.config.factorial_design import analyze_factorial_design

    rows = []
    subjects = {
        "C1": "ctl",
        "C2": "ctl",
        "C3": "ctl",
        **{f"P{i}": "trt" for i in range(n_trt_subjects)},
    }
    for subj, arm in subjects.items():
        for site in ("colon", "jej"):
            for rep in (1, 2):
                rows.append(
                    {"specimen": f"{subj}{site}{rep}", "subject": subj, "arm": arm, "site": site}
                )
    table = pd.DataFrame(rows).set_index("specimen")
    design = FactorialDesignConfig.model_validate(design_dict())
    report = analyze_factorial_design(
        table, design=design, sample_col="specimen", subject_col="subject"
    )
    return design, report


def test_factorial_cell_needs_distinct_subjects_not_just_specimens():
    from cellquorum.stages.comparative.differential_expression.config import PseudobulkSupportConfig
    from cellquorum.stages.comparative.differential_expression.dream_design import (
        support_gate_failures,
    )

    design, report = _two_specimens_per_subject_report(n_trt_subjects=1)
    # Every trt cell has 2 specimens (so the specimen gate passes) but only ONE subject.
    gates = PseudobulkSupportConfig(min_subjects_per_between_level=1)
    failures = support_gate_failures(report, design, gates)
    assert not any("specimens in" in f for f in failures)
    assert any("distinct subjects in arm x site cell" in f for f in failures)
    assert any("'trt', 'site': 'colon'" in f or "trt" in f for f in failures)


def test_distinct_subject_gate_passes_with_enough_subjects_and_is_configurable():
    from cellquorum.stages.comparative.differential_expression.config import PseudobulkSupportConfig
    from cellquorum.stages.comparative.differential_expression.dream_design import (
        support_gate_failures,
    )

    design, report = _two_specimens_per_subject_report(n_trt_subjects=2)
    assert support_gate_failures(report, design, PseudobulkSupportConfig()) == []
    strict = PseudobulkSupportConfig(
        min_subjects_per_factorial_cell=3, min_subjects_per_between_level=1
    )
    assert any(
        "fewer than 3 distinct subjects" in f for f in support_gate_failures(report, design, strict)
    )
    assert PseudobulkSupportConfig().min_subjects_per_factorial_cell == 2


def test_failed_genes_file_is_registered_and_counted(tmp_path):
    res, _, paths = _run(tmp_path, make_adata(), backend=FakeRscript(failed_genes=("g7", "g9")))
    assert any(a.name == "dream_failed_genes:a" for a in res.artifacts)
    failed = pd.read_csv(
        paths.results / "differential_expression/pseudobulk_dream/a/r_output/failed_genes.csv"
    )
    assert list(failed["gene"]) == ["g7", "g9"]
    summary = pd.read_csv(
        paths.results / "differential_expression/pseudobulk_dream/cell_type_summary.csv"
    )
    assert summary.loc[0, "n_genes_failed"] == 2


# --- DE eligibility (QC fit AND inference masks) -------------------------------------------------


def _with_qc(adata, ineligible_cells):
    """Add QC columns; ``ineligible_cells`` is a boolean array of cells QC excludes from DE."""
    ok = ~np.asarray(ineligible_cells, dtype=bool)
    adata.obs["qc_state_initial"] = np.where(ok, "core", "borderline")
    adata.obs["qc_fit_de"] = ok
    adata.obs["qc_inference_de"] = ok
    return adata


def test_ineligible_cells_never_enter_a_pseudobulk_but_the_specimen_stays_in_the_universe(tmp_path):
    adata = make_adata(cells=_b_cells())
    spec = adata.obs["specimen"].to_numpy()
    pop = (adata.obs["cell_type"] == "B").to_numpy()
    rng = np.random.default_rng(0)
    bad = np.zeros(adata.n_obs, dtype=bool)
    bad |= (spec == "C1C") & pop  # every B cell of C1C is QC-excluded
    idx = np.flatnonzero((spec == "C2J") & pop)
    bad[idx[:20]] = True  # 30 -> 10 eligible in C2J (below the 20-cell floor)
    bad |= rng.random(adata.n_obs) < 0.0
    _with_qc(adata, bad)
    res, backend, paths = _run(tmp_path, adata)
    assert res.status == "success"
    support = pd.read_csv(
        paths.results / "differential_expression/pseudobulk_dream/b/support.csv"
    ).set_index("sample_id")
    assert len(support) == 12
    c1c = support.loc["C1C"]
    assert (c1c["n_cells_observed"], c1c["n_cells_de_eligible"], c1c["n_cells_qc_excluded"]) == (
        30,
        0,
        30,
    )
    assert c1c["exclusion_reason"] == "population_present_but_no_de_eligible_cells"
    c2j = support.loc["C2J"]
    assert (c2j["n_cells_observed"], c2j["n_cells_de_eligible"]) == (30, 10)
    assert c2j["exclusion_reason"] == "below_min_cells_after_qc"
    counts_used = backend.calls[-1]["specimens"]  # cell type B was fitted last
    assert "C1C" not in counts_used and "C2J" not in counts_used
    prov = json.loads(
        (
            paths.results / "differential_expression/pseudobulk_dream/b/model_provenance.json"
        ).read_text()
    )
    assert prov["de_eligibility"]["rule"] == "qc_fit_de AND qc_inference_de"
    assert prov["n_cells_qc_excluded"] == 50


def test_pseudobulk_counts_exclude_the_ineligible_cells_exactly(tmp_path):
    adata = make_adata()
    spec = adata.obs["specimen"].to_numpy()
    bad = np.zeros(adata.n_obs, dtype=bool)
    idx = np.flatnonzero(spec == "P2J")
    bad[idx[:15]] = True
    _with_qc(adata, bad)
    _run(tmp_path, adata)
    got = pd.read_csv(
        tmp_path / "results/differential_expression/pseudobulk_dream/a/pseudobulk_counts.csv",
        index_col=0,
    )
    dense = adata.layers["counts"].toarray()
    keep = (spec == "P2J") & ~bad
    np.testing.assert_array_equal(got.loc["P2J"].to_numpy(), dense[keep].sum(axis=0))


def test_qc_state_without_its_de_masks_is_fatal_not_a_silent_fallback(tmp_path):
    from cellquorum.core.exceptions import CellQuorumDataError

    adata = make_adata()
    adata.obs["qc_state_initial"] = "core"  # QC ran, but the DE masks are absent
    with pytest.raises(CellQuorumDataError, match="mask.*missing"):
        _run(tmp_path, adata)


def test_no_qc_columns_means_every_cell_is_eligible_and_it_is_recorded(tmp_path):
    _, _, paths = _run(tmp_path, make_adata())
    prov = json.loads(
        (
            paths.results / "differential_expression/pseudobulk_dream/a/model_provenance.json"
        ).read_text()
    )
    assert prov["de_eligibility"]["rule"].startswith("none")
    assert prov["de_eligibility"]["n_cells_excluded"] == 0
