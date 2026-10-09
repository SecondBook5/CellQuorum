"""``pseudobulk_dream`` against the real R stack (edgeR + variancePartition).

Synthetic 6-subject / 12-specimen split-plot data with a KNOWN log-rate shift in a block
of genes. The primary oracle is the estimand's sign and approximate magnitude -- never a
p-value threshold -- plus an independent hand calculation of the per-subject tissue
differences ``D_i = site_jej_i - site_colon_i`` on the same pseudobulk counts.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from _dream_synth import EFFECT_GENES, N_GENES, SUBJECTS, make_adata, make_config

from cellquorum.backends.registry import build_default_backend_registry
from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.stages.comparative.differential_expression.stage import DifferentialExpressionStage


def _r_has(*packages: str) -> bool:
    if shutil.which("Rscript") is None:
        return False
    expr = " && ".join(f"requireNamespace('{p}', quietly=TRUE)" for p in packages)
    r = subprocess.run(
        ["Rscript", "--vanilla", "-e", f"quit(status=ifelse({expr},0,1))"],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


pytestmark = [
    pytest.mark.r,
    pytest.mark.integration,
    pytest.mark.skipif(
        not _r_has("edgeR", "variancePartition", "jsonlite"),
        reason="edgeR/variancePartition/jsonlite not installed",
    ),
]

MAG = 1.0  # natural-log shift on the effect genes
LOG2_MAG = MAG / np.log(2)  # the expected log2 fold change
ROOT = "differential_expression/pseudobulk_dream"


def _fit(tmp_path, adata, **de):
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    ctx = PipelineContext(
        config=make_config(de=de or None),
        paths=paths,
        adata=adata,
        backend_registry=build_default_backend_registry(),
    )
    res = DifferentialExpressionStage().run(ctx)
    return res, paths


def _table(res) -> pd.DataFrame:
    return pd.read_csv(next(a.path for a in res.artifacts if a.name == "de_results"))


def _median_lfc(table, contrast, genes="effect", cell_type="A") -> float:
    t = table[(table["contrast"] == contrast) & (table["cell_type"] == cell_type)]
    idx = {f"g{i}" for i in range(N_GENES)[EFFECT_GENES]}
    t = t[t["gene"].isin(idx)] if genes == "effect" else t[~t["gene"].isin(idx)]
    return float(t["logFC"].median())


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory):
    out = {}
    for effect in ("interaction", "arm", "site"):
        res, paths = _fit(tmp_path_factory.mktemp(effect), make_adata(effect=effect, magnitude=MAG))
        out[effect] = (_table(res), paths, res)
    return out


def test_installed_stack_is_recorded(scenarios):
    _, paths, _ = scenarios["interaction"]
    prov = json.loads((paths.results / ROOT / "a/model_provenance.json").read_text())
    print("variancePartition", prov["r"]["variancePartition"], "edgeR", prov["r"]["edgeR"])
    assert prov["r"]["variancePartition"] and prov["r"]["status"] == "ok"
    assert prov["r"]["normalization"].startswith("TMM")
    assert prov["r"]["ddf_requested"] == "adaptive"


def test_universal_population_has_twelve_pseudobulks_and_five_contrasts(scenarios):
    table, paths, _ = scenarios["interaction"]
    counts = pd.read_csv(paths.results / ROOT / "a/pseudobulk_counts.csv", index_col=0)
    assert counts.shape[0] == 12
    assert table["contrast"].nunique() == 5


def test_interaction_only_effect_has_the_difference_of_differences_sign(scenarios):
    table, *_ = scenarios["interaction"]
    dod = _median_lfc(table, "arm_x_site")
    assert dod == pytest.approx(LOG2_MAG, abs=0.4) and dod > 0
    # The shifted cell (trt, jej) shows up in exactly the contrasts that contain it.
    assert _median_lfc(table, "trt_vs_ctl_in_jej") == pytest.approx(LOG2_MAG, abs=0.4)
    assert _median_lfc(table, "jej_vs_colon_in_trt") == pytest.approx(LOG2_MAG, abs=0.4)
    assert abs(_median_lfc(table, "trt_vs_ctl_in_colon")) < 0.3
    assert abs(_median_lfc(table, "jej_vs_colon_in_ctl")) < 0.3
    # Null genes carry no interaction.
    assert abs(_median_lfc(table, "arm_x_site", genes="null")) < 0.15
    # The interaction is detected on most of the shifted genes.
    inter = table[(table["contrast"] == "arm_x_site") & (table["cell_type"] == "A")]
    shifted = inter[inter["gene"].isin({f"g{i}" for i in range(N_GENES)[EFFECT_GENES]})]
    assert (shifted["FDR"] < 0.05).mean() > 0.5


def test_interaction_estimate_agrees_with_the_per_subject_difference_oracle(scenarios):
    table, paths, _ = scenarios["interaction"]
    counts = pd.read_csv(paths.results / ROOT / "a/pseudobulk_counts.csv", index_col=0)
    logcpm = np.log2((counts + 0.5).div(counts.sum(axis=1) + 1, axis=0) * 1e6)
    arm_of = {f"{s}{c}": a for s, a in SUBJECTS.items() for c in "CJ"}
    d = pd.DataFrame(
        {s: logcpm.loc[f"{s}J"] - logcpm.loc[f"{s}C"] for s in SUBJECTS}
    ).T  # subjects x genes: D_i = jej_i - colon_i
    trt = [s for s, a in SUBJECTS.items() if a == "trt"]
    ctl = [s for s, a in SUBJECTS.items() if a == "ctl"]
    oracle = d.loc[trt].mean() - d.loc[ctl].mean()
    assert arm_of  # (kept for readability of the oracle above)
    t = table[(table["contrast"] == "arm_x_site") & (table["cell_type"] == "A")].set_index("gene")
    genes = [f"g{i}" for i in range(N_GENES)[EFFECT_GENES]]
    diff = (t.loc[genes, "logFC"] - oracle[genes]).abs()
    # voom precision weights differ from the unweighted mean, so allow modest slack.
    assert diff.median() < 0.25
    assert np.corrcoef(t.loc[genes, "logFC"], oracle[genes])[0, 1] > 0.8


def test_genotype_like_main_effect_does_not_create_an_interaction(scenarios):
    table, *_ = scenarios["arm"]
    assert _median_lfc(table, "trt_vs_ctl_in_colon") == pytest.approx(LOG2_MAG, abs=0.4)
    assert _median_lfc(table, "trt_vs_ctl_in_jej") == pytest.approx(LOG2_MAG, abs=0.4)
    assert abs(_median_lfc(table, "arm_x_site")) < 0.3
    assert abs(_median_lfc(table, "jej_vs_colon_in_ctl")) < 0.3
    assert abs(_median_lfc(table, "jej_vs_colon_in_trt")) < 0.3


def test_tissue_like_main_effect_does_not_create_an_interaction(scenarios):
    table, *_ = scenarios["site"]
    assert _median_lfc(table, "jej_vs_colon_in_ctl") == pytest.approx(LOG2_MAG, abs=0.4)
    assert _median_lfc(table, "jej_vs_colon_in_trt") == pytest.approx(LOG2_MAG, abs=0.4)
    assert abs(_median_lfc(table, "arm_x_site")) < 0.3
    assert abs(_median_lfc(table, "trt_vs_ctl_in_colon")) < 0.3
    assert abs(_median_lfc(table, "trt_vs_ctl_in_jej")) < 0.3


def test_output_columns_are_finite_and_consistent(scenarios):
    table, *_ = scenarios["interaction"]
    t = table[table["contrast"] == "arm_x_site"]
    assert np.isfinite(t[["logFC", "t", "PValue", "FDR", "df_residual", "df_total"]]).all().all()
    # The moderated t is logFC over the moderated SE.
    np.testing.assert_allclose(t["logFC"] / t["moderated_standard_error"], t["t"], rtol=1e-6)
    assert (t["standard_error"] > 0).all() and (t["df_total"] >= t["df_residual"]).all()
    assert (t["FDR"] >= t["PValue"] - 1e-12).all()


def test_adaptive_ddf_is_kenward_roger_at_twelve_specimens(tmp_path_factory, scenarios):
    # Observed behaviour of the installed package: with <= 20 samples 'adaptive' IS
    # Kenward-Roger (identical per-gene denominator df), unlike Satterthwaite.
    base, *_ = scenarios["interaction"]
    adata = make_adata(effect="interaction", magnitude=MAG)
    kr, _ = _fit(tmp_path_factory.mktemp("kr"), adata, dream={"ddf": "Kenward-Roger"})
    sat, _ = _fit(tmp_path_factory.mktemp("sat"), adata, dream={"ddf": "Satterthwaite"})

    def df(t):
        return t[(t["contrast"] == "arm_x_site") & (t["cell_type"] == "A")].set_index("gene")[
            "df_residual"
        ]

    np.testing.assert_allclose(df(base), df(_table(kr)), rtol=1e-9)
    assert not np.allclose(df(base), df(_table(sat)), rtol=1e-3)


def test_one_missing_repeated_specimen_is_reported_and_still_fitted(tmp_path):
    cells = {("A", "C1J"): 0}  # every other A specimen keeps its default 40 cells
    res, paths = _fit(tmp_path, make_adata(effect="interaction", magnitude=MAG, cells=cells))
    table = _table(res)
    support = pd.read_csv(paths.results / ROOT / "a/support.csv").set_index("sample_id")
    # The specimen has no cells of ANY population, so the experiment itself lacks it.
    assert "C1J" not in support.index and len(support) == 11
    design = json.loads((paths.results / ROOT / "a/design_report.json").read_text())
    site = next(f for f in design["factors"] if f["name"] == "site")
    assert site["incomplete_subjects"] == {"C1": ["jej"]}
    assert design["n_samples"] == 11 and design["n_subjects"] == 6
    assert _median_lfc(table, "arm_x_site") > 0.5  # sign survives the missing specimen


def test_sparse_population_skips_cleanly_and_others_still_fit(tmp_path):
    specimens = [f"{k}{c}" for k in SUBJECTS for c in "CJ"]
    cells = {("B", s): 30 for s in specimens if s not in ("P2J", "P3J", "P4J")}
    res, paths = _fit(tmp_path, make_adata(cells=cells))
    summary = pd.read_csv(paths.results / ROOT / "cell_type_summary.csv").set_index("cell_type")
    assert summary.loc["A", "status"] == "fitted" and summary.loc["B", "status"] == "skipped"
    assert "arm x site" in summary.loc["B", "reason"] or "empty" in summary.loc["B", "reason"]
    assert set(_table(res)["cell_type"]) == {"A"}


def test_no_genes_surviving_filter_is_a_recorded_skip(tmp_path):
    res, paths = _fit(tmp_path, make_adata(), min_count=10_000_000, min_total_count=10_000_000)
    assert res.status == "skipped"
    summary = pd.read_csv(paths.results / ROOT / "cell_type_summary.csv")
    assert "no_genes_after_filter" in summary.loc[0, "reason"]


def test_filter_design_is_the_complete_fixed_model_matrix_including_the_intercept(scenarios):
    _, paths, _ = scenarios["interaction"]
    cell = paths.results / ROOT / "a"
    filt = pd.read_csv(cell / "r_output/filter_design.csv", index_col=0)
    meta = pd.read_csv(cell / "pseudobulk_meta.csv", index_col=0)
    fixed = [c for c in meta.columns if c.startswith("cq_fx")]
    assert list(filt.columns) == ["(Intercept)", *fixed]
    assert (filt["(Intercept)"] == 1).all()
    # Same rows as the model's fixed part, in the same order.
    np.testing.assert_array_equal(filt[fixed].to_numpy(), meta.loc[filt.index, fixed].to_numpy())
    assert np.linalg.matrix_rank(filt.to_numpy()) == filt.shape[1] == 4
    prov = json.loads((cell / "model_provenance.json").read_text())
    assert prov["filter_design_columns"] == ["(Intercept)", *fixed]
    assert prov["r"]["filter_design_rank"] == 4


def test_serial_and_parallel_execution_agree(tmp_path_factory):
    import time

    adata = make_adata(effect="interaction", magnitude=MAG)
    out = {}
    for n_jobs in (1, 4):
        paths = PipelinePaths.from_output_dir(tmp_path_factory.mktemp(f"j{n_jobs}"))
        paths.ensure_directories()
        ctx = PipelineContext(
            config=make_config(n_jobs=n_jobs),
            paths=paths,
            adata=adata,
            backend_registry=build_default_backend_registry(),
        )
        t0 = time.perf_counter()
        res = DifferentialExpressionStage().run(ctx)
        wall = time.perf_counter() - t0
        prov = json.loads((paths.results / ROOT / "a/model_provenance.json").read_text())
        out[n_jobs] = (_table(res), prov, wall)
    print(
        {
            k: (v[1]["parallel"], round(v[1]["r"]["fit_seconds"], 1), round(v[2], 1))
            for k, v in out.items()
        }
    )
    (ser, ps, _), (par, pp, _) = out[1], out[4]
    assert ps["parallel"]["class"] == "SerialParam" and ps["parallel"]["workers"] == 1
    assert pp["parallel"]["class"] == "MulticoreParam" and pp["parallel"]["workers"] == 4
    keys = ["gene", "contrast"]
    a = ser.sort_values(keys).reset_index(drop=True)
    b = par.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys])
    for col in ("logFC", "t", "PValue", "FDR", "df_residual", "standard_error", "AveExpr"):
        np.testing.assert_allclose(a[col], b[col], rtol=1e-8, atol=1e-10, err_msg=col)


def test_fixed_effect_single_site_fit_recovers_the_arm_effect_and_matches_a_hand_ols(tmp_path):
    """No random effect (one specimen per subject): an ordinary 3 v 3 voom-limma fit."""

    adata = make_adata(effect="arm", magnitude=MAG)
    adata = adata[adata.obs["site"] == "colon"].copy()
    cfg = make_config(
        factorial={
            "subject_col": "subject",
            "sample_col": "specimen",
            "factors": {"arm": {"reference": "ctl", "scope": "between_subject"}},
            "interactions": [],
            "random_effects": [],
        },
        contrasts=[
            {"kind": "simple_effect", "name": "trt_vs_ctl", "factor": "arm", "numerator": "trt",
             "denominator": "ctl"}
        ],
    )  # fmt: skip
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    ctx = PipelineContext(
        config=cfg, paths=paths, adata=adata, backend_registry=build_default_backend_registry()
    )
    res = DifferentialExpressionStage().run(ctx)
    table = _table(res)
    assert _median_lfc(table, "trt_vs_ctl") == pytest.approx(LOG2_MAG, abs=0.4)
    assert abs(_median_lfc(table, "trt_vs_ctl", genes="null")) < 0.15
    # 6 specimens, one intercept + one arm coefficient: 4 residual degrees of freedom.
    t = table[(table["contrast"] == "trt_vs_ctl") & (table["cell_type"] == "A")]
    assert (t["df_residual"].round(6) == 4).all()
    # Independent oracle: difference of arm means of the voom-free logCPM on the same pseudobulk.
    counts = pd.read_csv(paths.results / ROOT / "a/pseudobulk_counts.csv", index_col=0)
    logcpm = np.log2((counts + 0.5).div(counts.sum(axis=1) + 1, axis=0) * 1e6)
    trt = [s for s in logcpm.index if SUBJECTS[s[:2]] == "trt"]
    ctl = [s for s in logcpm.index if SUBJECTS[s[:2]] == "ctl"]
    oracle = logcpm.loc[trt].mean() - logcpm.loc[ctl].mean()
    t = t.set_index("gene")
    genes = [f"g{i}" for i in range(N_GENES)[EFFECT_GENES]]
    assert np.corrcoef(t.loc[genes, "logFC"], oracle[genes])[0, 1] > 0.8
