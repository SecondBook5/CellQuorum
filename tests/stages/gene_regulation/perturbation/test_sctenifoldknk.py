"""Gene-KO configuration, cohort selection and native-result lifecycle checks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy import sparse

from cellquorum.backends.base import BackendStatus
from cellquorum.backends.sctenifoldknk_backend import ScTenifoldKnkBackend
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.gene_regulation.perturbation.config import (
    PerturbationConfig,
    ScTenifoldKnkConfig,
)
from cellquorum.stages.gene_regulation.perturbation.sctenifoldknk_method import (
    ScTenifoldKnkMethod,
    pathway_enrichment,
    prepare_matrix,
    select_cells,
)


def counts() -> ad.AnnData:
    rng = np.random.default_rng(5)
    x = rng.poisson(1.5, (120, 25)).astype(float)
    a = ad.AnnData(
        sparse.csr_matrix(x),
        obs=pd.DataFrame(
            {
                "condition": ["LE"] * 60 + ["Normal"] * 60,
                "state": ["Basal"] * 30 + ["Spinous"] * 30 + ["Basal"] * 30 + ["Spinous"] * 30,
                "donor": ["P1", "P2", "P3"] * 40,
            },
            index=[f"c{i}" for i in range(120)],
        ),
        var=pd.DataFrame(index=["IL33"] + [f"G{i}" for i in range(24)]),
    )
    a.layers["counts"] = a.X.copy()
    return a


def config() -> dict:
    return {
        "method": "sctenifoldknk",
        "condition_key": "condition",
        "cluster_key": "state",
        "sctenifoldknk": {
            "ko_genes": ["IL33"],
            "condition_label": "LE",
            "min_cells": 10,
            "max_cells": 20,
            "max_genes": 15,
            "donor_key": "donor",
            "figure_formats": [],
        },
    }


def context(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(
            results=tmp_path / "results", scratch=tmp_path / "scratch", figures=tmp_path / "figures"
        )
    )


def test_registration_and_default_unchanged() -> None:
    assert METHOD_REGISTRY.has("perturbation", "sctenifoldknk")
    assert METHOD_REGISTRY.has("perturbation", "celloracle")
    assert PerturbationConfig().method == "celloracle"
    assert PerturbationConfig(**config()).sctenifoldknk.ko_genes == ["IL33"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ko_genes": ["IL33", "IL33"]},
        {"ko_genes": [""]},
        {"n_nets": 0},
        {"by_cluster": False, "pooled": False},
        {"min_pathway_genes": 10, "max_pathway_genes": 5},
        {"ko_method": "fabricated"},
        {"unexpected": 1},
    ],
)
def test_invalid_method_options(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        ScTenifoldKnkConfig(**kwargs)


def test_explicit_intervention_and_condition_required() -> None:
    with pytest.raises(ValidationError, match="ko_genes"):
        PerturbationConfig(method="sctenifoldknk")
    cfg = config()
    cfg["sctenifoldknk"].pop("condition_label")
    with pytest.raises(ValidationError, match="condition"):
        PerturbationConfig(**cfg)


def test_sampling_reproducible_and_keeps_small_donor_strata() -> None:
    obs = pd.DataFrame({"donor": ["large"] * 95 + ["small"] * 5})
    selected = select_cells(obs, 20, 3, ["donor"])
    assert len(selected) == 20
    assert len(set(obs.iloc[selected].donor)) == 2
    np.testing.assert_array_equal(selected, select_cells(obs, 20, 3, ["donor"]))
    with pytest.raises(ValueError, match="strata"):
        select_cells(obs, 1, 3, ["donor"])


def test_sparse_matrix_retains_knockout_without_modifying_input() -> None:
    a = counts()
    before = a.layers["counts"].copy()
    x = prepare_matrix(a, "counts", ScTenifoldKnkConfig(ko_genes=["IL33"], max_genes=10))
    assert "IL33" in x.index and x.shape == (10, 120)
    assert (a.layers["counts"] != before).nnz == 0
    assert (x.to_numpy() == np.rint(x.to_numpy())).all()


def test_unidentifiable_knockout_is_not_forced_into_model() -> None:
    a = counts()
    a.layers["counts"] = a.layers["counts"].toarray()
    a.layers["counts"][:, 0] = 0
    with pytest.raises(ValueError, match="expression or variation"):
        prepare_matrix(a, "counts", ScTenifoldKnkConfig(ko_genes=["IL33"]))


def test_normalized_counts_rejected() -> None:
    a = counts()
    a.layers["counts"] = a.layers["counts"].astype(float) / 3
    with pytest.raises(ValueError, match="integer counts"):
        prepare_matrix(a, "counts", ScTenifoldKnkConfig(ko_genes=["IL33"]))


def test_pathway_background_and_zero_hit_terms(tmp_path: Path) -> None:
    ranking = pd.DataFrame({"gene": ["IL33", "A", "B", "C", "D"], "FDR": [0, 0.01, 0.02, 1, 1]})
    gmt = tmp_path / "sets.gmt"
    gmt.write_text("hits\tdescription\tIL33\tA\tB\nzero\tdescription\tC\tD\n")
    cfg = ScTenifoldKnkConfig(ko_genes=["IL33"], min_pathway_genes=1)
    result = pathway_enrichment(ranking, gmt, cfg).set_index("pathway")
    assert result.loc["hits", "n_background"] == 4
    assert result.loc["hits", "n_overlap"] == 2
    assert result.loc["zero", "p"] == 1
    assert "IL33" not in result.loc["hits", "genes"]


def fake_backend(
    monkeypatch: pytest.MonkeyPatch, *, fail: bool = False, no_edges: bool = False
) -> None:
    monkeypatch.setattr(
        ScTenifoldKnkBackend,
        "status",
        lambda self: BackendStatus("sctenifoldknk", "external", True),
    )

    def run(self, args: list[str], *, timeout: int) -> SimpleNamespace:
        if fail:
            return SimpleNamespace(returncode=1, stdout="", stderr="native failure")
        output = Path(args[args.index("--output") + 1])
        matrix = pd.read_csv(args[args.index("--input") + 1], index_col=0)
        pd.DataFrame(
            {
                "Gene": matrix.index,
                "Distance": np.linspace(0, 1, len(matrix)),
                "p-value": np.linspace(0.01, 1, len(matrix)),
                "adjusted p-value": np.linspace(0.1, 1, len(matrix)),
            }
        ).to_csv(output, index=False)
        output.with_suffix(".metadata.json").write_text(
            json.dumps({"version": "test", "n_knockout_outgoing_edges": 0 if no_edges else 3})
        )
        pd.DataFrame({"source": ["IL33"], "target": ["G0"], "weight": [1]}).to_csv(
            output.with_suffix(".outgoing_edges.csv"), index=False
        )
        return SimpleNamespace(returncode=0, stdout="native workflow", stderr="")

    monkeypatch.setattr(ScTenifoldKnkBackend, "run_script", run)


def test_condition_filter_and_cohort_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_backend(monkeypatch)
    result = ScTenifoldKnkMethod()._run(counts(), config(), context(tmp_path))
    assert result.status == "success" and result.metrics["n_cohorts"] == 3
    for path in (tmp_path / "results/sctenifoldknk").glob("*_cells.csv"):
        membership = pd.read_csv(path)
        assert set(membership.condition) == {"LE"}
        assert len(membership) == 20
        assert membership.donor.nunique() == 3
    assert all(a.path.exists() for a in result.artifacts)
    assert any("No pathway GMT" in w for w in result.warnings)


def test_failed_native_run_is_failure_not_successful_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_backend(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="failed"):
        ScTenifoldKnkMethod()._run(counts(), config(), context(tmp_path))


def test_no_outgoing_edges_preserves_audit_but_no_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_backend(monkeypatch, no_edges=True)
    root = tmp_path / "results/sctenifoldknk"
    root.mkdir(parents=True)
    (root / "differential_regulation.csv").write_text("stale successful result")
    result = ScTenifoldKnkMethod()._run(counts(), config(), context(tmp_path))
    assert result.status == "skipped"
    assert any(a.name == "knk_cohort_audit" for a in result.artifacts)
    assert not (tmp_path / "results/sctenifoldknk/differential_regulation.csv").exists()


def test_missing_runtime_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ScTenifoldKnkBackend,
        "status",
        lambda self: BackendStatus("sctenifoldknk", "external", False),
    )
    result = ScTenifoldKnkMethod()._run(counts(), config(), context(tmp_path))
    assert isinstance(result, MethodSkip)


def test_unknown_subtype_rejected(tmp_path: Path) -> None:
    cfg = config()
    cfg["sctenifoldknk"]["groups"] = ["does_not_exist"]
    with pytest.raises(ValueError, match="groups absent"):
        ScTenifoldKnkMethod()._run(counts(), cfg, context(tmp_path))
