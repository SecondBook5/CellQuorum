"""Tests for the state-scoring stage (score_genes + aucell methods + dispatch)."""

from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.contracts import set_layer_tag
from cellquorum.methods.base import MethodSkip
from cellquorum.stages.state_scoring.aucell_method import AucellMethod
from cellquorum.stages.state_scoring.programs import STATE_PROGRAMS, read_gmt, resolve_programs
from cellquorum.stages.state_scoring.score_genes_method import ScoreGenesMethod
from cellquorum.stages.state_scoring.stage import StateScoringStage

LAYER = "cellquorum_normalized"

# Two synthetic programs whose genes we put in var_names, so eligibility is
# deterministic and independent of the curated STATE_PROGRAMS contents.
PROG_A = ["A1", "A2", "A3", "A4"]
PROG_B = ["B1", "B2", "B3", "B4"]
PROGRAMS = {"prog_a": PROG_A, "prog_b": PROG_B}


class _Paths:
    def __init__(self, tmp):
        self.root = tmp
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    """Dict-config context: resolve_stage_config reads config['state_scoring']."""

    def __init__(self, tmp, adata, stage_config):
        self.config = {"state_scoring": stage_config}
        self.paths = _Paths(tmp)
        self.adata = adata

    def require_adata(self):
        return self.adata


def _adata(n_cells: int = 40) -> ad.AnnData:
    """Log-normalized synthetic data: 8 program genes + 32 filler genes.

    The filler genes give ``sc.tl.score_genes`` a control pool to sample from;
    values are small positive floats so the lognorm contract (not-all-integer +
    log-range) passes.
    """
    rng = np.random.default_rng(0)
    filler = [f"F{i}" for i in range(32)]
    var_names = PROG_A + PROG_B + filler
    n_genes = len(var_names)
    x = rng.random(size=(n_cells, n_genes)).astype(float) * 4.0
    obs = pd.DataFrame(
        {"cell_type": (["T0"] * (n_cells // 2)) + (["T1"] * (n_cells - n_cells // 2))},
        index=[f"c{i}" for i in range(n_cells)],
    )
    a = ad.AnnData(X=x, obs=obs)
    a.var_names = var_names
    a.layers[LAYER] = x.copy()
    set_layer_tag(a, LAYER, kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    return a


def _config(**overrides) -> dict:
    base = {
        "use_builtin_programs": False,
        "programs": PROGRAMS,
        "layer": LAYER,
        "cell_type_col": "cell_type",
        "min_program_genes": 3,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# programs.py helpers
# --------------------------------------------------------------------------- #


def test_resolve_programs_curated_defaults():
    programs = resolve_programs({"use_builtin_programs": True})
    assert set(programs) == set(STATE_PROGRAMS)
    # Copies, not references, so callers cannot mutate the module-level defaults.
    assert programs["stress_hsp"] is not STATE_PROGRAMS["stress_hsp"]


def test_resolve_programs_user_overrides_and_subset():
    # builtin subset + a user program that overrides a curated name.
    programs = resolve_programs(
        {
            "use_builtin_programs": True,
            "builtin_programs": ["hypoxia_hif"],
            "programs": {"hypoxia_hif": ["X1", "X2"], "custom": ["Y1"]},
        }
    )
    assert set(programs) == {"hypoxia_hif", "custom"}
    # User program wins the name collision.
    assert programs["hypoxia_hif"] == ["X1", "X2"]


def test_read_gmt_drops_description_and_dedups(tmp_path):
    gmt = tmp_path / "sets.gmt"
    gmt.write_text("setA\tdesc\tG1\tG2\tG2\tG3\nsetB\thttp://x\tG4\tG5\n", encoding="utf-8")
    sets = read_gmt(str(gmt))
    assert sets == {"setA": ["G1", "G2", "G3"], "setB": ["G4", "G5"]}


# --------------------------------------------------------------------------- #
# ScoreGenesMethod (no decoupler needed)
# --------------------------------------------------------------------------- #


def test_score_genes_writes_obs_columns_and_table(tmp_path):
    a = _adata()
    cfg = _config()
    out = ScoreGenesMethod()._run(a, cfg, _Ctx(tmp_path, a, cfg))
    assert not isinstance(out, MethodSkip)
    # One obs column per program (default key_prefix 'state_').
    assert "state_prog_a" in a.obs.columns
    assert "state_prog_b" in a.obs.columns
    assert out.metrics["n_programs"] == 2
    # Per-cell-type mean-score table written with the expected columns.
    df = pd.read_csv(tmp_path / "results" / "state_scoring_score_genes_by_celltype.csv")
    assert list(df.columns) == ["cell_type", "program", "mean_score"]
    assert set(df["cell_type"]) == {"T0", "T1"}
    assert set(df["program"]) == {"prog_a", "prog_b"}


def test_score_genes_skips_when_no_program_meets_gate(tmp_path):
    a = _adata()
    # A program whose genes are absent from var_names → below the present-gene gate.
    cfg = _config(programs={"ghost": ["ZZZ1", "ZZZ2", "ZZZ3"]})
    out = ScoreGenesMethod()._run(a, cfg, _Ctx(tmp_path, a, cfg))
    assert isinstance(out, MethodSkip)
    assert "gate" in out.reason.lower()


# --------------------------------------------------------------------------- #
# AucellMethod (requires real decoupler for the happy path)
# --------------------------------------------------------------------------- #


def test_aucell_writes_obsm_matrix(tmp_path):
    pytest.importorskip("decoupler")
    a = _adata()
    cfg = _config()
    out = AucellMethod()._run(a, cfg, _Ctx(tmp_path, a, cfg))
    assert not isinstance(out, MethodSkip)
    assert "X_state_aucell" in a.obsm
    # cells x programs, one column per eligible program.
    assert a.obsm["X_state_aucell"].shape == (a.n_obs, 2)
    assert set(a.uns["state_aucell"]["programs"]) == {"prog_a", "prog_b"}
    df = pd.read_csv(tmp_path / "results" / "state_scoring_aucell_by_celltype.csv")
    assert list(df.columns) == ["cell_type", "program", "mean_auc"]


def test_aucell_records_the_genes_each_program_was_actually_scored_on(tmp_path):
    """The manifest's list is a request; ``uns`` has to record what was granted.

    Two downstream questions need it. Whether two program scores are independent
    readouts or the same genes read twice is a property of the gene lists, and every
    stage after this one sees only the score matrix — so the correlation table cannot
    disclose a shared-gene count without this. And a module that lost half its genes
    to the detection filter is not the module the manifest names, which is invisible
    from the AUC alone.
    """
    pytest.importorskip("decoupler")
    a = _adata()
    # prog_a asks for two genes that are not in var_names; prog_b is fully present.
    cfg = _config(programs={"prog_a": [*PROG_A, "GHOST1", "GHOST2"], "prog_b": PROG_B})
    out = AucellMethod()._run(a, cfg, _Ctx(tmp_path, a, cfg))
    assert not isinstance(out, MethodSkip)

    genes = a.uns["state_aucell"]["genes"]
    # Requested-and-present, not requested: a name recorded here that was never
    # scored would let a downstream overlap check describe genes nothing read.
    assert genes["prog_a"] == PROG_A
    assert genes["prog_b"] == PROG_B
    # Keyed on the surviving score columns, so the lists and the matrix agree.
    assert set(genes) == set(a.uns["state_aucell"]["programs"])


def test_aucell_skips_when_decoupler_absent(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "decoupler", None)
    a = _adata()
    cfg = _config()
    out = AucellMethod()._run(a, cfg, _Ctx(tmp_path, a, cfg))
    assert isinstance(out, MethodSkip)
    assert "decoupler" in out.reason.lower()


# --------------------------------------------------------------------------- #
# StateScoringStage dispatch
# --------------------------------------------------------------------------- #


def test_stage_defaults_to_both_methods(tmp_path):
    pytest.importorskip("decoupler")
    a = _adata()
    # No 'method' and no 'methods' → _augment_config injects both.
    ctx = _Ctx(tmp_path, a, _config())
    result = StateScoringStage().run(ctx)
    assert result.metrics["n_methods"] == 2
    method_names = {m["method"] for m in result.metrics["per_method"]}
    assert method_names == {"score_genes", "aucell"}
    # Both outputs landed on the same object.
    assert "state_prog_a" in a.obs.columns
    assert "X_state_aucell" in a.obsm


def test_stage_runs_explicit_single_method_without_decoupler(tmp_path):
    a = _adata()
    # An explicit methods list keeps the run decoupler-independent.
    ctx = _Ctx(tmp_path, a, _config(methods=[{"method": "score_genes"}]))
    result = StateScoringStage().run(ctx)
    assert result.metrics["n_methods"] == 1
    assert result.metrics["per_method"][0]["method"] == "score_genes"
    assert "state_prog_a" in a.obs.columns


def test_stage_disabled_returns_recorded_skip(tmp_path):
    a = _adata()
    ctx = _Ctx(tmp_path, a, _config(enabled=False))
    result = StateScoringStage().run(ctx)
    assert result.status == "skipped"
    assert "state_prog_a" not in a.obs.columns
