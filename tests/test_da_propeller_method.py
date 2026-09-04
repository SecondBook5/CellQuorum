# tests/test_da_propeller_method.py
import shutil
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.methods.base import MethodSkip
from cellquorum.stages.comparative.differential_abundance.propeller_method import (
    PropellerMethod,
    _pairing_requested,
    _resolve_pairing,
)


def _propeller_available() -> bool:
    """Check if Rscript and speckle package are available."""
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status=ifelse(requireNamespace('speckle', quietly=TRUE),0,1))",
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _cohort_adata():
    """Build a small cohort with ≥3 donors/arm, one cell type enriched in case."""
    rng = np.random.default_rng(42)
    n_genes = 20
    donors = ["d1", "d2", "d3", "d4", "d5", "d6"]
    blocks, obs_rows = [], []

    for i, donor in enumerate(donors):
        # First 3 donors are control, last 3 are case
        condition = "Normal" if i < 3 else "Disease"

        # TypeA: enriched in Disease (more cells in case)
        n_typeA = 5 if condition == "Normal" else 20
        for _ in range(n_typeA):
            counts = rng.poisson(5, size=n_genes).astype(float)
            blocks.append(counts)
            obs_rows.append({"patient_id": donor, "condition": condition, "cell_type": "TypeA"})

        # TypeB: balanced
        for _ in range(10):
            counts = rng.poisson(5, size=n_genes).astype(float)
            blocks.append(counts)
            obs_rows.append({"patient_id": donor, "condition": condition, "cell_type": "TypeB"})

    X = sp.csr_matrix(np.vstack(blocks))
    obs = pd.DataFrame(obs_rows)
    a = ad.AnnData(X=X, obs=obs)
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"G{i}" for i in range(n_genes)]
    set_layer_tag(a, "counts", kind="counts")
    return a


def _matched_cohort_adata(n_donors: int = 6):
    """A matched cohort whose donor baselines dwarf the within-donor shift.

    Every donor contributes both arms. TypeB's share sweeps from ~6% to ~25% across
    donors while the disease shift inside each donor is a constant, modest increment.
    An arms-only fit divides that increment by the between-donor spread and sees
    nothing; a donor-blocked fit removes the spread and sees it. That contrast is the
    point of the fixture -- it is the specification error, in miniature.
    """
    rng = np.random.default_rng(11)
    n_genes = 20
    blocks, obs_rows = [], []
    for i in range(n_donors):
        donor = f"d{i}"
        for condition in ("Normal", "Disease"):
            n_b = 12 + 18 * i + (22 if condition == "Disease" else 0)
            for cell_type, n_cells in (("TypeA", 120), ("TypeB", n_b), ("TypeC", 80)):
                for _ in range(n_cells):
                    blocks.append(rng.poisson(5, size=n_genes).astype(float))
                    obs_rows.append(
                        {"patient_id": donor, "condition": condition, "cell_type": cell_type}
                    )
    a = ad.AnnData(X=sp.csr_matrix(np.vstack(blocks)), obs=pd.DataFrame(obs_rows))
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"G{i}" for i in range(n_genes)]
    set_layer_tag(a, "counts", kind="counts")
    return a


def _sample_meta(donors, conditions) -> pd.DataFrame:
    return pd.DataFrame({"patient_id": donors, "condition": conditions})


class _Paths:
    def __init__(self, tmp_path):
        self.results = tmp_path / "results"
        self.scratch = tmp_path / "scratch"
        self.results.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)


class _Registry:
    def get(self, name):
        from cellquorum.backends.rscript import RscriptBackend

        if name == "rscript":
            return RscriptBackend()
        raise KeyError(name)


class _Ctx:
    def __init__(self, tmp_path):
        self.paths = _Paths(tmp_path)
        self.backend_registry = _Registry()


def test_contract_requires_design_obs():
    """Verify the method's input contract requires design obs columns."""
    method = PropellerMethod()
    contract = method.input_contract(
        {
            "condition_col": "condition",
            "donor_col": "patient_id",
            "cell_type_col": "cell_type",
            "case": "Disease",
            "control": "Normal",
        }
    )
    assert "condition" in contract.required_obs
    assert "patient_id" in contract.required_obs
    assert "cell_type" in contract.required_obs


def test_skips_when_rscript_absent(monkeypatch, tmp_path):
    """Verify the method skips when Rscript is unavailable."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    method = PropellerMethod()
    out = method._run(
        _cohort_adata(),
        {
            "condition_col": "condition",
            "donor_col": "patient_id",
            "cell_type_col": "cell_type",
            "case": "Disease",
            "control": "Normal",
            "transform": "asin",
        },
        _Ctx(tmp_path),
    )
    assert isinstance(out, MethodSkip)
    assert "rscript" in out.reason.lower() or "propeller" in out.reason.lower()


def test_skips_when_case_control_absent(tmp_path):
    """Verify the method skips when case/control labels are not set."""
    method = PropellerMethod()
    out = method._run(
        _cohort_adata(),
        {
            "condition_col": "condition",
            "donor_col": "patient_id",
            "cell_type_col": "cell_type",
            # case and control missing
        },
        _Ctx(tmp_path),
    )
    assert isinstance(out, MethodSkip)
    assert "case" in out.reason.lower() or "control" in out.reason.lower()


def test_skips_when_cell_type_col_absent(tmp_path):
    """Verify the method skips (not crashes) when cell_type column is missing.

    An ineligible stage must record a skip, not raise. This test pins the
    requires_obs() guard behavior so cohorts missing cell-type annotations
    skip cleanly at the executor level.
    """
    # Build an AnnData WITHOUT cell_type column.
    rng = np.random.default_rng(42)
    X = sp.csr_matrix(rng.poisson(5, size=(50, 20)).astype(float))
    obs = pd.DataFrame(
        {
            "patient_id": ["d1"] * 25 + ["d2"] * 25,
            "condition": ["Normal"] * 25 + ["Disease"] * 25,
        }
    )
    a = ad.AnnData(X=X, obs=obs)
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"G{i}" for i in range(20)]
    set_layer_tag(a, "counts", kind="counts")

    # Call the real .run() entrypoint (NOT ._run) so requires_obs is exercised.
    method = PropellerMethod()
    result = method.run(
        a,
        {
            "condition_col": "condition",
            "donor_col": "patient_id",
            "cell_type_col": "cell_type",
            "case": "Disease",
            "control": "Normal",
        },
        _Ctx(tmp_path),
    )

    # Must return a MethodSkip whose reason mentions the missing obs column.
    assert isinstance(result, MethodSkip)
    assert "cell_type" in result.reason or "obs" in result.reason


@pytest.mark.skipif(not _propeller_available(), reason="Rscript+speckle not available")
def test_propeller_detects_enriched_celltype(tmp_path):
    """Verify propeller detects a cell type enriched in case."""
    method = PropellerMethod()
    result = method._run(
        _cohort_adata(),
        {
            "condition_col": "condition",
            "donor_col": "patient_id",
            "cell_type_col": "cell_type",
            "case": "Disease",
            "control": "Normal",
            "transform": "asin",
        },
        _Ctx(tmp_path),
    )
    # A StageResult with a da_results artifact.
    artifact_paths = [a.path for a in result.artifacts if a.name == "da_results"]
    assert artifact_paths, "expected a da_results artifact"
    da = pd.read_csv(artifact_paths[0])

    # Verify the CSV has expected columns.
    assert "cell_type" in da.columns
    assert "PropRatio" in da.columns
    assert "PValue" in da.columns
    assert "FDR" in da.columns

    # TypeA should be detected as enriched (low p-value).
    typeA = da.loc[da["cell_type"] == "TypeA"].iloc[0]
    assert typeA["PValue"] < 0.05, "TypeA should be significantly enriched"

    # Check metrics.
    assert result.metrics["case"] == "Disease"
    assert result.metrics["control"] == "Normal"
    assert result.metrics["transform"] == "asin"
    assert result.metrics["n_samples"] == 6  # 6 donors
    assert result.metrics["n_celltypes"] == 2  # TypeA, TypeB


# --------------------------------------------------------------------------- #
# The donor block: which design was fitted, and whether it was said out loud
# --------------------------------------------------------------------------- #


def test_pairing_is_declined_when_no_donor_spans_both_arms():
    """The default cohort fixture is disjoint, so a blocked design is meaningless.

    A donor that appears in one arm only has a block coefficient that absorbs its
    condition, so blocking on it removes the contrast rather than the nuisance.
    """
    paired, reason, n_blocked = _resolve_pairing(
        _sample_meta(["d1", "d2", "d3", "d4"], ["Normal", "Normal", "Disease", "Disease"]),
        donor_col="patient_id",
        condition_col="condition",
        case="Disease",
        requested=True,
    )
    assert paired is False
    assert "both arms" in reason
    assert n_blocked == 0


def test_single_arm_donors_neither_enable_nor_starve_the_block():
    """The residual df is ``n_spanning - 1``, whatever the single-arm donors do.

    Each single-arm donor contributes one row and one coefficient, so it cancels out of
    the df exactly. That is why two spanning donors is the threshold and why padding a
    cohort with unmatched samples can neither rescue a block nor break one — the
    tempting shortcut of counting donors, or samples, gets both cases wrong.
    """
    two_pairs = _sample_meta(["d1", "d1", "d2", "d2"], ["Normal", "Disease"] * 2)
    ok, reason, n_blocked = _resolve_pairing(
        two_pairs,
        donor_col="patient_id",
        condition_col="condition",
        case="Disease",
        requested=True,
    )
    assert (ok, reason, n_blocked) == (True, "", 2)

    padded = _sample_meta(
        ["d1", "d1", "d2", "d2", "d3", "d4"],
        ["Normal", "Disease", "Normal", "Disease", "Normal", "Disease"],
    )
    still_ok, reason, n_blocked = _resolve_pairing(
        padded,
        donor_col="patient_id",
        condition_col="condition",
        case="Disease",
        requested=True,
    )
    assert still_ok is True
    assert reason == ""
    # All four donors enter the block; only two of them carry the contrast.
    assert n_blocked == 4

    # One spanning donor plus any number of single-arm donors is still one pair.
    starved, reason, _ = _resolve_pairing(
        _sample_meta(
            ["d1", "d1", "d2", "d3", "d4", "d5"],
            ["Normal", "Disease", "Normal", "Normal", "Disease", "Disease"],
        ),
        donor_col="patient_id",
        condition_col="condition",
        case="Disease",
        requested=True,
    )
    assert starved is False
    assert "residual degrees of freedom" in reason


def test_pairing_is_not_attempted_when_the_design_did_not_ask_for_it():
    paired, reason, n_blocked = _resolve_pairing(
        _sample_meta(["d1", "d1", "d2", "d2"], ["Normal", "Disease"] * 2),
        donor_col="patient_id",
        condition_col="condition",
        case="Disease",
        requested=False,
    )
    assert (paired, reason, n_blocked) == (False, "", 0)


def test_the_stage_switch_wins_over_the_declared_design():
    """``pair_by_donor`` and ``paired`` can contradict each other; one of them has to win.

    They arrive from different places -- ``paired`` is bridged from the project design
    block, ``pair_by_donor`` is the DA stage's own switch -- so a config can set
    ``pair_by_donor: never`` on a cohort the design declares matched. The more specific
    instruction wins, and ``auto`` defers to the declaration rather than second-guessing
    it from the data.
    """
    assert _pairing_requested({"paired": True}) is True
    assert _pairing_requested({"paired": False}) is False
    assert _pairing_requested({"paired": True, "pair_by_donor": "never"}) is False
    assert _pairing_requested({"paired": False, "pair_by_donor": "always"}) is True
    assert _pairing_requested({"paired": False, "pair_by_donor": "auto"}) is False
    # Nothing declared at all: block by default, since an unblocked fit on a matched
    # cohort is the error and an unmatched cohort declines on estimability anyway.
    assert _pairing_requested({}) is True


def test_a_missing_donor_column_declines_rather_than_raises():
    paired, reason, _ = _resolve_pairing(
        pd.DataFrame({"condition": ["Normal", "Disease"]}),
        donor_col="patient_id",
        condition_col="condition",
        case="Disease",
        requested=True,
    )
    assert paired is False
    assert "patient_id" in reason


@pytest.mark.skipif(not _propeller_available(), reason="Rscript+speckle not available")
def test_a_matched_cohort_is_fitted_within_donor(tmp_path):
    """The blocked fit finds the shift the arms-only fit on the same data cannot.

    This is the regression guard for a real defect: propeller was wired with an
    arms-only design even when the project design declared the cohort matched, and on
    a nine-donor cohort in this project that turned two FDR-0.03 lineages into a table
    with nothing under 0.39.
    """
    adata = _matched_cohort_adata()
    base = {
        "condition_col": "condition",
        "donor_col": "patient_id",
        "cell_type_col": "cell_type",
        "case": "Disease",
        "control": "Normal",
        "transform": "asin",
    }

    blocked = PropellerMethod()._run(adata, {**base, "paired": True}, _Ctx(tmp_path / "paired"))
    arms_only = PropellerMethod()._run(
        adata, {**base, "paired": False}, _Ctx(tmp_path / "unpaired")
    )

    assert blocked.metrics["paired"] is True
    assert blocked.metrics["n_donors_blocked"] == 6
    assert blocked.metrics["paired_fallback_reason"] == ""
    assert arms_only.metrics["paired"] is False

    def table(result):
        path = [a.path for a in result.artifacts if a.name == "da_results"][0]
        return pd.read_csv(path).set_index("cell_type")

    blocked_da, arms_da = table(blocked), table(arms_only)

    # The design that was fitted is on the table, not only in the metrics.
    assert bool(blocked_da["paired"].iloc[0]) is True
    assert bool(arms_da["paired"].iloc[0]) is False

    # Same effects, different spread: the blocked fit calls TypeB, arms-only does not.
    assert blocked_da.loc["TypeB", "effect_pp"] == pytest.approx(arms_da.loc["TypeB", "effect_pp"])
    assert blocked_da.loc["TypeB", "FDR"] < 0.05
    assert arms_da.loc["TypeB", "FDR"] > 0.05


@pytest.mark.skipif(not _propeller_available(), reason="Rscript+speckle not available")
def test_the_ratio_column_is_a_ratio(tmp_path):
    """``PropRatio`` under a treatment-contrast design is silently a difference.

    speckle builds the column as ``prod(coef ** contrast)``, which is a ratio only when
    the coefficients are the two arm means. Fitting ``~ grp`` instead of ``~ 0 + grp``
    makes the coefficients (control mean, difference) and the same arithmetic returns
    the difference under a column named for a ratio.
    """
    result = PropellerMethod()._run(
        _matched_cohort_adata(),
        {
            "condition_col": "condition",
            "donor_col": "patient_id",
            "cell_type_col": "cell_type",
            "case": "Disease",
            "control": "Normal",
            "transform": "asin",
            "paired": True,
        },
        _Ctx(tmp_path),
    )
    da = pd.read_csv([a.path for a in result.artifacts if a.name == "da_results"][0])
    expected = da["case_mean_prop"] / da["control_mean_prop"]
    assert np.allclose(da["PropRatio"], expected)
    # And the difference is reported under its own name, in percentage points.
    assert np.allclose(da["effect_pp"], (da["case_mean_prop"] - da["control_mean_prop"]) * 100)


@pytest.mark.skipif(not _propeller_available(), reason="Rscript+speckle not available")
def test_the_design_floor_travels_with_the_propeller_fdr(tmp_path):
    """The floor describes the design that was fitted, not the cohort that was collected.

    Six pairs blocked on donor draw on 2**6 assignments; the same twelve samples fitted
    arms-only draw on C(12, 6), which is two orders of magnitude more. Reporting the
    paired floor beside an arms-only fit would overstate how constrained it was.
    """
    from math import comb

    adata = _matched_cohort_adata()
    base = {
        "condition_col": "condition",
        "donor_col": "patient_id",
        "cell_type_col": "cell_type",
        "case": "Disease",
        "control": "Normal",
        "transform": "asin",
    }
    blocked = PropellerMethod()._run(adata, {**base, "paired": True}, _Ctx(tmp_path / "p"))
    arms_only = PropellerMethod()._run(adata, {**base, "paired": False}, _Ctx(tmp_path / "u"))

    blocked_da = pd.read_csv([a.path for a in blocked.artifacts if a.name == "da_results"][0])
    arms_da = pd.read_csv([a.path for a in arms_only.artifacts if a.name == "da_results"][0])

    for column in (
        "design_floor_p",
        "p_below_design_floor",
        "family_size",
        "family_min_concordant",
        "family_floor_reachable",
    ):
        assert column in blocked_da.columns

    assert blocked_da["design_floor_p"].iloc[0] == pytest.approx(2 / 2**6)
    assert arms_da["design_floor_p"].iloc[0] == pytest.approx(2 / comb(12, 6))
    assert blocked_da["family_size"].iloc[0] == 3
    assert blocked.metrics["design_floor_p"] == pytest.approx(2 / 2**6)
