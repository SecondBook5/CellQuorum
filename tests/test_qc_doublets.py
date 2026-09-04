"""Tests for doublet detection + consensus."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from _external_data import r_package_available

from cellquorum.stages.qc import doublets as dbl
from cellquorum.stages.qc.config import QCDoubletConfig
from cellquorum.stages.qc.doublets import detect_doublets, run_scdblfinder


def _counts_adata(seed=0, n=200, g=500):
    rng = np.random.default_rng(seed)
    x = rng.poisson(1.0, size=(n, g)).astype(np.float32)
    a = ad.AnnData(X=x)
    a.layers["counts"] = x.copy()
    return a


@dataclass
class _RecordingBackend:
    """Stands in for ``RscriptBackend``, recording argv and writing a plausible result.

    The point of these tests is the SHAPE of the call — how many R processes the
    stage launches and what it hands them — which is the whole cost of the step
    (R startup plus ``library(scDblFinder)`` is ~5s, dwarfing the model fit on a
    few hundred cells). Asserting that against a fake backend keeps the check in
    the fast tier; the real-R test below covers what R does with the arguments.
    """

    calls: list[list[str]] = field(default_factory=list)
    #: The sample column each call was handed, or None when it was handed none.
    #: Captured HERE rather than read back from ``calls``: the adapter writes its
    #: inputs into a TemporaryDirectory that is gone by the time a test looks.
    samples_seen: list[list[str] | None] = field(default_factory=list)

    def run_script(self, script: Path, argv: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        self.samples_seen.append(
            pd.read_csv(argv[3])["sample"].astype(str).tolist() if len(argv) >= 4 else None
        )
        mtx_path, out_path = Path(argv[0]), Path(argv[1])
        # Cell count from the Matrix Market header, so the fake result is the
        # right length without the fake needing to know the caller's object.
        with open(mtx_path, encoding="utf-8") as handle:
            dims = [line for line in handle if not line.startswith("%")][0].split()
        n_cells = int(dims[1])
        pd.DataFrame({"score": np.zeros(n_cells), "class": ["singlet"] * n_cells}).to_csv(
            out_path, index=False
        )
        return subprocess.CompletedProcess(args=["Rscript"], returncode=0)


def test_scrublet_consensus_writes_calls():
    a = _counts_adata()
    cfg = QCDoubletConfig(
        enabled=True,
        methods=["scrublet"],
        consensus="any",
        remove=False,
        expected_doublet_rate=0.06,
    )
    metrics = detect_doublets(a, cfg, backend=None)
    assert "doublet_score" in a.obs
    assert "predicted_doublet" in a.obs
    assert "scrublet" in metrics["methods_run"]
    # No cells removed here (flag-only).
    assert a.n_obs == 200


def test_per_sample_detection_runs_per_library():
    """With per_sample + a sample_key, detection runs per library, not pooled."""
    import pandas as pd

    a = _counts_adata(n=120, g=400)
    # Two libraries; per-sample detection should score each independently.
    a.obs["sample_id"] = pd.Categorical(["libA"] * 60 + ["libB"] * 60)
    cfg = QCDoubletConfig(
        enabled=True,
        methods=["scrublet"],
        consensus="any",
        per_sample=True,
        expected_doublet_rate=0.06,
    )

    metrics = detect_doublets(a, cfg, backend=None, sample_key="sample_id")

    assert metrics["scored_scope"] == "per_sample"
    assert metrics["sample_key"] == "sample_id"
    assert "doublet_score" in a.obs
    # Every cell was scored by some library-local run.
    assert a.obs["doublet_score"].notna().all()


def test_scdblfinder_per_sample_launches_one_r_process_for_the_whole_cohort():
    """Per-sample scDblFinder splits inside R, not by one subprocess per sample.

    This is a wall-clock regression test with a real number behind it: the LEC arm
    of the mechanotransduction run has 18 captures, and driving the split from
    Python spent 136s of a 150s QC stage, ~95s of which was loading the same three
    R libraries eighteen times. scDblFinder takes the sample assignment itself via
    ``samples=``, with the same per-capture semantics.
    """

    a = _counts_adata(n=180, g=300)
    a.obs["sample_id"] = pd.Categorical(["libA"] * 60 + ["libB"] * 60 + ["libC"] * 60)
    backend = _RecordingBackend()
    cfg = QCDoubletConfig(enabled=True, methods=["scdblfinder"], consensus="any", per_sample=True)

    metrics = detect_doublets(a, cfg, backend=backend, sample_key="sample_id")

    assert metrics["scored_scope"] == "per_sample"
    assert len(backend.calls) == 1, (
        f"scDblFinder was launched {len(backend.calls)} times for 3 samples; "
        "the per-capture split belongs inside the one R session"
    )
    # And the samples file must actually have been passed, or "one call" would
    # silently mean "pooled" -- doublets searched across captures, which they
    # cannot form across.
    assert (
        backend.samples_seen[0] is not None
    ), f"no samples file passed to scDblFinder: {backend.calls[0]}"
    assert backend.samples_seen[0] == a.obs["sample_id"].astype(str).tolist()


def test_scdblfinder_pooled_passes_no_samples_file():
    """Pooled mode must not hand R a sample column; pooled means one capture."""

    a = _counts_adata(n=60, g=200)
    a.obs["sample_id"] = pd.Categorical(["libA"] * 30 + ["libB"] * 30)
    backend = _RecordingBackend()
    cfg = QCDoubletConfig(enabled=True, methods=["scdblfinder"], consensus="any", per_sample=False)

    metrics = detect_doublets(a, cfg, backend=backend, sample_key="sample_id")

    assert metrics["scored_scope"] == "pooled"
    assert len(backend.calls) == 1
    assert backend.samples_seen == [None], f"unexpected samples argument: {backend.calls[0]}"


def test_scrublet_per_sample_still_loops_in_python():
    """Only detectors in ``_NATIVE_PER_SAMPLE`` skip the Python-side split.

    Scrublet has no ``samples=`` equivalent, so handing it the pooled object would
    quietly change per-capture detection into pooled detection. Counted by how many
    distinct cell counts the detector was shown.
    """

    a = _counts_adata(n=120, g=300)
    a.obs["sample_id"] = pd.Categorical(["libA"] * 40 + ["libB"] * 40 + ["libC"] * 40)
    seen: list[int] = []

    def fake_scrublet(adata, *, expected_rate, random_state):
        seen.append(adata.n_obs)
        return np.zeros(adata.n_obs), None

    cfg = QCDoubletConfig(enabled=True, methods=["scrublet"], consensus="any", per_sample=True)
    original = dbl.run_scrublet
    dbl.run_scrublet = fake_scrublet
    try:
        detect_doublets(a, cfg, backend=None, sample_key="sample_id")
    finally:
        dbl.run_scrublet = original

    assert seen == [40, 40, 40], f"scrublet saw {seen}, expected one call per capture"


@pytest.mark.skipif(
    not r_package_available("scDblFinder"),
    reason="scDblFinder is not installed in this R library",
)
def test_scdblfinder_real_r_returns_scores_in_input_cell_order():
    """The R adapter's scores line up with the cells that were sent.

    With ``samples=``, scDblFinder splits the object, scores each part and rebinds,
    so the column order of what comes back is its business. The adapter assigns the
    returned rows to cells POSITIONALLY, so a reordered rebind would attach one
    donor's doublet calls to another donor's cells and nothing would complain.

    Made detectable by planting the doublets: cells 80-119 are literal sums of a
    type-A and a type-B cell from the same capture, and everything else is a
    singlet. Read on the SCORE, not the class call -- scDblFinder thresholds
    against an expected doublet rate of ~1% per thousand cells, so on 240 cells it
    correctly calls almost nothing whatever the data looks like.

    The capture labels sort in the OPPOSITE order to their position, so grouping or
    sorting by label inside R moves the planted block and the comparison collapses.
    """

    from cellquorum.backends.rscript import RscriptBackend

    rng = np.random.default_rng(0)

    # Two cell types per capture, so there is real structure for the kNN step.
    def _capture(n: int) -> np.ndarray:
        rates = np.concatenate([np.full(200, 3.0), np.full(200, 0.1)])
        first = rng.poisson(rates, size=(n // 2, 400))
        second = rng.poisson(rates[::-1], size=(n - n // 2, 400))
        return np.vstack([first, second]).astype(np.float32)

    spiked = _capture(80)
    # 40 planted doublets: one cell of each type summed together.
    planted = spiked[:40] + spiked[40:80]
    clean = _capture(120)

    a = ad.AnnData(X=np.vstack([spiked, planted, clean]))
    a.layers["counts"] = a.X.copy()
    # "zz" first, "aa" second: sorting by label reverses the blocks.
    a.obs["sample_id"] = pd.Categorical(["zz"] * 120 + ["aa"] * 120)

    scores, calls = run_scdblfinder(a, RscriptBackend(), random_state=0, sample_key="sample_id")

    assert scores.shape == (240,)
    assert not np.isnan(scores).any(), "scDblFinder left cells unscored"
    assert calls is not None and calls.shape == (240,)
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
    planted = float(scores[80:120].mean())
    singlets = float(scores[:80].mean())
    elsewhere = float(scores[120:].mean())
    assert planted > singlets and planted > elsewhere, (
        f"planted doublets (rows 80-119) mean score {planted:.3f} vs {singlets:.3f} "
        f"for the singlets beside them and {elsewhere:.3f} for the clean capture: "
        "the returned rows are not in input cell order"
    )


def test_per_sample_falls_back_to_pooled_without_sample_key():
    """per_sample=True but no sample_key resolves to pooled detection."""
    a = _counts_adata(n=100, g=300)
    cfg = QCDoubletConfig(enabled=True, methods=["scrublet"], consensus="any", per_sample=True)

    metrics = detect_doublets(a, cfg, backend=None, sample_key=None)

    assert metrics["scored_scope"] == "pooled"
    assert metrics["sample_key"] is None


def test_score_threshold_flags_at_ceiling(monkeypatch):
    """A score AT the default 0.5 threshold must flag (regression for `> 0.5`).

    The historical bug used ``scores > 0.5`` while observed scores ceiling at
    exactly 0.5, so no cell was ever flagged. The fix uses ``>=``.
    """

    a = _counts_adata(n=5)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    # No native call → the score-threshold fallback path is exercised.
    monkeypatch.setattr(
        dbl, "run_scrublet", lambda adata, *, expected_rate, random_state: (scores, None)
    )

    cfg = QCDoubletConfig(enabled=True, methods=["scrublet"], consensus="any", per_sample=False)
    metrics = detect_doublets(a, cfg, backend=None)

    assert a.obs["predicted_doublet"].to_numpy().tolist() == [False, False, False, False, True]
    assert metrics["used_native_calls"] == {"scrublet": False}
    assert int(metrics["n_predicted_doublets"]) == 1


def test_native_calls_take_precedence(monkeypatch):
    """The detector's own call is used, not a re-threshold of the score."""

    a = _counts_adata(n=4)
    # Every cell scores "high", but the detector's native call flags only 0 and 2.
    scores = np.array([0.9, 0.9, 0.1, 0.1], dtype=float)
    native = np.array([True, False, True, False])
    monkeypatch.setattr(
        dbl, "run_scrublet", lambda adata, *, expected_rate, random_state: (scores, native)
    )

    cfg = QCDoubletConfig(enabled=True, methods=["scrublet"], consensus="any", per_sample=False)
    metrics = detect_doublets(a, cfg, backend=None)

    assert a.obs["predicted_doublet"].to_numpy().tolist() == [True, False, True, False]
    assert metrics["used_native_calls"] == {"scrublet": True}


def test_zero_flagged_warns_loudly(monkeypatch, caplog):
    """A detector that scored cells but flagged none must warn (no-silent-decisions)."""

    a = _counts_adata(n=4)
    # All below the 0.5 threshold, no native call → zero flagged.
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    monkeypatch.setattr(
        dbl, "run_scrublet", lambda adata, *, expected_rate, random_state: (scores, None)
    )

    cfg = QCDoubletConfig(enabled=True, methods=["scrublet"], consensus="any", per_sample=False)
    with caplog.at_level("WARNING"):
        metrics = detect_doublets(a, cfg, backend=None)

    assert int(metrics["n_predicted_doublets"]) == 0
    assert any("flagged 0 doublets" in record.message for record in caplog.records)


def test_consensus_any_vs_all_semantics():
    # Two synthetic per-method call columns combined by rule.
    import pandas as pd

    from cellquorum.stages.qc.doublets import combine_consensus

    calls = pd.DataFrame({"m1": [True, True, False], "m2": [True, False, False]})
    assert list(combine_consensus(calls, "any")) == [True, True, False]
    assert list(combine_consensus(calls, "all")) == [True, False, False]
    assert list(combine_consensus(calls, "majority")) == [True, False, False]
