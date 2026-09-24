"""Tests for the in-env CellOracle KO script's pure helpers + skip path.

No real CellOracle in CI — only argument parsing, the ranking-schema writer, and
the graceful-skip writer are exercised.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "cellquorum"
    / "backends"
    / "celloracle_scripts"
    / "celloracle_ko.py"
)


def _load():
    return runpy.run_path(str(SCRIPT))


def test_module_imports_without_celloracle() -> None:
    ns = _load()
    assert "main" in ns
    assert "build_parser" in ns
    assert "write_skip" in ns
    assert "write_ranking" in ns


def test_parser_has_expected_args() -> None:
    ns = _load()
    parser = ns["build_parser"]()
    args = parser.parse_args(["--h5ad", "a.h5ad", "--out-dir", "o", "--tag", "t"])
    assert args.h5ad == "a.h5ad"
    assert args.out_dir == "o"
    assert args.tag == "t"
    assert args.organism == "human"
    assert args.n_top_targets == 20
    # filter_edge_number is the PER-CLUSTER edge cap (CellOracle default 10000),
    # distinct from n_top_targets (the figure top-N) -- must not collapse to 20.
    assert args.filter_edge_number == 10000
    # filter_links' p-value cutoff, not the edge-count cap, is usually the real
    # constraint -- exposed separately so it can be relaxed without touching
    # --filter-edge-number, which turned out to have zero effect in practice.
    assert args.filter_p == pytest.approx(0.001)
    assert args.knn_n_neighbors == 200
    # velocyto's compiled colDeltaCorpartial segfaults on a large real object at
    # the CellOracle default (1.0, i.e. every cell is a candidate neighbor); a
    # real prior run at ~2,000-cell scale completed cleanly, so this must stay
    # overridable to keep the candidate pool near a proven-working size.
    assert args.sampled_fraction == pytest.approx(1.0)
    assert args.n_propagation == 3
    assert args.seed == 0


def test_stratified_subsample_preserves_group_proportions() -> None:
    import pandas as pd

    ns = _load()
    subsample = ns["_stratified_subsample"]

    # 4 fibroblast_subclusters x 2 conditions, deliberately imbalanced sizes,
    # mirroring the real object's smallest cluster (Mechanosensory, ~2,913 of
    # 61,608 cells) so a naive random subsample could plausibly drop it entirely.
    rows = []
    sizes = {("0", "Normal"): 3000, ("0", "LE"): 4000, ("1", "Normal"): 100, ("1", "LE"): 150}
    for (cl, cond), n in sizes.items():
        rows += [{"cluster": cl, "condition": cond}] * n
    obs = pd.DataFrame(rows, index=[f"c{i}" for i in range(len(rows))])

    kept = subsample(obs, "cluster", "condition", max_cells=500, seed=0)

    assert len(kept) <= 550  # rounding may push slightly over/under 500
    kept_obs = obs.loc[kept]
    # every group present in the input must still be represented, however small
    for cl, cond in sizes:
        assert ((kept_obs["cluster"] == cl) & (kept_obs["condition"] == cond)).sum() >= 1
    # the two real groups' relative sizes should roughly survive (proportional,
    # not equal) -- cluster 0 (7000 of 7250 cells) should dominate the subsample
    assert (kept_obs["cluster"] == "0").sum() > (kept_obs["cluster"] == "1").sum()


def test_stratified_subsample_is_deterministic() -> None:
    import pandas as pd

    ns = _load()
    subsample = ns["_stratified_subsample"]
    obs = pd.DataFrame({"cluster": ["a"] * 200 + ["b"] * 50}, index=[f"c{i}" for i in range(250)])

    first = subsample(obs, "cluster", None, max_cells=60, seed=7)
    second = subsample(obs, "cluster", None, max_cells=60, seed=7)
    assert list(first) == list(second)


def test_stratified_subsample_no_op_when_already_small() -> None:
    import pandas as pd

    ns = _load()
    subsample = ns["_stratified_subsample"]
    obs = pd.DataFrame({"cluster": ["a"] * 10}, index=[f"c{i}" for i in range(10)])

    kept = subsample(obs, "cluster", None, max_cells=500, seed=0)
    assert list(kept) == list(obs.index)


def test_write_skip_creates_empty_ranking_and_marker(tmp_path: Path) -> None:
    ns = _load()
    ns["write_skip"](tmp_path, "t", "no base GRN")
    ranking = tmp_path / "perturbation_ranking.csv"
    marker = tmp_path / "perturbation_SKIPPED_t.txt"
    assert ranking.exists()
    assert ranking.read_text().splitlines()[0] == "tf,score,n_cells,direction"
    # header only, no data rows
    assert len(ranking.read_text().strip().splitlines()) == 1
    assert marker.exists()
    assert "no base GRN" in marker.read_text()


def test_write_ranking_writes_rows(tmp_path: Path) -> None:
    ns = _load()
    rows = [
        {"tf": "PROX1", "score": 0.9, "n_cells": 100, "direction": "directional"},
        {"tf": "PIEZO1", "score": 0.5, "n_cells": 100, "direction": "directional"},
    ]
    out = ns["write_ranking"](tmp_path, "t", rows)
    assert Path(out).exists()
    text = Path(out).read_text()
    assert text.splitlines()[0] == "tf,score,n_cells,direction"
    assert "PROX1" in text and "PIEZO1" in text


def test_script_skips_gracefully_when_celloracle_absent(tmp_path: Path) -> None:
    # Running end-to-end with no celloracle installed must exit 0 and write a skip.
    h5ad = tmp_path / "in.h5ad"
    h5ad.write_text("")  # never read — import gate trips first
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--h5ad",
            str(h5ad),
            "--out-dir",
            str(out_dir),
            "--tag",
            "t",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert (out_dir / "perturbation_ranking.csv").exists()
    assert (out_dir / "perturbation_SKIPPED_t.txt").exists()
