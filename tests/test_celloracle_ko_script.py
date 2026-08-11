"""Tests for the in-env CellOracle KO script's pure helpers + skip path.

No real CellOracle in CI — only argument parsing, the ranking-schema writer, and
the graceful-skip writer are exercised.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

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
    assert args.knn_n_neighbors == 200
    assert args.n_propagation == 3
    assert args.seed == 0


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
