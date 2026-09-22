"""Tests for the in-env run_metacell2.py graceful-skip contract.

CI has no metacells_env / metacells package, so these exercise the
pure-Python skip paths only: the script must write a SKIPPED sentinel + an
empty assignments CSV and exit 0 when the metacells package is not
importable, never crash the caller.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/cellquorum/backends/metacells_scripts/run_metacell2.py"
)


def _load_script_module():
    """Import run_metacell2.py by path (it lives outside any package, only
    ever invoked as a subprocess) so its pure-Python helper functions --
    those reachable without a real `metacells` install -- can be unit tested
    directly, not just through a subprocess."""

    spec = importlib.util.spec_from_file_location("run_metacell2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_skips_when_metacells_package_not_importable(tmp_path: Path) -> None:
    out = tmp_path / "out"
    # This test's own interpreter (pytest's env) has no `metacells` package
    # installed, so the script's own import-gate fires exactly as it would
    # in CI -- no monkeypatching needed.
    res = _run(
        [
            str(tmp_path / "does_not_matter.h5ad"),
            str(out),
            "X_pca_harmony",
            "counts",
            "cell_type",
            "96",
            "5000",
            "0",
        ]
    )
    assert res.returncode == 0, res.stderr
    assert (out / "metacells_SKIPPED.txt").exists()
    assert (
        (out / "metacell_assignments.csv")
        .read_text()
        .startswith("cell_id,metacell_id,group,n_umis")
    )


def test_wrong_argument_count_exits_nonzero() -> None:
    res = _run(["only_one_arg"])
    assert res.returncode != 0


def test_no_study_specific_biology_in_source() -> None:
    text = SCRIPT.read_text()
    for forbidden in ("fibroblast", "lymphedema", "keratinocyte", "AAH", "AIS"):
        assert forbidden.lower() not in text.lower()


# ---------------------------------------------------------------------------
# Finding 4: an absent/renamed `metacell_name` column on a real MetaCell2 API
# version would leave `metacell_by_cell_id` empty and every cell falls
# through to "Outliers" -- a silently "successful" run. The real
# `metacells.pl.collect_metacells` call can't be exercised here (no real
# package installed, same constraint as the rest of this plan), but the
# extracted post-pipeline sanity check is a small, pure function and is
# fully testable in isolation.
# ---------------------------------------------------------------------------


def test_mapping_failure_reason_fires_when_metacells_exist_but_nothing_mapped() -> None:
    mod = _load_script_module()
    reason = mod.mapping_failure_reason(n_metacells_produced=12, n_mapped_cells=0)
    assert reason is not None
    assert "metacell_name" in reason


def test_mapping_failure_reason_is_none_when_mapping_worked() -> None:
    mod = _load_script_module()
    assert mod.mapping_failure_reason(n_metacells_produced=12, n_mapped_cells=340) is None


def test_mapping_failure_reason_is_none_when_no_metacells_produced() -> None:
    """A genuinely-empty run (e.g. all cells outliers because MetaCell2 itself
    found no coherent metacells) is a legitimate outcome, not this failure
    mode -- only "metacells exist but nothing mapped to them" should fire."""

    mod = _load_script_module()
    assert mod.mapping_failure_reason(n_metacells_produced=0, n_mapped_cells=0) is None
