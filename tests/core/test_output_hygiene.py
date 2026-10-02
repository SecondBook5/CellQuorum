"""Tests for the run-directory hygiene gate.

The gate exists because of a specific near-miss: under the global-QC-once
architecture every per-lineage run sets ``stages.qc: false``, and a reused run
directory kept 197 MB of the previous run's QC output -- thresholds, per-cell
decisions and a 188 MB ``qc.h5ad`` -- inside a run whose own summary said
``qc | skipped``. Nothing on disk marked those files as foreign, and the run
directory is what figures and tables are rebuilt from.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cellquorum.core.output_hygiene import (
    StaleOutputError,
    assert_output_dir_matches_config,
    find_disabled_stage_outputs,
    find_inherited_artifacts,
    format_inherited_artifacts,
    format_stale_outputs,
)

RUN_START = datetime(2026, 9, 2, 21, 4, 54, tzinfo=UTC)


def make_run_dir(tmp_path: Path, *, files: dict[str, int]) -> Path:
    """
    Build a run directory containing the given files.

    Args:
        tmp_path: Pytest temporary directory.
        files: Mapping of run-relative path to byte size.

    Returns:
        The run root.
    """

    root = tmp_path / "run"
    for relative, size in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_a_disabled_stage_with_output_on_disk_is_found(tmp_path: Path) -> None:
    """
    Verify output for a disabled stage is reported with its size.

    This is the case that motivated the gate. The size is part of the finding
    because a 188 MB object left behind is both the most misleading artifact and
    the most expensive one to keep.
    """

    root = make_run_dir(
        tmp_path,
        files={"results/qc/thresholds.csv": 100, "results/qc/qc.h5ad": 5_000},
    )

    findings = find_disabled_stage_outputs(root, disabled_stages=["qc"])

    assert len(findings) == 1
    assert findings[0].stage == "qc"
    assert findings[0].directory == "results/qc"
    assert findings[0].n_files == 2
    assert findings[0].total_bytes == 5_100


def test_an_enabled_stage_with_output_on_disk_is_ignored(tmp_path: Path) -> None:
    """
    Verify a stage that will actually run is not flagged.

    A stage that runs overwrites its own outputs, so pre-existing files there are
    normal. Flagging them would make the gate fire on every ordinary re-run and
    train people to bypass it.
    """

    root = make_run_dir(tmp_path, files={"results/qc/thresholds.csv": 100})

    assert find_disabled_stage_outputs(root, disabled_stages=["trajectory"]) == []


def test_an_empty_directory_for_a_disabled_stage_is_not_a_finding(tmp_path: Path) -> None:
    """
    Verify a bare directory is tolerated.

    An empty ``results/qc/`` misleads nobody -- there is nothing in it to read as a
    result -- and directories get created as a side effect of ordinary path handling.
    """

    root = tmp_path / "run"
    (root / "results" / "qc").mkdir(parents=True)

    assert find_disabled_stage_outputs(root, disabled_stages=["qc"]) == []


def test_nested_files_are_counted(tmp_path: Path) -> None:
    """
    Verify the walk is recursive.

    Stages nest their outputs (per-group subdirectories, figure format folders), so
    a shallow listing would report a stage directory as empty while it held
    gigabytes one level down.
    """

    root = make_run_dir(
        tmp_path,
        files={
            "figures/trajectory/group_0/umap.png": 10,
            "figures/trajectory/group_1/umap.png": 20,
        },
    )

    findings = find_disabled_stage_outputs(root, disabled_stages=["trajectory"])

    assert findings[0].n_files == 2
    assert findings[0].total_bytes == 30


def test_both_results_and_figures_are_checked(tmp_path: Path) -> None:
    """
    Verify a stage is caught in either output subtree, largest first.

    Stale figures are as citable as stale tables, and ordering by size puts the
    artifact most likely to be mistaken for a result at the top of the message.
    """

    root = make_run_dir(
        tmp_path,
        files={"results/qc/qc.h5ad": 5_000, "figures/qc/violin.png": 10},
    )

    findings = find_disabled_stage_outputs(root, disabled_stages=["qc"])

    assert [finding.directory for finding in findings] == ["results/qc", "figures/qc"]


def test_a_clean_run_directory_passes(tmp_path: Path) -> None:
    """
    Verify the gate stays silent on a directory consistent with the config.

    A gate that fires on correct input is worse than no gate, because the first
    thing a user does is disable it.
    """

    root = make_run_dir(tmp_path, files={"results/de_pseudobulk_edger.csv": 100})

    assert_output_dir_matches_config(root, disabled_stages=["qc", "grn"])


def test_the_gate_raises_and_names_the_remedy(tmp_path: Path) -> None:
    """
    Verify the error identifies the directory and the command that clears it.

    The gate halts a run, so it owes the user the specific path -- an error that
    only says "stale outputs" costs more time than the leftovers would have.
    """

    root = make_run_dir(tmp_path, files={"results/qc/thresholds.csv": 100})

    with pytest.raises(StaleOutputError) as excinfo:
        assert_output_dir_matches_config(root, disabled_stages=["qc"])

    message = str(excinfo.value)
    assert "results/qc" in message
    assert "disabled" in message
    assert f"rm -rf {root / 'results/qc'}" in message


def test_a_missing_run_directory_is_not_an_error(tmp_path: Path) -> None:
    """
    Verify a first run is unaffected.

    The gate runs before stages execute, which on a fresh run is before most of the
    output tree exists.
    """

    assert_output_dir_matches_config(tmp_path / "never_created", disabled_stages=["qc"])


def test_the_message_reports_total_size_in_megabytes(tmp_path: Path) -> None:
    """
    Verify the size is rendered, since it is the argument for acting.

    "197 MB of QC output" is what makes the problem legible; a file count alone
    reads as housekeeping.
    """

    root = make_run_dir(tmp_path, files={"results/qc/qc.h5ad": 2_000_000})

    message = format_stale_outputs(
        find_disabled_stage_outputs(root, disabled_stages=["qc"]), root=root
    )

    assert "2.0 MB" in message


def set_mtime(path: Path, when: datetime) -> None:
    """
    Force a file's modification time.

    Args:
        path: File to touch.
        when: Timezone-aware target time.
    """

    stamp = when.timestamp()
    os.utime(path, (stamp, stamp))


def test_an_artifact_from_a_dropped_group_is_inventoried(tmp_path: Path) -> None:
    """
    Verify a sibling left behind by a skipped group is found.

    The real case: a velocity run wrote 18 per-group objects and correctly skipped
    group 4 for holding 28 cells under a 30-cell floor. The previous run's
    ``4.h5ad`` stayed on disk, for a different partition, looking exactly like the
    18 that were current.
    """

    root = make_run_dir(
        tmp_path,
        files={
            "results/trajectory/velocity/3.h5ad": 100,
            "results/trajectory/velocity/4.h5ad": 900,
            "results/trajectory/velocity/5.h5ad": 100,
        },
    )
    set_mtime(root / "results/trajectory/velocity/4.h5ad", RUN_START - timedelta(hours=10))
    for fresh in ("3.h5ad", "5.h5ad"):
        set_mtime(root / "results/trajectory/velocity" / fresh, RUN_START + timedelta(minutes=20))

    findings = find_inherited_artifacts(root, run_started_at=RUN_START)

    assert [finding.path for finding in findings] == ["results/trajectory/velocity/4.h5ad"]
    assert findings[0].size_bytes == 900


def test_a_run_that_wrote_everything_reports_nothing(tmp_path: Path) -> None:
    """
    Verify a fully rewritten tree is clean.

    Anything else would make the report fire on every ordinary run.
    """

    root = make_run_dir(tmp_path, files={"figures/trajectory/fate_5.png": 10})
    set_mtime(root / "figures/trajectory/fate_5.png", RUN_START + timedelta(minutes=1))

    assert find_inherited_artifacts(root, run_started_at=RUN_START) == []


def test_files_just_before_the_run_start_are_within_the_grace_window(tmp_path: Path) -> None:
    """
    Verify a small negative skew is tolerated.

    Filesystem mtimes and the recorded run start come from different clocks and
    granularities, and a stage can begin writing moments before the recorded start.
    The report targets leftovers hours old, so a minute of slack loses nothing.
    """

    root = make_run_dir(tmp_path, files={"results/x.csv": 10})
    set_mtime(root / "results/x.csv", RUN_START - timedelta(seconds=30))

    assert find_inherited_artifacts(root, run_started_at=RUN_START) == []


def test_a_naive_run_start_is_refused(tmp_path: Path) -> None:
    """
    Verify a timestamp without a timezone is rejected rather than assumed.

    Treating a naive datetime as local time shifts the cutoff by the UTC offset,
    which either hides real leftovers or invents them -- and does so silently, which
    is exactly the failure mode this module exists to remove.
    """

    root = make_run_dir(tmp_path, files={"results/x.csv": 10})

    with pytest.raises(ValueError, match="timezone-aware"):
        find_inherited_artifacts(root, run_started_at=datetime(2026, 9, 2, 21, 4, 54))


def test_the_inventory_is_ordered_by_size(tmp_path: Path) -> None:
    """
    Verify the largest leftover is reported first.

    The message names only the first few, so ordering decides whether a 189 MB
    object or a 12 KB legend appears in the run summary.
    """

    root = make_run_dir(tmp_path, files={"results/small.csv": 10, "results/big.h5ad": 5_000})
    for name in ("small.csv", "big.h5ad"):
        set_mtime(root / "results" / name, RUN_START - timedelta(hours=1))

    findings = find_inherited_artifacts(root, run_started_at=RUN_START)

    assert [finding.path for finding in findings] == ["results/big.h5ad", "results/small.csv"]


def test_a_long_inventory_is_truncated_in_the_message(tmp_path: Path) -> None:
    """
    Verify the warning stays readable when a resume inherits many files.

    An unbounded list scrolls the run summary off screen, and a warning nobody can
    read is how the QC leftovers survived in the first place.
    """

    root = make_run_dir(tmp_path, files={f"results/f{index}.csv": 10 for index in range(25)})
    for index in range(25):
        set_mtime(root / "results" / f"f{index}.csv", RUN_START - timedelta(hours=1))

    message = format_inherited_artifacts(
        find_inherited_artifacts(root, run_started_at=RUN_START), max_listed=10
    )

    assert "25 file(s)" in message
    assert "and 15 more" in message


def test_an_empty_inventory_produces_no_message(tmp_path: Path) -> None:
    """
    Verify a clean run emits no warning text at all.

    The caller keys off the empty string to decide whether to warn and whether to
    write the CSV, so a non-empty "nothing to report" would create both.
    """

    assert format_inherited_artifacts([]) == ""
