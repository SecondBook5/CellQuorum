from __future__ import annotations

import json
from pathlib import Path

from cellquorum.cli.workflow.bundle import assemble_bundle


def _fake_run(dir_: Path) -> Path:
    for sub in ("figures", "results", "reports", "provenance"):
        (dir_ / sub).mkdir(parents=True)
    (dir_ / "figures" / "umap.png").write_bytes(b"PNG")
    (dir_ / "results" / "de_table.csv").write_text("gene,lfc\nIl33,1.2\n")
    (dir_ / "provenance" / "artifact_manifest.csv").write_text("name,path\n")
    (dir_ / "provenance" / "stage_execution_records.json").write_text("[]")
    return dir_


def _crashed_run(dir_: Path) -> Path:
    """A run that started (dir + partial artifacts) but never completed.

    The completion signal is ``provenance/artifact_manifest.csv`` (the
    ``run_analysis`` rule's declared output); a crashed run lacks it.
    """
    (dir_ / "logs").mkdir(parents=True)
    (dir_ / "logs" / "run.log").write_text("Traceback ...\n")
    return dir_


def test_assemble_bundle_collects_and_reports(tmp_path: Path) -> None:
    kc = _fake_run(tmp_path / "runs" / "il33_axis" / "KC")
    ilc = _fake_run(tmp_path / "runs" / "il33_axis" / "ILC")
    bundle_dir = tmp_path / "bundles" / "il33_axis"

    report = assemble_bundle(
        "il33_axis",
        "IL33/ST2 alarmin KC->ILC2 axis",
        {"KC": kc, "ILC": ilc},
        bundle_dir,
    )

    assert report == bundle_dir / "report.html"
    assert report.exists()
    assert (bundle_dir / "KC" / "figures" / "umap.png").exists()
    assert (bundle_dir / "ILC" / "results" / "de_table.csv").exists()
    # Provenance travels with the published bundle for reproducibility.
    assert (bundle_dir / "KC" / "provenance" / "artifact_manifest.csv").exists()
    assert (bundle_dir / "ILC" / "provenance" / "stage_execution_records.json").exists()
    html = report.read_text()
    assert "IL33/ST2 alarmin KC-&gt;ILC2 axis" in html or "IL33/ST2 alarmin KC->ILC2 axis" in html
    assert "KC" in html and "ILC" in html


def test_assemble_bundle_flags_failed_and_missing_pairs(tmp_path: Path) -> None:
    """A crashed or never-run pair must be flagged loudly, never shown as a
    silent empty section that looks like a successful run with no figures."""
    runs = tmp_path / "runs" / "il33_axis"
    kc = _fake_run(runs / "KC")  # completed
    ilc = _crashed_run(runs / "ILC")  # started but no artifact_manifest.csv
    mast = runs / "MAST"  # never created on disk at all
    bundle_dir = tmp_path / "bundles" / "il33_axis"

    report = assemble_bundle(
        "il33_axis",
        "IL33/ST2 alarmin axis",
        {"KC": kc, "ILC": ilc, "MAST": mast},
        bundle_dir,
    )

    assert report.exists()

    # Machine-readable status travels with the bundle (the engine never leaves a
    # failure implicit).
    status_path = bundle_dir / "bundle_status.json"
    assert status_path.exists()
    status = json.loads(status_path.read_text())
    assert status["hypothesis"] == "il33_axis"
    assert status["cell_types"]["KC"]["status"] == "completed"
    assert status["cell_types"]["ILC"]["status"] == "failed"
    assert status["cell_types"]["MAST"]["status"] == "missing"
    assert status["completed"] == ["KC"]
    assert status["failed"] == ["ILC"]
    assert status["missing"] == ["MAST"]

    # The HTML report surfaces the failure loudly next to the cell type.
    html = report.read_text()
    assert "FAILED" in html
    assert "MISSING" in html
    # A one-glance summary of how many completed.
    assert "1 of 3" in html

    # The completed pair's artifacts still travel; a missing pair copies nothing.
    assert (bundle_dir / "KC" / "figures" / "umap.png").exists()
    assert not (bundle_dir / "MAST").exists()
