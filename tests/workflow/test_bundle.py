from __future__ import annotations

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
