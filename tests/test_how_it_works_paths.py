"""docs/how-it-works.md walks a run through REAL files — pin the spine to disk.

Every file on the run-trace spine must (a) exist and (b) be cited in the doc,
so the walkthrough can never point a newcomer at a path that moved or vanished.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "how-it-works.md"

SPINE = [
    "src/cellquorum/cli/app.py",
    "src/cellquorum/api/pipeline.py",
    "src/cellquorum/config/models.py",
    "src/cellquorum/core/stage_catalog.py",
    "src/cellquorum/core/planner.py",
    "src/cellquorum/io/manifest.py",
    "src/cellquorum/core/executor.py",
    "src/cellquorum/core/stage_artifact_writer.py",
    "src/cellquorum/stages/qc/stage.py",
]


def test_spine_files_exist():
    missing = [p for p in SPINE if not (REPO / p).is_file()]
    assert not missing, f"walkthrough spine files missing on disk: {missing}"


def test_walkthrough_cites_every_spine_file():
    text = DOC.read_text(encoding="utf-8")
    # Cite by module-relative path (e.g. cli/app.py, core/executor.py, stages/qc/stage.py).
    uncited = [p for p in SPINE if p.split("src/cellquorum/", 1)[1] not in text]
    assert not uncited, f"how-it-works.md does not cite: {uncited}"
