"""Assemble one hypothesis's per-cell-type run outputs into a publication bundle."""

from __future__ import annotations

import html
import shutil
from pathlib import Path

# "provenance" carries the reproducibility record (artifact_manifest.csv,
# stage_execution_records.json, pipeline_plan.json) into the published bundle.
# It is copied on the same silent-degrade-if-missing basis as figures/results.
_COPY_SUBDIRS = ("figures", "results", "provenance")


def assemble_bundle(
    hypothesis_id: str,
    title: str,
    run_dirs: dict[str, Path],
    bundle_dir: Path,
) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    for cell_type, run_dir in sorted(run_dirs.items()):
        dest = bundle_dir / cell_type
        items: list[str] = []
        for sub in _COPY_SUBDIRS:
            src = Path(run_dir) / sub
            if src.is_dir():
                shutil.copytree(src, dest / sub, dirs_exist_ok=True)
                for artifact in sorted(src.rglob("*")):
                    if artifact.is_file():
                        rel = artifact.relative_to(run_dir)
                        items.append(f"<li>{html.escape(str(rel))}</li>")
        listing = "\n".join(items) or "<li><em>no artifacts</em></li>"
        sections.append(f"<h2>{html.escape(cell_type)}</h2>\n<ul>\n{listing}\n</ul>")

    body = "\n".join(sections)
    doc = (
        "<!doctype html>\n<html>\n<head>\n<meta charset='utf-8'>\n"
        f"<title>{html.escape(title)}</title>\n</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n"
        f"<p>Hypothesis: <code>{html.escape(hypothesis_id)}</code></p>\n"
        f"{body}\n</body>\n</html>\n"
    )
    report = bundle_dir / "report.html"
    report.write_text(doc)
    return report
