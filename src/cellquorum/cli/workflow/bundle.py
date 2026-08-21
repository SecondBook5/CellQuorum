"""Assemble one hypothesis's per-cell-type run outputs into a publication bundle."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

# "provenance" carries the reproducibility record (artifact_manifest.csv,
# stage_execution_records.json, pipeline_plan.json) into the published bundle.
# It is copied on the same silent-degrade-if-missing basis as figures/results.
_COPY_SUBDIRS = ("figures", "results", "provenance")

# A run is "completed" iff it produced this file -- the declared output of the
# run_analysis rule. Its presence is the same completion signal the status
# matrix uses (see cli.workflow.status); a crashed run lacks it.
_COMPLETION_MARKER = Path("provenance") / "artifact_manifest.csv"

_COMPLETED = "completed"
_FAILED = "failed"
_MISSING = "missing"

# A warning triangle as an HTML entity keeps the emitted document pure ASCII,
# so writing it is independent of the process locale encoding.
_WARN = "&#9888;"


def _pair_status(run_dir: Path) -> str:
    """Classify one pair's run directory.

    Mirrors the honest states the status matrix uses -- a failure is never left
    implicit:

    - ``missing``: no run directory on disk (the run never produced output).
    - ``failed``: a directory exists but the completion marker is absent (the
      run crashed before finishing).
    - ``completed``: the completion marker is present.

    Args:
        run_dir: The per-(hypothesis, cell_type) run directory.

    Returns:
        One of ``"completed"``, ``"failed"``, ``"missing"``.
    """
    if not run_dir.exists():
        return _MISSING
    if (run_dir / _COMPLETION_MARKER).is_file():
        return _COMPLETED
    return _FAILED


def assemble_bundle(
    hypothesis_id: str,
    title: str,
    run_dirs: dict[str, Path],
    bundle_dir: Path,
) -> Path:
    """Collect each cell type's run outputs into a publishable hypothesis bundle.

    A crashed or never-run pair is flagged loudly in both the HTML report and a
    machine-readable ``bundle_status.json`` -- it is never shown as a silent
    empty section that a reader could mistake for a successful run with no
    figures.

    Args:
        hypothesis_id: The hypothesis identifier (used in filenames/headings).
        title: Human-readable hypothesis title for the report heading.
        run_dirs: Mapping of cell type to its run directory.
        bundle_dir: Destination directory for the assembled bundle.

    Returns:
        Path to the written ``report.html``.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    cell_status: dict[str, dict] = {}
    buckets: dict[str, list[str]] = {_COMPLETED: [], _FAILED: [], _MISSING: []}

    for cell_type, run_dir in sorted(run_dirs.items()):
        run_dir = Path(run_dir)
        status = _pair_status(run_dir)
        buckets[status].append(cell_type)
        dest = bundle_dir / cell_type
        items: list[str] = []
        # Copy whatever exists (a crashed run's partial provenance still aids
        # debugging); a missing run has nothing to copy. The section LABEL, not
        # the byte-copy, is what keeps a failure from masquerading as success.
        for sub in _COPY_SUBDIRS:
            src = run_dir / sub
            if src.is_dir():
                shutil.copytree(src, dest / sub, dirs_exist_ok=True)
                for artifact in sorted(src.rglob("*")):
                    if artifact.is_file():
                        rel = artifact.relative_to(run_dir)
                        items.append(f"<li>{html.escape(str(rel))}</li>")
        cell_status[cell_type] = {"status": status, "artifact_count": len(items)}

        name = html.escape(cell_type)
        if status == _COMPLETED:
            heading = name
            listing = "\n".join(items) or "<li><em>no artifacts</em></li>"
        elif status == _FAILED:
            heading = f"{name} &mdash; {_WARN} FAILED (run did not complete)"
            listing = "\n".join(items) or "<li><em>no artifacts recovered</em></li>"
        else:  # missing
            heading = f"{name} &mdash; {_WARN} MISSING (no run output found)"
            listing = "<li><em>no run directory on disk</em></li>"
        sections.append(f"<h2>{heading}</h2>\n<ul>\n{listing}\n</ul>")

    n_total = len(run_dirs)
    summary_bits = [f"{len(buckets[_COMPLETED])} of {n_total} cell types completed"]
    if buckets[_FAILED]:
        summary_bits.append("failed: " + ", ".join(html.escape(c) for c in buckets[_FAILED]))
    if buckets[_MISSING]:
        summary_bits.append("missing: " + ", ".join(html.escape(c) for c in buckets[_MISSING]))
    summary = "; ".join(summary_bits)

    body = "\n".join(sections)
    doc = (
        "<!doctype html>\n<html>\n<head>\n<meta charset='utf-8'>\n"
        f"<title>{html.escape(title)}</title>\n</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n"
        f"<p>Hypothesis: <code>{html.escape(hypothesis_id)}</code></p>\n"
        f"<p class='status'>{summary}</p>\n"
        f"{body}\n</body>\n</html>\n"
    )
    report = bundle_dir / "report.html"
    report.write_text(doc)

    # Machine-readable companion so a caller (or CI) can act on partial failure
    # without scraping HTML -- the engine never leaves a failure implicit.
    status_doc = {
        "hypothesis": hypothesis_id,
        "title": title,
        "cell_types": cell_status,
        "completed": buckets[_COMPLETED],
        "failed": buckets[_FAILED],
        "missing": buckets[_MISSING],
    }
    (bundle_dir / "bundle_status.json").write_text(json.dumps(status_doc, indent=2) + "\n")
    return report
