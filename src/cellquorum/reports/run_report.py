"""Human-readable run report: Markdown, HTML, and a methods paragraph.

The report is a *consumer* of provenance, not a recomputation. It reads the
stage execution records the executor already produced and renders three
publication-oriented artifacts under ``reports/``:

* ``report.md`` — a per-stage status table with skip reasons and warnings;
* ``report.html`` — the same content as a standalone HTML page;
* ``methods.txt`` — a short Methods paragraph naming the seed, backends, and the
  stages that ran, suitable as a starting point for a manuscript methods section.

Rendering is best-effort by contract (see the report hook in
``core.pipeline.execute_pipeline_run``): a rendering failure should not fail the
run unless the user opts in via ``report.fail_on_report_error``.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.core.stage import StageExecutionRecord


def _status_rows(records: list[StageExecutionRecord]) -> list[dict[str, str]]:
    """Flatten records into simple display rows (execution order preserved)."""

    rows: list[dict[str, str]] = []
    for record in records:
        skip_reason = ""
        skip = getattr(record, "skip_reason", None)
        if skip is not None:
            skip_reason = getattr(skip, "reason", str(skip))
        rows.append(
            {
                "stage": record.stage_name,
                "status": record.status,
                "backend": record.backend_used or "",
                "duration_s": (
                    "" if record.duration_seconds is None else f"{record.duration_seconds:.2f}"
                ),
                "detail": skip_reason,
                "n_warnings": str(len(record.warnings)),
            }
        )
    return rows


def render_markdown(
    *,
    config: CellQuorumConfig,
    records: list[StageExecutionRecord],
) -> str:
    """Render the run report as Markdown."""

    project = getattr(getattr(config, "project", None), "name", "cellquorum-run")
    rows = _status_rows(records)
    n_success = sum(1 for r in rows if r["status"] == "success")
    n_skipped = sum(1 for r in rows if r["status"] == "skipped")
    n_failed = sum(1 for r in rows if r["status"] == "failed")

    lines: list[str] = []
    lines.append(f"# CellQuorum run report — {project}")
    lines.append("")
    lines.append(
        f"**Stages:** {len(rows)} total — {n_success} success, "
        f"{n_skipped} skipped, {n_failed} failed."
    )
    lines.append("")
    lines.append("| Stage | Status | Backend | Duration (s) | Detail | Warnings |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['stage']} | {r['status']} | {r['backend']} | "
            f"{r['duration_s']} | {r['detail']} | {r['n_warnings']} |"
        )
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append(render_methods_text(config=config, records=records))
    lines.append("")
    return "\n".join(lines)


def render_html(
    *,
    config: CellQuorumConfig,
    records: list[StageExecutionRecord],
) -> str:
    """Render the run report as a standalone HTML page."""

    project = getattr(getattr(config, "project", None), "name", "cellquorum-run")
    rows = _status_rows(records)

    body: list[str] = []
    body.append(f"<h1>CellQuorum run report — {html.escape(str(project))}</h1>")
    body.append("<table border='1' cellspacing='0' cellpadding='4'>")
    body.append(
        "<tr><th>Stage</th><th>Status</th><th>Backend</th>"
        "<th>Duration (s)</th><th>Detail</th><th>Warnings</th></tr>"
    )
    for r in rows:
        cells = "".join(
            f"<td>{html.escape(str(r[key]))}</td>"
            for key in ("stage", "status", "backend", "duration_s", "detail", "n_warnings")
        )
        body.append(f"<tr>{cells}</tr>")
    body.append("</table>")
    body.append("<h2>Methods</h2>")
    body.append(f"<p>{html.escape(render_methods_text(config=config, records=records))}</p>")

    return (
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        f"<title>CellQuorum run report — {html.escape(str(project))}</title></head>"
        f"<body>{''.join(body)}</body></html>"
    )


def render_methods_text(
    *,
    config: CellQuorumConfig,
    records: list[StageExecutionRecord],
) -> str:
    """
    Render a short Methods paragraph from the resolved config and records.

    Names the successful stages, the backends used, and the random seed, so the
    text is a faithful starting point for a manuscript methods section.
    """

    seed = getattr(getattr(config, "run", None), "random_seed", None)
    succeeded = [r.stage_name for r in records if r.status == "success"]
    backends = sorted({r.backend_used for r in records if r.backend_used})

    if not succeeded:
        stages_phrase = "no analysis stages completed"
    else:
        stages_phrase = "the following stages were run: " + ", ".join(succeeded)

    sentence = f"Analysis was performed with CellQuorum; {stages_phrase}. "
    if backends:
        sentence += f"Computation used the {', '.join(backends)} backend(s). "
    if seed is not None:
        sentence += f"A fixed random seed ({seed}) was used for reproducibility. "
    sentence += (
        "Full per-stage parameters, versions, and provenance are recorded under "
        "the run's provenance/ directory."
    )
    return sentence


def write_run_report(
    *,
    config: CellQuorumConfig,
    records: list[StageExecutionRecord],
    artifact_manager: object,
) -> list[str]:
    """
    Render and write the configured report artifacts under ``reports/``.

    Args:
        config: The resolved run configuration (honors ``config.report``).
        records: Stage execution records to summarize.
        artifact_manager: An ArtifactManager rooted at the run directory.

    Returns:
        Warnings produced while rendering (empty on success).

    Raises:
        Exception: Propagates rendering/writing errors so the caller can honor
            ``report.fail_on_report_error``.
    """

    report_config = getattr(config, "report", None)
    if report_config is None or not getattr(report_config, "enabled", False):
        return []

    warnings: list[str] = []

    # Always write the methods paragraph; it is the publication payoff.
    methods_text = render_methods_text(config=config, records=records)
    artifact_manager.write_text(
        methods_text,
        name="run_report_methods",
        relative_path="reports/methods.txt",
        kind="text",
        description="Auto-generated methods paragraph for the run.",
    )

    if getattr(report_config, "markdown", False):
        artifact_manager.write_text(
            render_markdown(config=config, records=records),
            name="run_report_markdown",
            relative_path="reports/report.md",
            kind="markdown",
            description="Human-readable Markdown run report.",
        )

    if getattr(report_config, "html", False):
        artifact_manager.write_text(
            render_html(config=config, records=records),
            name="run_report_html",
            relative_path="reports/report.html",
            kind="html",
            description="Standalone HTML run report.",
        )

    # PDF rendering is a documented follow-up; note it rather than failing.
    if getattr(report_config, "pdf", False):
        warnings.append("PDF report rendering is not implemented yet; skipped.")

    return warnings


__all__ = [
    "render_html",
    "render_markdown",
    "render_methods_text",
    "write_run_report",
]
