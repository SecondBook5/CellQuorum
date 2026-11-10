"""Human-readable run report: Markdown, HTML, and a methods paragraph.

The report is a *consumer* of provenance, not a recomputation. It reads the
stage execution records the executor already produced and renders three
publication-oriented artifacts under ``reports/``:

* ``report.md`` — a per-stage status table (skip reason or error in the detail
  column) followed by every warning, verbatim, grouped by the stage that emitted
  it;
* ``report.html`` — the same content as a standalone HTML page;
* ``methods.txt`` — a short Methods paragraph naming the seed, backends, and the
  stages that ran, suitable as a starting point for a manuscript methods section.

Rendering is best-effort by contract (see the report hook in
``core.pipeline.execute_pipeline_run``): a rendering failure should not fail the
run unless the user opts in via ``report.fail_on_report_error``.
"""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.core.artifacts import ArtifactManager
    from cellquorum.core.planner import PipelinePlan
    from cellquorum.core.stage import StageExecutionRecord


def _cell(text: str, *, limit: int = 140) -> str:
    """Make ``text`` safe for one Markdown table cell.

    A pipe or a newline inside a skip reason or an error message silently breaks
    the table for every row after it, and skip reasons are free-form strings from
    thirty-odd stages. Truncation keeps the table readable; the full text is in
    the sections below it.
    """
    flat = " ".join(str(text).split())
    flat = flat.replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _detail(record: StageExecutionRecord) -> str:
    """Why this stage did not simply succeed: skip reason, or the error.

    A failed stage's ``StageExecutionError`` was previously never rendered
    anywhere in the report — the row showed a blank detail and a warning count of
    zero, so a failed run's report did not say what failed. ``StageExecutionError``
    exists precisely so failures stay auditable, so it belongs here.
    """
    error = getattr(record, "error", None)
    if error is not None:
        error_type = getattr(error, "error_type", "") or "error"
        return f"{error_type}: {getattr(error, 'message', str(error))}"
    skip = getattr(record, "skip_reason", None)
    return getattr(skip, "reason", str(skip)) if skip is not None else ""


def _status_rows(records: list[StageExecutionRecord]) -> list[dict[str, str]]:
    """Flatten records into simple display rows (execution order preserved)."""

    rows: list[dict[str, str]] = []
    for record in records:
        rows.append(
            {
                "stage": record.stage_name,
                "status": "reused" if record.reused_from_checkpoint else record.status,
                "backend": record.backend_used or "",
                "duration_s": (
                    "" if record.duration_seconds is None else f"{record.duration_seconds:.2f}"
                ),
                "detail": _detail(record),
                "n_warnings": str(len(record.warnings)),
            }
        )
    return rows


def _study_scope(
    records: list[StageExecutionRecord],
    plan: PipelinePlan | None = None,
) -> tuple[str, list[StageExecutionRecord], dict[str, list[str]], list[str]]:
    """Summarize analysis records and collect the stated reason for each omitted stage."""
    active = []
    omissions: dict[str, list[str]] = {}
    for record in records:
        if record.disabled_by_config:
            reason = _detail(record)
            omissions.setdefault(reason, []).append(record.stage_name)
        else:
            active.append(record)
    counts = Counter(
        "reused" if r.reused_from_checkpoint else r.status
        for r in active
        if r.stage_name != "bootstrap"
    )
    observed = {r.stage_name for r in records}
    unreached = []
    for stage in plan.stages if plan else ():
        if stage.name in observed:
            continue
        if stage.enabled:
            unreached.append(stage.name)
        else:
            omissions.setdefault(stage.reason, []).append(stage.name)
    omitted = sum(map(len, omissions.values()))
    summary = (
        f"Analysis stages: {counts['success']} completed · {counts['skipped']} skipped · "
        f"{counts['failed']} failed · {omitted} not selected"
    )
    if counts["reused"]:
        summary += f" · {counts['reused']} reused from checkpoint"
    if unreached:
        summary += f" · {len(unreached)} selected but not reached"
    return summary, active, omissions, unreached


def _warning_groups(records: list[StageExecutionRecord]) -> list[tuple[str, list[str]]]:
    """(stage, warnings) for stages that emitted any, in execution order."""

    return [(r.stage_name, list(r.warnings)) for r in records if r.warnings]


def _uncertain_calls(records: list[StageExecutionRecord]) -> Iterator[tuple[str, str]]:
    """Summarize only marker-vote evidence emitted by this execution."""
    for record in records:
        if record.stage_name != "annotation":
            continue
        methods = record.metrics.get("per_method", [record.metrics])
        if not isinstance(methods, list):
            continue
        for metrics in methods:
            if "pbmc_refinement" not in metrics:
                continue
            summary = metrics["pbmc_refinement"]
            count, total = summary["uncertain"], summary["n_cells"]
            reasons = ", ".join(f"{reason} ({n:,})" for reason, n in summary["reasons"].items())
            text = f"{count:,} of {total:,} cells ({count / max(total, 1):.1%}) are Unassigned."
            if reasons:
                text += " Reasons include " + reasons + "."
            key = metrics["key_added"]
            yield text, f"../results/annotation/{quote(key, safe='')}_evidence.csv"


def render_markdown(
    *,
    config: CellQuorumConfig,
    records: list[StageExecutionRecord],
    plan: PipelinePlan | None = None,
) -> str:
    """Render the run report as Markdown."""

    project = getattr(getattr(config, "project", None), "name", "cellquorum-run")
    summary, active, omissions, unreached = _study_scope(records, plan)
    rows = _status_rows(active)
    lines = [f"# CellQuorum run report — {project}", "", summary, ""]
    if unreached:
        lines.extend(["Selected stages without an execution record: " + ", ".join(unreached), ""])
    for message, link in _uncertain_calls(records):
        lines.extend(
            [
                "## Uncertain cell identities",
                "",
                message,
                "",
                f"[Per-cell calls and RNA evidence]({link})",
                "",
            ]
        )
    for method, reason in getattr(config.run, "method_omissions", {}).items():
        lines.extend([f"Method omitted: `{method}`. {reason}", ""])
    lines.append("| Stage | Status | Backend | Duration (s) | Detail | Warnings |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['stage']} | {r['status']} | {r['backend']} | "
            f"{r['duration_s']} | {_cell(r['detail'])} | {r['n_warnings']} |"
        )
    lines.append("")

    if omissions:
        lines.extend(["## Stages not selected", ""])
        for reason, names in omissions.items():
            stages = ", ".join(f"`{name}`" for name in names)
            lines.append(f"- {stages} — {' '.join(reason.split())}")
        lines.append("")

    # The warnings themselves, not just the count. A real 36-stage run emitted 12
    # warnings — a cluster that lost its velocity to an eigensolver, a kernel that
    # never built, a figure that failed to write — and the report showed only the
    # integer 12, so the reader had to go dig in provenance JSON to learn any of
    # it. The warnings are the most decision-relevant thing the run produces.
    groups = _warning_groups(records)
    total = sum(len(w) for _, w in groups)
    lines.append("## Warnings")
    lines.append("")
    if not groups:
        lines.append("No warnings were emitted.")
    else:
        lines.append(f"{total} warning(s) across {len(groups)} stage(s).")
        lines.append("")
        for stage, warnings in groups:
            lines.append(f"### {stage}")
            lines.append("")
            for warning in warnings:
                lines.append(f"- {' '.join(str(warning).split())}")
            lines.append("")
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
    plan: PipelinePlan | None = None,
    run_root: Path | None = None,
    panels: list[dict] | None = None,
) -> str:
    """Render the run report as a standalone HTML page."""

    project = getattr(getattr(config, "project", None), "name", "cellquorum-run")
    summary, active, omissions, unreached = _study_scope(records, plan)
    rows = _status_rows(active)

    body: list[str] = []
    body.append(f"<h1>CellQuorum run report — {html.escape(str(project))}</h1>")
    body.append(f"<p>{html.escape(summary)}</p>")
    if unreached:
        body.append(
            "<p>Selected stages without an execution record: "
            + html.escape(", ".join(unreached))
            + "</p>"
        )
    for message, link in _uncertain_calls(records):
        body.append(
            f"<h2>Uncertain cell identities</h2><p>{html.escape(message)}</p>"
            f'<p><a href="{link}">Per-cell calls and RNA evidence</a></p>'
        )
    if getattr(config.run, "study_questions", []):
        body.append(
            '<p><a href="publication/index.html">Manuscript '
            "figures, source data and coverage by question</a></p>"
        )
        for method, reason in getattr(config.run, "method_omissions", {}).items():
            body.append(f"<p><code>{html.escape(method)}</code> omitted. {html.escape(reason)}</p>")
        for question, reason in getattr(config.run, "question_omissions", {}).items():
            body.append(f"<p>{html.escape(question.title())} omitted. {html.escape(reason)}</p>")
    if panels:
        for family in dict.fromkeys(row["question"] for row in panels):
            title = {
                "communication": "How might populations communicate?",
                "regulation": "Which regulatory programs differ?",
                "differences": "What changes between conditions?",
                "lineage": "How are cells ordered along the proposed lineage?",
                "populations": "Which populations and states are present?",
                "perturbation": "Which responses are predicted after perturbation?",
            }[family]
            body.append(
                f"<h2>{title}</h2><p><a "
                f'href="publication/{family}.pdf">Assembled figure (PDF)</a> · '
                f'<a href="publication/{family}.svg">Editable figure (SVG)</a></p>'
            )
            for row in panels:
                if row["question"] != family:
                    continue
                key = html.escape(row["panel"], quote=True)
                if row.get("preview"):
                    alt = html.escape(row["title"], quote=True)
                    body.append(
                        f'<img src="publication/{key}.svg" alt="{alt}" '
                        'style="display:block;max-width:100%;height:auto">'
                    )
                body.append(
                    f"<p><strong>{html.escape(row['method'])} "
                    f"· {html.escape(row['title'])}</strong><br>"
                    f'{html.escape(row["legend"])} <a href="publication/{key}.svg">Panel</a> · '
                    f'<a href="publication/{key}.csv">Exact source data</a></p>'
                )
        body.append(
            '<p><a href="publication/index.json">Figure index and resolved settings</a> · '
            '<a href="methods.txt">Methods</a> · <a '
            'href="publication/citations.md">Citations</a></p>'
        )
    if getattr(config.run, "sensitivity_analysis", False) and panels:
        body.append(
            '<p><a href="publication/abundance_sensitivity.csv">Leave-one-donor-out '
            "effects and ranks</a></p>"
        )
    if run_root is not None:
        body.extend(_html_stage_outputs(active, run_root))
    body.append("<h2>Stage execution</h2><div class='scroll'><table>")
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
    body.append("</table></div>")
    if omissions:
        body.append("<details><summary>Stages not selected and why</summary><ul>")
        for reason, names in omissions.items():
            stages = ", ".join(f"<code>{html.escape(name)}</code>" for name in names)
            body.append(f"<li>{stages} — {html.escape(reason)}</li>")
        body.append("</ul></details>")

    groups = _warning_groups(records)
    body.append("<h2>Warnings</h2>")
    if not groups:
        body.append("<p>No warnings were emitted.</p>")
    else:
        total = sum(len(w) for _, w in groups)
        body.append(f"<p>{total} warning(s) across {len(groups)} stage(s).</p>")
        for stage, warnings in groups:
            body.append(f"<h3>{html.escape(str(stage))}</h3><ul>")
            for warning in warnings:
                body.append(f"<li>{html.escape(' '.join(str(warning).split()))}</li>")
            body.append("</ul>")

    body.append("<h2>Methods</h2>")
    body.append(f"<p>{html.escape(render_methods_text(config=config, records=records))}</p>")

    return (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>CellQuorum run report — {html.escape(str(project))}</title>"
        "<style>body{font:1rem/1.5 system-ui,sans-serif;color:#203239;max-width:68rem;"
        "margin:2rem auto;padding:0 1rem}h1,h2{line-height:1.2}h2{margin-top:2rem}"
        "a{color:#075d70;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600}"
        "details{margin:.5rem 0;padding:.45rem .75rem;border-left:3px solid #b5d0ce}"
        ".scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}"
        "th,td{text-align:left;vertical-align:top;padding:.4rem;border-bottom:1px solid #ccd7db}"
        "th{background:#eff4f4}</style></head>"
        f"<body><main>{''.join(body)}</main></body></html>"
    )


def _html_stage_outputs(records: list[StageExecutionRecord], run_root: Path) -> list[str]:
    """Link files reported by the stages of this execution, relative to reports/report.html."""
    sections = []
    root = run_root.resolve()
    for record in records:
        links: dict[str, str] = {}
        for artifact in record.output_artifacts:
            path = (root / artifact.path).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            href = "../" + quote(relative)
            links[relative] = (
                f'<li><a href="{href}">{html.escape(path.name)}</a>'
                f" — {html.escape(artifact.description)}</li>"
            )
        if links:
            stage = html.escape(record.stage_name.replace("_", " "))
            sections.append(
                f"<details><summary>{stage} ({len(links)})</summary>"
                f"<ul>{''.join(links.values())}</ul></details>"
            )
    return ["<h2>Analysis outputs</h2>", *sections] if sections else []


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
    succeeded = [
        r.stage_name
        for r in records
        if (r.status == "success" or r.reused_from_checkpoint) and r.stage_name != "bootstrap"
    ]
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
    plan: PipelinePlan | None = None,
    artifact_manager: ArtifactManager,
    obs: pd.DataFrame | None = None,
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

    panels = []
    if getattr(config.run, "study_questions", ()) or getattr(
        config.run, "sensitivity_analysis", False
    ):
        from cellquorum.core.publication import write_publication_bundle

        panels = write_publication_bundle(config, records, artifact_manager, obs)
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
            render_markdown(config=config, records=records, plan=plan),
            name="run_report_markdown",
            relative_path="reports/report.md",
            kind="markdown",
            description="Human-readable Markdown run report.",
        )

    if getattr(report_config, "html", False):
        artifact_manager.write_text(
            render_html(
                config=config,
                records=records,
                plan=plan,
                run_root=artifact_manager.root,
                panels=panels,
            ),
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
