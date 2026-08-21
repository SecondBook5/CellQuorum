#!/usr/bin/env python
"""One-shot generator for the README pipeline diagram.

Reads the live stage registry, renders a Graphviz DOT DAG, and writes
``docs/assets/pipeline.dot`` + ``docs/assets/pipeline.svg`` (via ``dot -Tsvg``).
This is a *dev-time* script — the engine does not depend on it and it is not part
of the CLI. Regenerate the figure only when the stage set changes:

    python docs/assets/gen_pipeline_diagram.py

Requires the Graphviz ``dot`` binary on PATH (Debian/Ubuntu: ``apt install graphviz``).

The figure is generated from code (stages, order, implemented-vs-planned, and the
selectable methods per stage all come from the registry), so a regenerated figure
always matches the engine.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellquorum.core.stage_catalog import StageSpec

# Middle dot separating method names in a sub-line (literal UTF-8; the .dot/.svg
# are UTF-8 and Graphviz HTML-like labels accept it directly).
_MIDDOT = " · "

# Ink tokens — text never wears a series colour; identity comes from the border.
_INK = "#0b0b0b"  # primary ink for implemented stage names
_SUBTLE_INK = "#52514e"  # secondary ink for method sub-lines
_PLANNED_STROKE = "#898781"  # dashed grey border for planned stages
_PLANNED_FILL = "#f4f3f1"  # faint grey fill for planned stages
_PLANNED_INK = "#52514e"  # muted ink for planned stage text
_IO_FILL = "#0b0b0b"  # dark source/sink nodes
_IO_INK = "#ffffff"
_SURFACE = "#ffffff"  # explicit light surface (renders cleanly in GitHub dark mode)


@dataclass(frozen=True)
class Phase:
    """One coloured lane of the pipeline diagram (a contiguous ``order`` band)."""

    key: str
    title: str
    max_order: int  # holds stages with order <= max_order and > the previous band
    stroke: str  # CVD-validated categorical hue (borders + lane label)
    fill: str  # light tint of the hue (lane background)


# Seven contiguous order-bands → a monotonic spine with no back-edges. Strokes are
# the CVD-validated 7-hue set (validate_palette.js PASS, light + dark).
PHASES: tuple[Phase, ...] = (
    Phase("preprocessing", "1 · Preprocessing", 50, "#2a78d6", "#eaf2fc"),
    Phase("integration", "2 · Integration and clustering", 100, "#eb6834", "#fdeee7"),
    Phase("annotation", "3 · Annotation and identity", 160, "#1baf7a", "#e6f7f1"),
    Phase("embeddings", "4 · State and embeddings", 200, "#eda100", "#fdf3dd"),
    Phase("differential", "5 · Differential analysis", 250, "#e87ba4", "#fcecf2"),
    Phase("regulation", "6 · Gene regulation", 290, "#008300", "#e3f2e3"),
    Phase("communication", "7 · Communication and trajectory", 10_000, "#4a3aa7", "#eae7f6"),
)

# Curated one-line descriptors for backbone stages that expose a single
# implementation (so register no *selectable* methods). Short + accurate to
# current behaviour; every other sub-line derives from the method registry.
_STAGE_DESCRIPTORS: dict[str, str] = {
    "ambient_correction": "SoupX · R",
    "qc": "MAD · doublets · cell-cycle",
    "preprocessing": "normalize · log1p · GPU",
    "subclustering": "recursive Leiden · sc-SHC",
    "adjudication": "label reconciliation",
    "annotation_consensus": "cross-method consensus",
    "population_identity": "evidence-ranked identity",
}


def phase_of(order: int) -> Phase:
    """Return the phase whose contiguous order-band contains ``order``."""
    for phase in PHASES:
        if order <= phase.max_order:
            return phase
    return PHASES[-1]


def _methods_for(category: str | None) -> list[str]:
    """Registered method names for a stage category (``[]`` when none)."""
    if category is None:
        return []
    from cellquorum.methods.registry import METHOD_REGISTRY

    return list(METHOD_REGISTRY.names(category))


def _sub_line(spec: StageSpec) -> str:
    """Muted second line for a stage card (methods / descriptor / planned)."""
    methods = _methods_for(spec.category)
    if methods:
        return _MIDDOT.join(methods)
    if not spec.is_implemented:
        return "planned"
    return _STAGE_DESCRIPTORS.get(spec.name, "")


def _node_label(spec: StageSpec) -> str:
    """Graphviz HTML-like label: bold stage name over a muted sub-line."""
    sub = _sub_line(spec)
    ink = _SUBTLE_INK if spec.is_implemented else _PLANNED_INK
    if sub:
        return f'<<B>{spec.name}</B><BR/><FONT POINT-SIZE="9" COLOR="{ink}">{sub}</FONT>>'
    return f"<<B>{spec.name}</B>>"


def _node_stmt(spec: StageSpec) -> str:
    """One DOT node statement for a stage, styled by implemented/planned."""
    label = _node_label(spec)
    if spec.is_implemented:
        stroke = phase_of(spec.order).stroke
        attrs = f'label={label}, color="{stroke}", fontcolor="{_INK}", penwidth=1.8'
    else:
        attrs = (
            f'label={label}, style="rounded,filled,dashed", '
            f'color="{_PLANNED_STROKE}", fillcolor="{_PLANNED_FILL}", '
            f'fontcolor="{_PLANNED_INK}", penwidth=1.5'
        )
    return f"    {spec.name} [{attrs}];"


def render_dot(specs: list[StageSpec]) -> str:
    """Render the stage catalog as a Graphviz DOT pipeline DAG (deterministic)."""
    specs = sorted(specs, key=lambda s: s.order)
    by_phase: dict[str, list] = {phase.key: [] for phase in PHASES}
    for spec in specs:
        by_phase[phase_of(spec.order).key].append(spec)

    dot = _MIDDOT.strip()
    lines: list[str] = [
        "digraph cellquorum_pipeline {",
        "    // Generated from the stage registry by docs/assets/gen_pipeline_diagram.py.",
        "    // Do not edit by hand — regenerate when the stage set changes.",
        f'    graph [rankdir=TB, fontname="Helvetica", bgcolor="{_SURFACE}", '
        "nodesep=0.28, ranksep=0.42, splines=true];",
        '    node [shape=box, style="rounded,filled", fillcolor="#ffffff", '
        'fontname="Helvetica", fontsize=12, margin="0.16,0.09"];',
        '    edge [color="#898781", penwidth=1.3, arrowsize=0.75];',
        "",
        "    // Source + sink (run inputs and the run directory).",
        f"    IN [label=<<B>config.yaml</B><BR/>AnnData / 10x matrices>, "
        f'fillcolor="{_IO_FILL}", color="{_IO_FILL}", fontcolor="{_IO_INK}", penwidth=1];',
        f"    OUT [label=<<B>Run directory</B>"
        f'<BR/><FONT POINT-SIZE="9">figures {dot} results {dot} objects {dot} provenance</FONT>>, '
        f'fillcolor="{_IO_FILL}", color="{_IO_FILL}", fontcolor="{_IO_INK}", penwidth=1];',
    ]

    # One tinted, labelled cluster per phase; stages chained top-to-bottom inside.
    for phase in PHASES:
        members = by_phase[phase.key]
        lines.append("")
        lines.append(f"    subgraph cluster_{phase.key} {{")
        lines.append(f'        label="{phase.title}";')
        lines.append('        labeljust="l";')
        lines.append('        labelloc="t";')
        lines.append('        fontname="Helvetica-Bold";')
        lines.append("        fontsize=13;")
        lines.append(f'        fontcolor="{phase.stroke}";')
        lines.append('        style="rounded,filled";')
        lines.append(f'        color="{phase.stroke}";')
        lines.append(f'        fillcolor="{phase.fill}";')
        lines.append("        penwidth=1.6;")
        lines.append("        margin=14;")
        for spec in members:
            lines.append("    " + _node_stmt(spec).lstrip())
        if len(members) > 1:
            chain = " -> ".join(spec.name for spec in members)
            lines.append(f"        {chain};")
        lines.append("    }")

    # Monotonic spine: IN -> first stage; last-of-phase -> first-of-next; last -> OUT.
    firsts = [by_phase[phase.key][0].name for phase in PHASES if by_phase[phase.key]]
    lasts = [by_phase[phase.key][-1].name for phase in PHASES if by_phase[phase.key]]
    lines.append("")
    lines.append("    // Spine (canonical order across the lanes).")
    lines.append(f"    IN -> {firsts[0]};")
    for last, nxt in zip(lasts, firsts[1:], strict=False):
        lines.append(f"    {last} -> {nxt};")
    lines.append(f"    {lasts[-1]} -> OUT;")

    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write pipeline.dot and render pipeline.svg next to this script."""
    # Import here so a missing engine import fails loudly with a clear message.
    import cellquorum.core.stages as stages_mod

    here = Path(__file__).resolve().parent
    dot_path = here / "pipeline.dot"
    svg_path = here / "pipeline.svg"

    dot_src = render_dot(stages_mod.all_stage_specs())
    dot_path.write_text(dot_src, encoding="utf-8")
    print(f"wrote {dot_path} ({len(dot_src)} bytes)")

    try:
        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=True,
        )
    except FileNotFoundError:
        print("ERROR: `dot` not found on PATH — install graphviz to render the SVG.")
        return 1
    print(f"wrote {svg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
