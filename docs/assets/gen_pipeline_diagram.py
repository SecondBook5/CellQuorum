#!/usr/bin/env python
"""One-shot generator for the README pipeline diagram.

Reads the live stage registry, renders a Graphviz DOT fan-out DAG, and writes
``docs/assets/pipeline.dot`` + ``docs/assets/pipeline.svg`` (via ``dot -Tsvg``).
This is a *dev-time* script — the engine does not depend on it and it is not part
of the CLI. Regenerate the figure only when the stage set changes:

    python docs/assets/gen_pipeline_diagram.py

Requires the Graphviz ``dot`` binary on PATH (Debian/Ubuntu: ``apt install graphviz``).

The figure is a *fan-out*: a shared preprocessing → integration → annotation
backbone produces one annotated object, which the four downstream analysis
families (state, differential, regulation, communication) each consume in
parallel before converging on the run directory. Stages, order, the
implemented-vs-planned split, and the selectable methods per stage all come from
the registry, so a regenerated figure always matches the engine.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellquorum.core.stage_catalog import StageSpec

# Middle dot separating method names in a detail line (literal UTF-8; the .dot/.svg
# are UTF-8 and Graphviz HTML-like labels accept it directly).
_MIDDOT = " · "

# Ink tokens — text never wears a series colour; identity comes from the border.
_INK = "#0b0b0b"  # primary ink for implemented stage names
_SUBTLE_INK = "#52514e"  # secondary ink for method detail lines
_PLANNED_INK = "#8a8880"  # muted ink for planned (reserved) stage rows
_IO_FILL = "#0b0b0b"  # dark source/sink nodes
_IO_INK = "#ffffff"
_SURFACE = "#ffffff"  # explicit light surface (renders cleanly in GitHub dark mode)


@dataclass(frozen=True)
class Phase:
    """One coloured node of the fan-out DAG (a contiguous ``order`` band)."""

    key: str
    title: str
    max_order: int  # holds stages with order <= max_order and > the previous band
    stroke: str  # CVD-validated categorical hue (border + title)


# Seven contiguous order-bands. The first three are the shared backbone; the last
# four fan out from the annotated object. Strokes are the CVD-validated 7-hue set
# (validate_palette.js PASS, light + dark).
PHASES: tuple[Phase, ...] = (
    Phase("preprocessing", "1 · Preprocessing", 50, "#2a78d6"),
    Phase("integration", "2 · Integration and clustering", 100, "#eb6834"),
    Phase("annotation", "3 · Annotation and identity", 160, "#1baf7a"),
    Phase("embeddings", "4 · State and embeddings", 200, "#eda100"),
    Phase("differential", "5 · Differential analysis", 250, "#e87ba4"),
    Phase("regulation", "6 · Gene regulation", 290, "#008300"),
    Phase("communication", "7 · Communication and trajectory", 10_000, "#4a3aa7"),
)

# The backbone is the first three phases; the rest fan out in parallel.
_BACKBONE = PHASES[:3]
_FAN = PHASES[3:]

# Registry method ids are terse (``tensor_c2c``, ``sccoda``); the figure shows the
# tools' published names so the marquee backends read at a glance.
_PRETTY: dict[str, str] = {
    "pca": "PCA",
    "seurat": "Seurat",
    "seurat_v3": "Seurat v3",
    "pearson_residuals": "Pearson residuals",
    "harmony": "Harmony",
    "scvi": "scVI",
    "scanvi": "scANVI",
    "leiden": "Leiden",
    "marker_vote": "marker-vote",
    "celltypist": "CellTypist",
    "passthrough": "passthrough",
    "scarches": "scArches",
    "scdiagnostics": "scDiagnostics",
    "scib_benchmark": "scIB metrics",
    "umap": "UMAP",
    "phate": "PHATE",
    "paga": "PAGA",
    "categorical_embedding": "categorical",
    "continuous_overlay": "MAGIC overlay",
    "pseudobulk_edger": "pseudobulk edgeR",
    "propeller": "propeller",
    "milo": "Milo",
    "sccoda": "scCODA",
    "proportion_ttest": "arcsin-sqrt t-test",
    "gsea": "GSEA",
    "ora": "ORA",
    "gsva": "GSVA",
    "activity": "decoupler activity",
    "hdwgcna": "hdWGCNA",
    "pyscenic": "pySCENIC",
    "celloracle": "CellOracle",
    "velocity": "scVelo",
    "cellrank": "CellRank",
    "dpt": "DPT",
    "palantir": "Palantir",
    "cytotrace": "CytoTRACE",
    "liana": "LIANA",
    "tensor_c2c": "Tensor-cell2cell",
    "multinichenet": "MultiNicheNet",
    "nichenet": "NicheNet",
    "dialogue": "DIALOGUE",
    "topology": "topology",
    "ricci": "Ollivier-Ricci",
}

# Curated detail lines (take precedence over the registry) for stages that either
# register no *selectable* method — single-implementation backbone steps — or whose
# many viz variants collapse to a compact phrase. Backbone GPU steps name the
# rapids-singlecell backend explicitly. Every other detail derives from the registry.
_CURATED: dict[str, str] = {
    "ambient_correction": "SoupX",
    "qc": "MAD · Scrublet · scDblFinder · cell-cycle",
    "preprocessing": "PFlog1pPF · rapids-singlecell (GPU)",
    "dimensionality": "PCA · rapids-singlecell (GPU)",
    "clustering": "Leiden · rapids-singlecell (GPU)",
    "subclustering": "recursive Leiden · sc-SHC",
    "adjudication": "label reconciliation",
    "annotation_consensus": "cross-method consensus",
    "population_identity": "evidence-ranked identity",
    "de_viz": "volcano",
    "enrichment_viz": "GSEA · ORA · GSVA plots",
    "ccc_viz": "dotplot · chord · Sankey · curvature",
    "trajectory_viz": "pseudotime · fate · drivers · gene trends",
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


def _stage_detail(spec: StageSpec) -> str:
    """The muted method/tool line for a stage row (curated → registry → empty)."""
    if spec.name in _CURATED:
        return _CURATED[spec.name]
    methods = _methods_for(spec.category)
    if methods:
        return _MIDDOT.join(_PRETTY.get(m, m) for m in methods)
    return ""  # planned rows carry their marker from the label builder


def _text_on(hex_color: str) -> str:
    """Black or white ink, whichever has more WCAG contrast on a filled hue."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    on_white = 1.05 / (lum + 0.05)  # contrast if white ink
    on_black = (lum + 0.05) / 0.05  # contrast if black ink
    return "#ffffff" if on_white >= on_black else _INK


def _phase_node_label(phase: Phase, members: list[StageSpec]) -> str:
    """HTML-like label: a coloured phase title over one left-aligned row per stage."""
    rows = [f'<B><FONT POINT-SIZE="15" COLOR="{phase.stroke}">{phase.title}</FONT></B>']
    for spec in members:
        detail = _stage_detail(spec)
        if spec.is_implemented:
            name = f'<FONT COLOR="{_INK}"><B>{spec.name}</B></FONT>'
            tail = f'&#160;&#160;<FONT POINT-SIZE="9.5" COLOR="{_SUBTLE_INK}">{detail}</FONT>'
            rows.append(name + (tail if detail else ""))
        else:
            rows.append(
                f'<FONT COLOR="{_PLANNED_INK}">{spec.name}'
                f'<FONT POINT-SIZE="9.5">&#160;&#160;· planned</FONT></FONT>'
            )
    body = '<BR ALIGN="LEFT"/>'.join(rows) + '<BR ALIGN="LEFT"/>'
    return f"<{body}>"


def _phase_node_stmt(phase: Phase, members: list[StageSpec]) -> str:
    """One DOT node statement for a phase card (white fill, phase-coloured border)."""
    label = _phase_node_label(phase, members)
    return (
        f'    P_{phase.key} [label={label}, color="{phase.stroke}", '
        f'fillcolor="#ffffff", penwidth=1.9, margin="0.20,0.13"];'
    )


def render_dot(specs: list[StageSpec]) -> str:
    """Render the stage catalog as a landscape fan-out DAG (deterministic).

    A shared backbone reads left to right — ``config.yaml`` → Preprocessing →
    Integration → Annotation — producing one annotated object. From there the four
    downstream analysis families (state, differential, regulation, communication)
    fan out in parallel and converge on the run directory. Each phase is a
    colour-coded card listing its stages and their config-selectable methods;
    reserved (not-yet-implemented) stages appear as muted rows.
    """
    specs = sorted(specs, key=lambda s: s.order)
    by_phase: dict[str, list[StageSpec]] = {phase.key: [] for phase in PHASES}
    for spec in specs:
        by_phase[phase_of(spec.order).key].append(spec)

    dot = _MIDDOT.strip()
    lines: list[str] = [
        "digraph cellquorum_pipeline {",
        "    // Generated from the stage registry by docs/assets/gen_pipeline_diagram.py.",
        "    // Do not edit by hand — regenerate when the stage set changes.",
        f'    graph [rankdir=LR, fontname="Helvetica", bgcolor="{_SURFACE}", '
        "nodesep=0.40, ranksep=0.85, splines=true];",
        '    node [shape=box, style="rounded,filled", fillcolor="#ffffff", '
        'fontname="Helvetica", fontsize=11];',
        "",
        "    // Source + sink (run inputs and the run directory).",
        f"    IN [label=<<B>config.yaml</B><BR/>AnnData / 10x matrices>, "
        f'fillcolor="{_IO_FILL}", color="{_IO_FILL}", fontcolor="{_IO_INK}", penwidth=1];',
        f"    OUT [label=<<B>Run directory</B>"
        f'<BR/><FONT POINT-SIZE="9">figures {dot} results {dot} objects {dot} provenance</FONT>>, '
        f'fillcolor="{_IO_FILL}", color="{_IO_FILL}", fontcolor="{_IO_INK}", penwidth=1];',
        "",
        "    // Phase cards (white fill, phase-coloured border).",
    ]

    for phase in PHASES:
        lines.append(_phase_node_stmt(phase, by_phase[phase.key]))

    # Backbone spine: config -> preprocessing -> integration -> annotation. A
    # heavier, darker edge than the fan so the shared trunk reads first.
    spine = ["IN", *[f"P_{phase.key}" for phase in _BACKBONE]]
    lines.append("")
    lines.append("    // Shared backbone (heavier spine).")
    lines.append("    " + " -> ".join(spine) + ' [color="#5f5e5b", penwidth=2.0, arrowsize=0.9];')

    # Fan-out from the annotated object into the four analysis families, each
    # converging on the run directory. Lighter curved grey edges.
    hub = f"P_{_BACKBONE[-1].key}"
    lines.append("")
    lines.append("    // Fan-out: the annotated object feeds each analysis family.")
    for phase in _FAN:
        lines.append(f'    {hub} -> P_{phase.key} [color="#9a988f", penwidth=1.5, arrowsize=0.8];')
    lines.append("")
    lines.append("    // Fan-in: every family converges on the run directory.")
    for phase in _FAN:
        lines.append(f'    P_{phase.key} -> OUT [color="#9a988f", penwidth=1.5, arrowsize=0.8];')

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
