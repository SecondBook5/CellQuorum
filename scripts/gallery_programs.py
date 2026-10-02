"""Reference-gallery fixtures for the module-remodeling and multicellular-program figures."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from _gallery_io import via_files
from matplotlib.figure import Figure

from cellquorum.stages.comparative.module_remodeling import mr_figures
from cellquorum.stages.comparative.multicellular_programs.mcp_figures import plot_mcp_summary
from cellquorum.visualization.figstyle import apply_theme

GROUPS = ["Papillary", "Reticular", "Perivascular", "Mechanosensory"]
PROGRAMS = {
    "Matrix": ["collagen_synthesis", "ecm_remodeling", "crosslinking"],
    "Signalling": ["tgfb_response", "wnt_signalling", "ifn_response"],
}


def _effects() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        [
            {
                "group": group,
                "program": program,
                "effect": float(rng.normal(0, 0.6)),
                "fdr": float(np.power(10.0, -rng.uniform(0, 5))),
                "method": "mixed_model",
            }
            for programs in PROGRAMS.values()
            for program in programs
            for group in GROUPS
        ]
    )


def module_dotgrid() -> Path:
    return via_files(
        lambda out: mr_figures.plot_module_dotgrid(
            _effects(),
            out_dir=out,
            categories=PROGRAMS,
            group_order=GROUPS,
            group_label="Fibroblast subtype",
            case="Lymphedema",
            control="Normal",
            title="Synthetic module remodeling",
        )
    )()


def contrast_index() -> Path:
    rng = np.random.default_rng(1)
    values = pd.DataFrame(
        [
            {"group": g, "condition": c, "index": float(v)}
            for g in GROUPS
            for c, shift in (("Normal", -0.2), ("Lymphedema", 0.3))
            for v in rng.normal(shift, 0.4, 60)
        ]
    )
    return via_files(
        lambda out: mr_figures.plot_contrast_index(
            values,
            out_dir=out,
            group_col="group",
            condition_col="condition",
            index_col="index",
            group_label="Fibroblast subtype",
            index_label="Signed contrast index",
            case="Lymphedema",
            control="Normal",
            title="Synthetic contrast index",
        )
    )()


def permanova() -> Path:
    table = pd.DataFrame(
        {
            "group": GROUPS,
            "R2": [0.42, 0.18, 0.31, 0.07],
            "p_value": [0.004, 0.2, 0.03, 0.6],
            "fdr": [0.016, 0.27, 0.06, 0.6],
            "n_samples": [12, 12, 11, 9],
        }
    )
    return via_files(
        lambda out: mr_figures.plot_permanova(
            table, out_dir=out, group_label="Fibroblast subtype", title="Synthetic PERMANOVA"
        )
    )()


def mcp_summary() -> Path:
    rng = np.random.default_rng(2)
    cell_types = ["Fibroblast", "LEC", "Macrophage"]
    programs = pd.DataFrame(
        [
            {
                "program": f"MCP{p}",
                "cell_type": ct,
                "gene": f"G{p}_{ct[:2]}_{i}",
                "loading": float(rng.normal(0, 0.5)),
                "direction": "up" if rng.uniform() > 0.4 else "down",
            }
            for p in range(1, 4)
            for ct in cell_types
            for i in range(5)
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "cell_id": f"c{p}_{s}_{ct}",
                "sample": f"s{s}",
                "cell_type": ct,
                "program": f"MCP{p}",
                "score": float(rng.normal(0, 1)),
            }
            for p in range(1, 4)
            for s in range(6)
            for ct in cell_types
        ]
    )
    support = pd.DataFrame(
        {
            "program": ["MCP1", "MCP2", "MCP3"],
            "n_donors": [5, 4, 1],
            "donor_fraction": [1.0, 0.8, 0.2],
            "supported": [True, True, False],
        }
    )
    return via_files(
        lambda out: [
            plot_mcp_summary(
                programs, scores, support, cell_type_col_values=cell_types, out_dir=out
            )
        ]
    )()


THEME = partial(apply_theme, "compact")

FIGURES: dict[str, Callable[[], Figure | Path]] = {
    "programs_module_dotgrid": module_dotgrid,
    "programs_contrast_index": contrast_index,
    "programs_permanova": permanova,
    "programs_mcp_summary": mcp_summary,
}
