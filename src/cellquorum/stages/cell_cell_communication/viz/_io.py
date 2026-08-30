"""I/O helpers for CCC visualization: discovery + figure saving."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from cellquorum.stages.cell_cell_communication.network._networks import liana_to_canonical
from cellquorum.core.stage import StageArtifact
from cellquorum.visualization.figstyle import apply_cellquorum_theme

# ─── Input discovery ───────────────────────────────────────────────────────


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    return df if not df.empty else None


def load_canonical_lr_sources(
    results_dir: Path, uns: dict | None
) -> list[tuple[str, pd.DataFrame]]:
    """Return (label, canonical-LR df) for each present source, in fixed order."""
    out: list[tuple[str, pd.DataFrame]] = []

    liana_df: pd.DataFrame | None = None
    if uns is not None and "liana_res" in uns:
        try:
            res = uns["liana_res"]
            if isinstance(res, pd.DataFrame) and not res.empty:
                canon, _notes = liana_to_canonical(res)
                if not canon.empty:
                    liana_df = canon
        except Exception:  # noqa: BLE001
            liana_df = None
    if liana_df is None:
        liana_df = _read_csv(results_dir / "cell_cell_communication" / "liana_ranks.csv")
    if liana_df is not None:
        out.append(("liana", liana_df))

    mnn = _read_csv(results_dir / "mnn_canonical_lr.csv")
    if mnn is not None:
        out.append(("multinichenet", mnn))

    nn = _read_csv(results_dir / "nichenet_canonical_lr.csv")
    if nn is not None:
        out.append(("nichenet", nn))

    return out


def load_topology(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Read topology CSVs from results_dir/ccc_network. Missing keys omitted."""
    net = results_dir / "ccc_network"
    mapping = {
        "cci": "topology_cci.csv",
        "gci": "topology_gci.csv",
        "comparative_cci": "comparative_cci.csv",
        "comparative_gci": "comparative_gci.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, fname in mapping.items():
        df = _read_csv(net / fname)
        if df is not None:
            out[key] = df
    return out


def load_curvature(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Read curvature CSVs from results_dir/ccc_network. Missing keys omitted."""
    net = results_dir / "ccc_network"
    mapping = {
        "cci_edges": "curvature_cci_edges.csv",
        "cci_nodes": "curvature_cci_nodes.csv",
        "gci_edges": "curvature_gci_edges.csv",
        "gci_nodes": "curvature_gci_nodes.csv",
        "differential_cci": "differential_curvature_cci.csv",
        "differential_gci": "differential_curvature_gci.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, fname in mapping.items():
        df = _read_csv(net / fname)
        if df is not None:
            out[key] = df
    return out


# ─── Figure saving ─────────────────────────────────────────────────────────


def apply_theme() -> None:
    """Apply the house theme plus vector-font overrides."""
    apply_cellquorum_theme()
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def save_figure(
    fig: Figure,
    out_dir: str | Path,
    stem: str,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Write ``fig`` to ``out_dir/stem.<fmt>`` for each format, then close it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def figure_artifacts(paths: list[Path], *, name: str, description: str) -> list[StageArtifact]:
    """Wrap saved figure paths as ``kind="figure"`` stage artifacts."""
    return [
        StageArtifact(name=name, path=path, kind="figure", description=description)
        for path in paths
    ]


__all__ = [
    "load_canonical_lr_sources",
    "load_topology",
    "load_curvature",
    "apply_theme",
    "save_figure",
    "figure_artifacts",
]
