"""I/O helpers for CCC visualization: discovery + figure saving."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.stages.cell_cell_communication.network._networks import liana_to_canonical
from cellquorum.visualization.figio import figure_artifacts, save_figure

# ─── Input discovery ───────────────────────────────────────────────────────


def relabel_clusters(
    df: pd.DataFrame, cols: list[str], labels: dict[str, str] | None
) -> pd.DataFrame:
    """Display-only {raw cluster code -> name} override on the given columns.

    Real ask: every CCC figure (dotplot, chord, sankey, network, topology)
    showed bare numeric subcluster codes with no way to tell what they mean
    -- same pattern trajectory_viz's fate_labels already solves. A code with
    no entry in ``labels`` is left unchanged, and ``labels`` itself is never
    required: the engine never defaults or invents biology, a caller opts in
    by passing a project-declared mapping.
    """
    if not labels:
        return df
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).map(lambda v: labels.get(v, v))
    return df


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


def load_nichenet(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Read NicheNet's own-format outputs (not the canonical LR schema).

    nichenet_method.py writes three tables the generic canonical-LR pipeline
    never touches: the ligand-activity ranking, the ligand->target weight
    matrix, and per-sender ligand expression. Each key is independently
    present/absent (a run can have activities without weights, e.g. if no
    ligand had any weighted target links), so this returns whichever tables
    exist rather than tensor_factors' all-or-nothing dict.
    """
    mapping = {
        "activities": "nichenet_activities.csv",
        "weights": "nichenet_ligand_target_weights.csv",
        "sender_expression": "nichenet_sender_expression.csv",
        "receptor_weights": "nichenet_ligand_receptor_weights.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, fname in mapping.items():
        df = _read_csv(Path(results_dir) / fname)
        if df is not None:
            out[key] = df
    return out


def load_mnn_prioritization(results_dir: Path) -> pd.DataFrame | None:
    """Read MultiNicheNet's native prioritization table (not the canonical
    LR schema): group, sender, receiver, ligand, receptor,
    prioritization_score, etc. -- the columns a top-LR-interactions figure
    needs and the canonical projection (mnn_canonical_lr.csv) doesn't carry.
    """
    return _read_csv(Path(results_dir) / "mnn_prioritization.csv")


def load_mnn_sample_activity(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Read MultiNicheNet's per-sample and pre-trim prioritization tables
    (sample_prioritization_tbl, group_prioritization_table_source) --
    together, since plot_mnn_sample_activity needs both or neither.
    """
    sample_df = _read_csv(Path(results_dir) / "mnn_sample_prioritization.csv")
    source_df = _read_csv(Path(results_dir) / "mnn_group_prioritization_source.csv")
    if sample_df is None or source_df is None:
        return None
    return sample_df, source_df


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


def load_tensor_factors(results_dir: Path) -> dict[str, pd.DataFrame] | None:
    """Read the 4 tensor-cell2cell factor-loading CSVs, or None if incomplete.

    tensor_c2c_method.py writes contexts/senders/receivers/lr_pairs as four
    separate tables (each: index=entity, columns=Factor 1..N) under
    results_dir/cell_cell_communication/. A factor decomposition plot needs
    all four to say anything, so this returns None rather than a partial
    dict when even one is missing.
    """
    ccc = results_dir / "cell_cell_communication"
    mapping = {
        "contexts": "tensor_factors_contexts.csv",
        "senders": "tensor_factors_senders.csv",
        "receivers": "tensor_factors_receivers.csv",
        "lr_pairs": "tensor_factors_lr_pairs.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, fname in mapping.items():
        df = _read_csv(ccc / fname)
        if df is None:
            return None
        out[key] = df.set_index(df.columns[0])
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


# save_figure/figure_artifacts are re-exported, not redefined: the local copy was
# a bare savefig loop that left truncated files behind and abandoned the remaining
# formats when one raised mid-write. See visualization.figio.


__all__ = [
    "load_canonical_lr_sources",
    "load_mnn_prioritization",
    "load_mnn_sample_activity",
    "load_nichenet",
    "load_topology",
    "load_curvature",
    "load_tensor_factors",
    "save_figure",
    "figure_artifacts",
]
