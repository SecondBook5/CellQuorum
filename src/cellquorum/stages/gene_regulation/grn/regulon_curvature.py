"""Build a TF -> target regulatory graph from pySCENIC regulons, for Ricci
curvature analysis.

The pruned, motif-validated regulon table (not the raw ~2.5M-row GRNBoost2
adjacency output, which carries every candidate TF-target pair before any
biological filtering) is the correct source for a curvature graph: it is
what pySCENIC itself considers a real regulatory relationship. A TF can
appear on more than one row (one per supporting motif); its real regulon is
the union of all of them, each target's edge weight the strongest importance
seen across those rows.
"""

from __future__ import annotations

import ast
from pathlib import Path

import networkx as nx
import pandas as pd


def load_regulons_csv(path: str | Path) -> pd.DataFrame:
    """Load pySCENIC's real regulons.csv shape into flat, named columns.

    pySCENIC writes a 2-row column MultiIndex (('Enrichment', 'AUC'), ...)
    over a 2-level row index (TF, MotifID). A bare ``pd.read_csv`` reads TF
    and MotifID as unnamed data rows rather than real columns -- every
    downstream ``TF``/``TargetGenes`` lookup on that frame would silently
    return the wrong thing (confirmed: it inflated n_tfs on the fibroblast
    cohort from the true 161 to 162, one column-0 value came from the
    embedded header text "TF" itself).
    """
    df = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    df.columns = df.columns.get_level_values(1)
    return df.reset_index()


def _parse_target_genes(cell: object) -> list[tuple[str, float]]:
    """Parse one TargetGenes cell into [(gene, importance), ...].

    TargetGenes is free text written by an external tool (pySCENIC); a
    malformed cell returns an empty list rather than raising, so one bad row
    never takes down the whole graph build.
    """
    try:
        parsed = ast.literal_eval(str(cell))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list | tuple):
        return []
    out = []
    for item in parsed:
        if isinstance(item, list | tuple) and len(item) == 2:
            try:
                out.append((str(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return out


def build_regulon_graph(
    regulons: pd.DataFrame,
    *,
    tf_col: str = "TF",
    target_genes_col: str = "TargetGenes",
    top_k_targets: int = 15,
) -> nx.DiGraph:
    """Build a directed TF -> target graph from a pySCENIC regulons table.

    Args:
        regulons: One row per (TF, motif); must carry tf_col and
            target_genes_col (a string-literal list of (gene, importance)
            tuples, pySCENIC's own regulons.csv shape).
        tf_col: Column naming the transcription factor.
        target_genes_col: Column holding the stringified target-gene list.
        top_k_targets: Cap on targets kept per TF, ranked by the strongest
            importance seen for that target across all of the TF's motif
            rows, descending.

    Returns:
        A directed nx.DiGraph, edges TF -> target weighted by importance
        (max across motif rows for that TF-target pair). Empty when the
        input is empty or every row is unparseable.
    """
    if regulons is None or regulons.empty:
        return nx.DiGraph()

    # TF -> {target: max importance seen}, unioned across motif rows.
    per_tf: dict[str, dict[str, float]] = {}
    for _, row in regulons.iterrows():
        tf = str(row[tf_col])
        for gene, importance in _parse_target_genes(row[target_genes_col]):
            targets = per_tf.setdefault(tf, {})
            if gene not in targets or importance > targets[gene]:
                targets[gene] = importance

    G = nx.DiGraph()
    for tf, targets in per_tf.items():
        if not targets:
            continue
        ranked = sorted(targets.items(), key=lambda kv: kv[1], reverse=True)[:top_k_targets]
        for gene, importance in ranked:
            G.add_edge(tf, gene, weight=importance)

    return G


__all__ = ["build_regulon_graph", "load_regulons_csv"]
