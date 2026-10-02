"""Reference-gallery fixtures for the cell-cell communication figures.

Every figure is built from the same small synthetic cohort (six clusters, a handful of
ligand-receptor pairs, a fixed seed), so the drawing is identical on every run and any
change to the figure code shows up as a pixel difference against the stored baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cellquorum.stages.cell_cell_communication.viz import _plots as plots
from cellquorum.visualization.figstyle import apply_theme

CLUSTERS = ["0", "1", "2", "3", "4", "5"]
LABELS = {
    "0": "Fibroblast",
    "1": "LEC",
    "2": "Macrophage",
    "3": "T cell",
    "4": "Pericyte",
    "5": "Keratinocyte",
}
LIGANDS = ["TGFB1", "CCN2", "VEGFC", "CXCL12", "IL6", "PDGFB", "WNT5A", "FN1"]
RECEPTORS = ["TGFBR1", "FGFR2", "FLT4", "CXCR4", "IL6R", "PDGFRB", "FZD2", "ITGB1"]


def _palette() -> dict[str, str]:
    return plots.celltype_palette(CLUSTERS)


def _lr_table() -> pd.DataFrame:
    """Canonical LIANA-shaped frame: one row per sender, receiver, pair, sample."""
    rng = np.random.default_rng(0)
    rows = []
    for sample in ("P1", "P2"):
        for s in CLUSTERS:
            for t in CLUSTERS:
                if s == t:
                    continue
                for i in rng.choice(len(LIGANDS), size=2, replace=False):
                    rows.append(
                        {
                            "source": s,
                            "target": t,
                            "ligand": LIGANDS[i],
                            "receptor": RECEPTORS[i],
                            "weight": float(rng.uniform(0.1, 1.0)),
                            "sample": sample,
                        }
                    )
    return pd.DataFrame(rows)


def _prioritized() -> pd.DataFrame:
    """MultiNicheNet-shaped prioritization table for the Lymphedema group."""
    lr = _lr_table().drop_duplicates(["source", "target", "ligand"])
    return pd.DataFrame(
        {
            "group": "Lymphedema",
            "sender": lr["source"].to_numpy(),
            "receiver": lr["target"].to_numpy(),
            "ligand": lr["ligand"].to_numpy(),
            "receptor": lr["receptor"].to_numpy(),
            "prioritization_score": lr["weight"].to_numpy(),
        }
    )


def interaction_dotplot() -> Figure:
    return plots.interaction_dotplot(_lr_table(), top_k=8)


def cci_heatmap() -> Figure:
    return plots.cci_heatmap(_lr_table())


def chord_diagram() -> Figure:
    return plots.chord_diagram(_lr_table(), palette=_palette(), top_k=15)


def sankey_flow() -> Figure:
    return plots.sankey_flow(_lr_table(), palette=_palette(), top_k=15)


def circos() -> Figure:
    return plots.cluster_focused_circos(
        _prioritized(), "0", palette=_palette(), top_k=8, cluster_labels=LABELS
    )


def top_lr_lollipop() -> Figure:
    return plots.mnn_top_lr_lollipop(_prioritized(), top_n=15, cluster_labels=LABELS)


def sample_activity() -> Figure:
    pairs = [
        ("TGFB1", "TGFBR1", "0", "3"),
        ("CCN2", "FGFR2", "3", "4"),
        ("VEGFC", "FLT4", "1", "2"),
        ("CXCL12", "CXCR4", "4", "5"),
    ]
    samples = {"P1_LE": "Lymphedema", "P2_LE": "Lymphedema", "P1_N": "Normal", "P2_N": "Normal"}
    rng = np.random.default_rng(5)
    sample_rows, source_rows = [], []
    for k, (lig, rec, sender, receiver) in enumerate(pairs):
        pair_id = f"{lig}_{rec}_X{sender}_X{receiver}"
        for sample, group in samples.items():
            direction = 1.0 if group == "Lymphedema" else -1.0
            sample_rows.append(
                {
                    "id": pair_id,
                    "sample": sample,
                    "group": group,
                    "sender": sender,
                    "receiver": receiver,
                    "ligand": lig,
                    "receptor": rec,
                    "scaled_LR_pb_prod": direction * float(rng.uniform(0.2, 1.2)),
                    "keep_sender_receiver": "Sender & Receiver present",
                }
            )
        for group, direction in (("Lymphedema", 1.0), ("Normal", -1.0)):
            source_rows.append(
                {
                    "id": pair_id,
                    "sender": sender,
                    "receiver": receiver,
                    "ligand": lig,
                    "receptor": rec,
                    "group": group,
                    "activity": 0.03 + 0.01 * direction,
                    "activity_scaled": direction * (1.5 - 0.2 * k),
                    "direction_regulation": "up" if direction > 0 else "down",
                    "scaled_pb_ligand": direction * (1.2 - 0.2 * k),
                    "scaled_pb_receptor": direction * (0.9 - 0.15 * k),
                    "fraction_ligand_group": 0.8 if direction > 0 else 0.3,
                    "fraction_receptor_group": 0.7 if direction > 0 else 0.25,
                    "prioritization_score": 0.95 - 0.1 * k,
                }
            )
    return plots.plot_mnn_sample_activity(
        pd.DataFrame(sample_rows), pd.DataFrame(source_rows), cluster_labels=LABELS
    )


def ligand_activity_bar() -> Figure:
    activities = pd.DataFrame(
        {"test_ligand": LIGANDS, "aupr_corrected": np.linspace(0.12, 0.01, len(LIGANDS))}
    )
    return plots.ligand_activity_bar(activities, top_n=8)


def ligand_target_heatmap() -> Figure:
    rng = np.random.default_rng(6)
    targets = [f"Target{i}" for i in range(10)]
    weights = pd.DataFrame(
        [(lig, tgt, float(rng.uniform(0, 0.1))) for lig in LIGANDS[:5] for tgt in targets],
        columns=["ligand", "target", "weight"],
    )
    return plots.ligand_target_heatmap(weights)


def ligand_receptor_heatmap() -> Figure:
    rng = np.random.default_rng(7)
    weights = pd.DataFrame(
        [(lig, rec, float(rng.uniform(0, 1))) for lig in LIGANDS[:5] for rec in RECEPTORS[:6]],
        columns=["ligand", "receptor", "weight"],
    )
    return plots.ligand_receptor_heatmap(weights)


def sender_ligand_dotplot() -> Figure:
    rng = np.random.default_rng(8)
    expr = pd.DataFrame(
        [
            (sender, lig, float(rng.uniform(0, 1)), float(rng.uniform(0, 3)))
            for sender in CLUSTERS
            for lig in LIGANDS[:5]
        ],
        columns=["sender", "ligand", "fraction_expressing", "mean_expression"],
    )
    return plots.sender_ligand_dotplot(expr)


def curvature_network() -> Figure:
    rng = np.random.default_rng(9)
    edges = pd.DataFrame(
        [
            {
                "source": s,
                "target": t,
                "ricci_curvature": float(rng.uniform(-0.4, 0.3)),
                "weight": float(rng.uniform(0.2, 1.0)),
            }
            for s in CLUSTERS
            for t in CLUSTERS
            if s != t and rng.uniform() < 0.5
        ]
    )
    nodes = pd.DataFrame({"node": CLUSTERS, "ricci_curvature": np.linspace(-0.3, 0.25, 6)})
    return plots.curvature_network(edges, nodes)


def topology_listener() -> Figure:
    rng = np.random.default_rng(10)
    topo = pd.DataFrame(
        {
            "node": CLUSTERS,
            "Listener": rng.normal(size=6),
            "Influencer": rng.uniform(0, 1, 6),
            "Mediator": rng.uniform(0, 1, 6),
            "Pagerank": rng.dirichlet(np.ones(6)),
        }
    )
    return plots.topology_facet_figures(topo, top_k=6)["Listener"]


def tensor_factors() -> Figure:
    rng = np.random.default_rng(11)

    def frame(index: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            rng.dirichlet(np.ones(len(index)), size=3).T,
            index=index,
            columns=["Factor 1", "Factor 2", "Factor 3"],
        )

    pairs = [f"{lig}^{rec}" for lig, rec in zip(LIGANDS, RECEPTORS, strict=True)]
    return plots.plot_tensor_factors(
        frame(["P1_LE", "P2_LE", "P1_N", "P2_N"]),
        frame(CLUSTERS),
        frame(CLUSTERS),
        frame(pairs),
    )


THEME = partial(apply_theme, "notebook")

FIGURES: dict[str, Callable[[], Figure]] = {
    "ccc_interaction_dotplot": interaction_dotplot,
    "ccc_cci_heatmap": cci_heatmap,
    "ccc_chord_diagram": chord_diagram,
    "ccc_sankey_flow": sankey_flow,
    "ccc_circos": circos,
    "ccc_top_lr_lollipop": top_lr_lollipop,
    "ccc_sample_activity": sample_activity,
    "ccc_ligand_activity_bar": ligand_activity_bar,
    "ccc_ligand_target_heatmap": ligand_target_heatmap,
    "ccc_ligand_receptor_heatmap": ligand_receptor_heatmap,
    "ccc_sender_ligand_dotplot": sender_ligand_dotplot,
    "ccc_curvature_network": curvature_network,
    "ccc_topology_listener": topology_listener,
    "ccc_tensor_factors": tensor_factors,
}
