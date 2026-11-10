"""Prepare the ranked LIANA routes for the CellQuorum ligand–receptor ribbon panel."""

import json
from collections import Counter
from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.core.anndata_access import obs_frame
from cellquorum.core.stage import StageArtifact
from cellquorum.visualization.communication import ligand_receptor_chord
from cellquorum.visualization.figio import figure_artifacts, save_figure
from cellquorum.visualization.figstyle import muted_palette_colors


def marker_display_labels(adata: ad.AnnData, group_key: str, marker_key: str) -> dict[str, str]:
    """Keep each inferred cluster distinct, including clusters with mixed marker calls."""
    if group_key == marker_key or group_key not in adata.obs or marker_key not in adata.obs:
        return {}
    display = {}
    for cluster, cells in obs_frame(adata).groupby(group_key, observed=True):
        labels = cells[marker_key].dropna().unique()
        name = (
            str(labels[0])
            if len(labels) == 1 and cells[marker_key].notna().all()
            else "Mixed calls"
        )
        display[str(cluster)] = f"{name} (C{cluster})"
    return display


def population_palette(names: list[str]) -> dict[str, str]:
    """Give clusters with the same marker identity one hue and separate labelled sectors."""
    families = {}
    for name in names:
        base, sep, suffix = name.rpartition(" (C")
        families[name] = base if sep and suffix.endswith(")") and suffix[:-1].isdecimal() else name
    counts = Counter(families.values())
    ordered = sorted(counts, key=lambda name: (-counts[name], name))
    colors = muted_palette_colors(max(4, len(ordered)))
    # Start with blue, wine and green so abundant families have distinct hues.
    colors[:4] = [colors[i] for i in (0, 2, 3, 1)]
    hues = dict(zip(ordered, colors[: len(ordered)], strict=True))
    return {name: hues[families[name]] for name in sorted(families)}


def liana_receiver_panel(
    ranks: pd.DataFrame,
    labels: dict[str, str],
    *,
    figures: Path,
    results: Path,
    formats: tuple[str, ...],
    dpi: int,
    top_k: int,
    palette: dict[str, str],
) -> list[StageArtifact]:
    """Show the best cross-population ligand–receptor routes into the top-ranked receiver."""
    keys = ["source", "target", "ligand", "receptor"]
    edges = ranks.loc[ranks.source.ne(ranks.target), [*keys, "sample", "weight"]].dropna()
    if top_k < 1:
        raise ValueError("chord_viz top_k must be at least 1")
    if edges.empty:
        return []
    by_library = edges.groupby([*keys, "sample"], observed=True)["weight"].median().reset_index()
    routes = (
        by_library.groupby(keys, observed=True, as_index=False)
        .agg(weight=("weight", "median"), n_libraries=("sample", "nunique"))
        .sort_values(["weight", *keys], ascending=[False, True, True, True, True], kind="stable")
    )
    routes = routes.loc[routes.weight.gt(0)]
    if routes.empty:
        return []
    receiver = routes.iloc[0].target
    shown = routes.loc[routes.target.eq(receiver)].head(min(8, top_k)).copy()
    libraries = by_library.merge(shown[keys], on=keys, validate="many_to_one").groupby(
        keys, observed=True
    )
    members = (
        libraries["sample"].agg(lambda ids: json.dumps(sorted(ids.astype(str)))).rename("libraries")
    )
    shown = shown.merge(members, on=keys, validate="one_to_one")
    shown["sender_label"] = [labels.get(str(k), str(k)) for k in shown.source]
    shown["receiver_label"] = [labels.get(str(k), str(k)) for k in shown.target]
    title = f"Predicted signaling to {shown.receiver_label.iloc[0]}"
    n_libraries = edges["sample"].nunique()
    library_count = f"{n_libraries} {'library' if n_libraries == 1 else 'libraries'}"
    note = (
        f"LIANA  ·  {len(shown)} ranked routes  ·  {library_count}  ·  "
        "Width = median (1 − magnitude rank)"
    )
    fig = ligand_receptor_chord(
        shown,
        sender_col="sender_label",
        receiver_col="receiver_label",
        sender_palette=palette,
        receiver_palette=palette,
        ribbon_color="sender",
        aligned_labels=True,
        title=title,
        note=note,
    )
    stem = "liana_receiver_ligand_receptor"
    paths = save_figure(fig, figures, stem, formats=formats, dpi=dpi)
    results.mkdir(parents=True, exist_ok=True)
    table = results / f"{stem}.csv"
    shown.to_csv(table, index=False)
    legend = (
        f"{title}. The highest-ranked cross-population route selects the receiver. "
        f"The panel shows the top {len(shown)} routes into it across {library_count}. "
        "Ligands are on the left and receptors on the right. Color identifies the sender. "
        "Width represents the median of 1 − LIANA magnitude rank across libraries reporting "
        "each route. The source table lists these libraries for every ribbon. Predictions use "
        "RNA expression. Cluster marker labels are descriptive."
    )
    return [
        *figure_artifacts(paths, name=stem, description=legend),
        StageArtifact(stem + "_source", table, "csv", legend),
    ]
