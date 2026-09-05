"""Figures for graded QC: what the evidence said, and what it cost which populations.

The pre-existing QC panels answer "what did the *threshold* do to this cohort" — a funnel, a
rule-attribution bar, a mixture fit. None of them can describe the graded model, because that
model has no single rule to attribute: a verdict is a *concordance* of families, scored within a
cell's own lineage, and turned into per-analysis permissions. A funnel cannot show any of that.

The gap this closes is not cosmetic. Every finding from the graded rollout so far was produced
by ad-hoc queries against ``obs`` — the mast-cell rescue, a 22,541-cell regression from
re-scaling a calibrated probability, a doublet cluster misread as a lost population. A reader
with no access to those queries could not have seen any of it, and a wet-lab user never will.
So these panels are built around the questions that actually caught those problems:

    lineage x family      is one cell type being condemned for its own biology?
    family co-occurrence  is concordance doing work, or is one family deciding everything?
    capture landscape     where does severity fall on the plot everyone already reads?
    sample x family       is attrition tracking the study design rather than the biology?

## Conventions, inherited from panels.py and extended

Horizontal layout so labels read left to right. Populations ordered by the quantity under
discussion rather than alphabetically, so the figure has a direction. Severity uses a
monotonic-lightness map (``magma_r``) because it is a magnitude, and it is scaled to the observed
range rather than to a nominal 0-1: quarantine is rare by design, so a fixed range would put every
bar in the first few percent of the colormap and encode nothing. Panels are omitted when their
input is absent rather than stubbed with an empty axis.

One addition specific to this model: **counts are always annotated on severity cells.** A median
severity of 0.9 over eleven cells and over eleven thousand are different claims, and a heatmap
alone cannot distinguish them — which is precisely how a five-cell ``unassigned`` group can look
like a catastrophe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cellquorum.visualization.qc.summarise import as_float

# ─── Column contract ────────────────────────────────────────────────────────────────

#: Per-cell family severity, written by the graded QC block.
FAMILY_SEVERITY_PREFIX = "qc_ev_family_"
FAMILY_SEVERITY_SUFFIX = "_severity"

#: The verdict and grouping columns these panels read.
STATE_COLUMN = "qc_state_initial"
REASON_COLUMN = "qc_state_reason"
LINEAGE_COLUMN = "qc_provisional_lineage"
FIT_COLUMN = "qc_fit_manifold"
COVERAGE_COLUMN = "qc_evidence_coverage"

#: Display names. The stored names are machine keys; a figure axis is read by a human.
FAMILY_LABELS: dict[str, str] = {
    "capture_complexity": "Capture /\ncomplexity",
    "nuclear_integrity": "Nuclear\nintegrity",
    "metabolic_stress": "Metabolic /\nstress",
    "ambient_background": "Ambient /\nbackground",
    "multiplet": "Multiplet",
}


def family_severity(obs: pd.DataFrame) -> pd.DataFrame:
    """Per-cell family severities present on this object, with readable column names.

    Args:
        obs: The observation frame.

    Returns:
        Cells x families. Empty when the graded block did not run.
    """
    columns = {
        column[len(FAMILY_SEVERITY_PREFIX) : -len(FAMILY_SEVERITY_SUFFIX)]: column
        for column in obs.columns
        if column.startswith(FAMILY_SEVERITY_PREFIX) and column.endswith(FAMILY_SEVERITY_SUFFIX)
    }
    if not columns:
        return pd.DataFrame(index=obs.index)
    return pd.DataFrame(
        {name: pd.to_numeric(obs[column], errors="coerce") for name, column in columns.items()},
        index=obs.index,
    )


def _pretty(family: str) -> str:
    """Display label for a family key."""
    return FAMILY_LABELS.get(family, family.replace("_", " ").capitalize())


# ─── 1. Lineage x evidence family ───────────────────────────────────────────────────


def plot_lineage_family_heatmap(
    obs: pd.DataFrame,
    output_path: str | Path,
    *,
    group_column: str = LINEAGE_COLUMN,
    min_cells: int = 20,
    dpi: int = 300,
) -> Path | None:
    """Median severity per family for each cell population, beside what it cost them.

    The panel that makes the central failure mode visible. A population whose *constitutive*
    biology is low-RNA and high-mitochondrial — mast cells, neutrophils, erythrocytes — reads as
    damaged on two families at once against a cohort-wide null, and the concordance rule then
    condemns it for being itself. On the validation cohort that was 18,002 mast cells at 59% fit
    eligibility; here it would appear as one row bright across two families with a high excluded
    fraction, which is a shape a reader can learn to recognise.

    Populations are ordered by excluded fraction, so the rows most worth interrogating are
    adjacent rather than scattered. Cell counts are printed in the severity cells because a
    median over a handful of cells is not the same claim as a median over thousands.

    Args:
        obs: Observation frame carrying graded QC columns.
        output_path: Destination PNG; a vector twin is written alongside.
        group_column: Population grouping. Defaults to the provisional lineage, which needs no
            annotation stage; pass a cell-type column when one exists.
        min_cells: Groups smaller than this are pooled into one "small groups" row rather than
            dropped, so no cell silently vanishes from a QC figure.

    Returns:
        The written path, or None when the graded columns are absent.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cellquorum.visualization.figstyle import save_cellquorum_figure, set_style

    severity = family_severity(obs)
    if severity.empty or group_column not in obs.columns:
        return None

    groups = _collapse_small_groups(obs[group_column].astype(str), min_cells=min_cells)
    excluded = _excluded_mask(obs)

    table = severity.groupby(groups).median()
    counts = groups.groupby(groups).size()
    exclusion = excluded.groupby(groups).mean()

    order = exclusion.sort_values(ascending=False).index
    table = table.loc[order]
    columns = [column for column in _family_order() if column in table.columns]
    table = table[columns]

    # Drop families that are identically zero for every population. A dead column costs a
    # quarter of the panel and tells a reader nothing: nuclear integrity is 0.00 across the board
    # on any dataset where MALAT1 carries no signal, which includes every single-nucleus run.
    informative = [column for column in columns if float(table[column].abs().max()) >= 0.005]
    if informative:
        columns, table = informative, table[informative]

    set_style()
    height = max(3.2, 0.42 * len(table) + 2.0)
    figure = plt.figure(figsize=(9.4, height))
    # An explicit grid, because letting colorbar() steal space from a 2-axes layout drew the bar
    # on top of the right panel and clipped its labels.
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[2.4, 1.0],
        height_ratios=[1.0, 0.03],
        hspace=0.06,
        wspace=0.30,
    )
    heat = figure.add_subplot(grid[0, 0])
    bar = figure.add_subplot(grid[0, 1])
    legend_axis = figure.add_subplot(grid[1, 0])

    # Severity is a magnitude, so a monotonic-lightness map. Fixed 0-1 limits, because the
    # scale is absolute by construction and letting it autoscale would make two runs
    # incomparable — the whole point of a calibrated severity.
    # Scaled to the observed range, not fixed to [0, 1]. Severity is absolute by construction, so
    # a fixed scale is tempting for cross-run comparability — but real cohorts occupy 0.00-0.20,
    # which renders as a uniformly pale grid carrying no information at all. The ceiling is
    # printed on the legend instead, so the scale is stated rather than guessed.
    # The ceiling comes from groups large enough to mean something. A pooled three-cell row at
    # severity 1.00 otherwise sets the scale for the whole panel and renders every real lineage
    # pale — the tiny group is the least informative row and was dominating the most.
    substantial = counts.loc[table.index] >= min_cells
    scale_source = table[substantial.to_numpy()] if bool(substantial.any()) else table
    ceiling = float(np.nanmax(scale_source.to_numpy(dtype=float)))
    ceiling = max(ceiling, 0.05)
    image = heat.imshow(
        table.to_numpy(dtype=float), cmap="magma_r", vmin=0.0, vmax=ceiling, aspect="auto"
    )
    heat.set_xticks(range(len(columns)), [_pretty(column) for column in columns], fontsize=8)
    heat.set_yticks(range(len(table)), table.index, fontsize=8)
    heat.set_title("Median severity within population", fontsize=10, loc="left")

    for row in range(len(table.index)):
        for column_index in range(len(columns)):
            value = table.iloc[row, column_index]
            if not np.isfinite(as_float(value)):
                # An unmeasurable family is not a severity of zero. Marking it keeps absent
                # evidence from reading as evidence of health, which is the model's core rule.
                heat.text(column_index, row, "·", ha="center", va="center", fontsize=9, color="0.5")
                continue
            heat.text(
                column_index,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if as_float(value) > 0.55 * ceiling else "0.15",
            )

    legend = figure.colorbar(image, cax=legend_axis, orientation="horizontal")
    legend.set_label(
        f"median severity  (scale 0-{ceiling:.2f}, set by groups of >={min_cells} cells; "
        f"darker cells exceed it)",
        fontsize=8,
    )
    legend.ax.tick_params(labelsize=7)

    # The consequence panel: severity is only interesting insofar as it removed something.
    cohort_rate = float(excluded.mean())
    values = exclusion.loc[order]
    colors = ["#C41E3A" if value > cohort_rate else "#5A8BC4" for value in values]
    bar.barh(range(len(values)), values.to_numpy(dtype=float), color=colors, height=0.72)
    bar.axvline(cohort_rate, color="0.25", linewidth=0.9, linestyle="--")
    bar.text(
        cohort_rate,
        len(values) - 0.2,
        f"  cohort {cohort_rate:.0%}",
        fontsize=7,
        color="0.25",
        va="top",
    )
    bar.set_yticks(range(len(values)), [""] * len(values))
    for row, group in enumerate(order):
        bar.text(
            1.02,
            row,
            f"n={counts.loc[group]:,}",
            fontsize=6.5,
            va="center",
            color="0.3",
            transform=bar.get_yaxis_transform(),
        )
    bar.set_xlim(0, 1)
    bar.set_xlabel("excluded from fitting", fontsize=8)
    bar.set_title("Cost", fontsize=10, loc="left")
    bar.invert_yaxis()
    heat.invert_yaxis()
    for spine in ("top", "right"):
        bar.spines[spine].set_visible(False)

    destination = Path(output_path)
    save_cellquorum_figure(figure, destination, dpi=dpi, companion_formats=(".pdf", ".svg"))
    plt.close(figure)
    return destination


# ─── 2. Which families co-fire ──────────────────────────────────────────────────────


def plot_family_cooccurrence(
    obs: pd.DataFrame,
    output_path: str | Path,
    *,
    concern_severity: float = 0.50,
    max_combinations: int = 12,
    dpi: int = 300,
) -> Path | None:
    """How often families raise concern together — the concordance rule, made auditable.

    Quarantine requires two independent families to agree, so this panel answers whether that
    requirement is doing anything. Two shapes are worth recognising:

    * One family dominating every combination means concordance is decorative, and that family
      is effectively the whole filter.
    * A large single-family bar next to a small paired bar means most flagged cells would have
      been condemned by a one-axis rule and are being held back deliberately.

    Args:
        obs: Observation frame carrying graded QC columns.
        output_path: Destination PNG; a vector twin is written alongside.
        concern_severity: Severity at or above which a family counts as raising concern.
        max_combinations: Combinations to show, most frequent first.

    Returns:
        The written path, or None when the graded columns are absent.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cellquorum.visualization.figstyle import save_cellquorum_figure, set_style

    severity = family_severity(obs)
    if severity.empty:
        return None

    concerning = severity >= concern_severity
    families = [column for column in _family_order() if column in concerning.columns]
    concerning = concerning[families]

    labels = concerning.apply(
        lambda row: " + ".join(family for family in families if row[family]) or "none", axis=1
    )
    counts = labels.value_counts()
    none_count = int(counts.get("none", 0))
    counts = counts.drop(labels=["none"], errors="ignore").head(max_combinations)
    if counts.empty:
        return None

    # Coloured by QUARANTINE rate, not by exclusion-from-fitting.
    #
    # Exclusion was the original choice and it is a dead encoding: any cell with a concerning
    # family loses FIT, so the value was 1.00 for every combination except multiplet, and every
    # bar rendered at the dark end of the map. Measured on the 201,871-cell cohort it spanned
    # 0.631-1.000, almost all of it exactly 1.
    #
    # Quarantine rate spans 0.000-0.374 on the same data, and it varies with concordance by
    # construction: every single-family combination is 0.000 because one family can never condemn
    # a cell, two families reach ~0.11, three reach 0.37. So the colour now shows the panel's
    # actual question — is requiring concordance doing any work — instead of showing nothing.
    quarantined = obs[STATE_COLUMN].astype(str).eq("quarantine") if STATE_COLUMN in obs else None
    if quarantined is None:
        return None
    consequence = quarantined.groupby(labels).mean().reindex(counts.index)

    set_style()
    figure, axis = plt.subplots(figsize=(8.4, max(3.0, 0.42 * len(counts) + 1.8)))

    positions = range(len(counts))
    # Coloured by consequence, not by size: a frequent combination that condemns nobody is a
    # different story from a rare one that condemns a third of its cells.
    #
    # Scaled to the observed maximum rather than to 1.0. Quarantine is rare by design — the
    # cohort rate is 0.5% — so a fixed 0-1 range would put every bar in the first 3% of the
    # colormap and flatten the encoding a second time, for the opposite reason.
    ceiling = float(consequence.max()) if len(consequence) else 0.0
    scale = ceiling if ceiling > 0 else 1.0
    image = axis.barh(
        positions,
        counts.to_numpy(),
        color=[
            plt.get_cmap("magma_r")(min(value / scale, 1.0)) for value in consequence.fillna(0.0)
        ],
        height=0.74,
    )
    axis.set_yticks(
        positions,
        [_combination_label(name, families) for name in counts.index],
        fontsize=8,
    )
    axis.invert_yaxis()
    axis.set_xlabel(f"cells with family severity ≥ {concern_severity:g}", fontsize=9)
    axis.set_title(
        f"Which evidence families agree   ({none_count:,} cells raise no concern)",
        fontsize=10,
        loc="left",
    )
    for bar, count, rate in zip(image, counts.to_numpy(), consequence.fillna(0.0), strict=False):
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {count:,}  ({rate:.1%} quarantined)",
            va="center",
            fontsize=7,
            color="0.2",
        )
    axis.set_xlim(0, counts.max() * 1.35)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    destination = Path(output_path)
    save_cellquorum_figure(figure, destination, dpi=dpi, companion_formats=(".pdf", ".svg"))
    plt.close(figure)
    return destination


def _combination_label(name: str, families: list[str]) -> str:
    """Readable label for a family combination, with its size."""
    parts = [part.strip() for part in name.split("+")]
    pretty = " + ".join(_pretty(part).replace("\n", " ") for part in parts)
    return f"{pretty}  [{len(parts)}]"


# ─── Shared helpers ─────────────────────────────────────────────────────────────────


def _family_order() -> list[str]:
    """Families in a fixed reading order, damage first and multiplet last.

    Fixed rather than data-driven so two runs' figures can be laid side by side. Multiplet sits
    last because it is not damage — a doublet is an excellent library that is not one cell — and
    grouping it with the damage families invites exactly that misreading.
    """
    return [
        "capture_complexity",
        "nuclear_integrity",
        "metabolic_stress",
        "ambient_background",
        "multiplet",
    ]


def _excluded_mask(obs: pd.DataFrame) -> pd.Series:
    """Per-cell "barred from fitting", from the mask when present, else from the state.

    The eligibility mask is authoritative; the state is a fallback for objects written before
    the masks existed, so an older run still plots rather than erroring.
    """
    if FIT_COLUMN in obs.columns:
        return ~obs[FIT_COLUMN].astype(bool)
    if STATE_COLUMN in obs.columns:
        return obs[STATE_COLUMN].astype(str) != "core"
    return pd.Series(False, index=obs.index)


def _collapse_small_groups(groups: pd.Series, *, min_cells: int) -> pd.Series:
    """Pool groups below ``min_cells`` into one row rather than dropping them.

    Dropping them would be the more common choice and the wrong one: a QC figure that omits the
    smallest populations omits precisely the ones at risk. Pooling keeps every cell on the plot
    while stopping a five-cell group from dominating a colour scale.
    """
    sizes = groups.groupby(groups).transform("size")
    pooled = groups.where(sizes >= min_cells, f"small groups (<{min_cells})")
    return pooled.astype(str)


__all__ = [
    "FAMILY_LABELS",
    "family_severity",
    "plot_family_cooccurrence",
    "plot_lineage_family_heatmap",
]


# ─── 3. Severity distributions, per group ───────────────────────────────────────────


def plot_severity_ecdf(
    obs: pd.DataFrame,
    output_path: str | Path,
    *,
    group_column: str = "sample_id",
    pair_column: str | None = "donor_id",
    condition_column: str | None = "condition",
    family: str = "capture_complexity",
    concern_severity: float = 0.50,
    dpi: int = 300,
) -> Path | None:
    """Empirical CDF of one family's severity, per group, arms contrasted.

    An ECDF and not a raincloud, and the reason is in the data rather than in taste. Severity is
    bounded on [0, 1] and **45-65% of cells sit at exactly zero** on three of the four families,
    because a cell inside its healthy mode scores zero by construction. A kernel density over a
    spike at a boundary produces an artifact, not a distribution: the first version of this panel
    drew a collapsed sliver at the left edge, boxes that rendered as empty rectangles wider than
    the data, and jittered points that merged into a solid streak.

    An ECDF has none of those failure modes. The zero mass appears as an honest jump at the
    origin, nothing is smoothed, and the flagged fraction can be read directly off the concern
    bar — which is the number the panel exists to communicate. The paired inset states it
    numerically per donor, because for a paired design "is this arm worse" is the question and a
    pair of curves invites the eye to guess at it.

    Args:
        obs: Observation frame with graded QC columns.
        output_path: Destination PNG; a vector twin is written alongside.
        group_column: One curve per group. The **library**, not the donor: a donor in a paired
            design contributes a sample to each arm, so grouping by donor collapses the very
            contrast the panel exists to show. An earlier version did exactly that and drew
            one point per donor in a single colour.
        pair_column: Column joining a donor's libraries across arms. None leaves points
            unjoined.
        condition_column: Study arm, used for colour and for the paired inset.
        family: Evidence family to draw.
        concern_severity: Bar above which a family raises concern.

    Returns:
        The written path, or None when the inputs are absent.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from cellquorum.visualization.figstyle import (
        LE_RED,
        NORMAL_BLUE,
        save_cellquorum_figure,
        set_style,
    )

    severity = family_severity(obs)
    if severity.empty or family not in severity.columns or group_column not in obs.columns:
        return None

    values = severity[family]
    groups = obs[group_column].astype(str)
    arms = (
        obs[condition_column].astype(str)
        if condition_column and condition_column in obs.columns
        else pd.Series("all", index=obs.index)
    )
    arm_levels = sorted(arms.unique())
    palette = _arm_palette(arm_levels, NORMAL_BLUE, LE_RED)

    set_style()
    figure, (curves, paired) = plt.subplots(
        1, 2, figsize=(10.0, 4.4), gridspec_kw={"width_ratios": [2.1, 1.0], "wspace": 0.28}
    )

    curves.axvspan(concern_severity, 1.0, color=LE_RED, alpha=0.055, zorder=0)
    curves.axvline(concern_severity, color=LE_RED, linewidth=0.7, alpha=0.5, zorder=1)

    # One thin curve per group, one heavy curve per arm. The thin curves show between-donor
    # spread — which is what decides whether an arm difference is real or one outlier library.
    pairs = obs[pair_column].astype(str) if pair_column and pair_column in obs.columns else groups
    flagged: dict[str, dict[str, float]] = {}
    for group in sorted(groups.unique()):
        selected = (groups == group).to_numpy()
        sample = values[selected].dropna().to_numpy(dtype=float)
        if sample.size < 10:
            continue
        arm = arms[selected].iloc[0]
        grid, curve = _ecdf(sample)
        curves.step(grid, curve, where="post", color=palette[arm], linewidth=0.7, alpha=0.45)
        # Keyed by the PAIR, so the inset can join a donor's two libraries. Keying by the library
        # is what made every donor appear once, in one arm.
        flagged.setdefault(arm, {})[pairs[selected].iloc[0]] = float(
            (sample >= concern_severity).mean()
        )

    for arm in arm_levels:
        sample = values[(arms == arm).to_numpy()].dropna().to_numpy(dtype=float)
        if sample.size < 10:
            continue
        grid, curve = _ecdf(sample)
        curves.step(grid, curve, where="post", color=palette[arm], linewidth=2.1, alpha=0.95)

    curves.set_xlim(-0.01, 1.01)
    curves.set_ylim(0, 1.005)
    curves.set_xlabel(f"{_pretty(family).replace(chr(10), ' ')} severity", fontsize=9)
    curves.set_ylabel("cumulative fraction of cells", fontsize=9)
    curves.set_title(
        f"Severity distribution   (thin: per {group_column.replace('_', ' ')}; heavy: per arm)",
        fontsize=10,
        loc="left",
    )
    curves.legend(
        handles=[
            Line2D([], [], color=palette[arm], linewidth=2.0, label=arm) for arm in arm_levels
        ],
        frameon=False,
        fontsize=8,
        loc="lower right",
    )

    _plot_paired_flagged(paired, flagged, palette, concern_severity)

    for axis in (curves, paired):
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    destination = Path(output_path)
    save_cellquorum_figure(figure, destination, dpi=dpi, companion_formats=(".pdf", ".svg"))
    plt.close(figure)
    return destination


def _ecdf(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid and cumulative fraction for an ECDF, starting at zero."""
    ordered = np.sort(sample)
    grid = np.concatenate(([0.0], ordered))
    curve = np.concatenate(([0.0], np.arange(1, ordered.size + 1) / ordered.size))
    return grid, curve


def _plot_paired_flagged(
    axis: object,
    flagged: dict[str, dict[str, float]],
    palette: dict[str, str],
    concern_severity: float,
) -> None:
    """Per-group flagged fraction, arms joined where a group appears in both.

    Joining the pair is the point. A cohort where every donor contributes to both arms answers
    "is this arm worse" within donor, which removes between-donor variation — the dominant term.
    Two clouds of unpaired points cannot make that comparison and invite the reader to make it
    by eye anyway.
    """
    arm_levels = sorted(flagged)
    if not arm_levels:
        return

    # Groups present in every arm can be paired; the rest are drawn but not joined.
    shared = set.intersection(*(set(flagged[arm]) for arm in arm_levels)) if flagged else set()
    order = sorted(shared, key=lambda group: flagged[arm_levels[0]][group])
    unpaired = sorted({group for arm in arm_levels for group in flagged[arm]} - shared)

    for row, group in enumerate([*order, *unpaired]):
        present = [arm for arm in arm_levels if group in flagged[arm]]
        if len(present) > 1:
            axis.plot(  # type: ignore[attr-defined]
                [flagged[arm][group] for arm in present],
                [row] * len(present),
                color="0.72",
                linewidth=0.9,
                zorder=1,
            )
        for arm in present:
            axis.scatter(  # type: ignore[attr-defined]
                flagged[arm][group], row, s=34, color=palette[arm], zorder=2, linewidths=0
            )

    labels = [*order, *unpaired]
    axis.set_yticks(range(len(labels)), labels, fontsize=7)  # type: ignore[attr-defined]
    axis.set_xlabel(f"fraction ≥ {concern_severity:g}", fontsize=9)  # type: ignore[attr-defined]
    axis.set_title("Paired by donor", fontsize=10, loc="left")  # type: ignore[attr-defined]
    axis.grid(axis="x", alpha=0.18, linewidth=0.5)  # type: ignore[attr-defined]


def _arm_palette(levels: list[str], normal: str, case: str) -> dict[str, str]:
    """Two-arm palette that puts the control arm in blue whichever way it is spelled."""
    if len(levels) == 1:
        return {levels[0]: normal}
    control = next(
        (level for level in levels if level.lower().startswith(("normal", "control", "healthy"))),
        levels[0],
    )
    return {level: (normal if level == control else case) for level in levels}


# ─── The writer a run calls ─────────────────────────────────────────────────────────


def write_graded_qc_figures(
    obs: pd.DataFrame,
    output_dir: str | Path,
    *,
    concern_severity: float = 0.50,
    sample_column: str = "sample_id",
    pair_column: str | None = "donor_id",
    condition_column: str | None = "condition",
    dpi: int = 300,
) -> tuple[list[Path], list[str]]:
    """Write every graded QC panel this object supports.

    Panels are skipped, never stubbed, when their input is absent — a cohort with no study arm
    should not get an empty condition panel, and a run without graded QC should get no graded
    figures at all rather than a directory of blank axes.

    Args:
        obs: Observation frame from the QC object.
        output_dir: Directory to write into; created if absent.
        concern_severity: The concern bar, drawn on panels that show it.
        sample_column: Library column, used as the unit for distribution curves.
        pair_column: Donor column, used to join arms within a subject.
        condition_column: Study-arm column.
        dpi: Raster resolution. A vector twin is written regardless.

    Returns:
        ``(paths, warnings)`` — every figure written, and anything a reader must know.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    severity = family_severity(obs)
    if severity.empty:
        return [], ["Graded QC figures skipped: no graded evidence columns on the object."]

    written: list[Path] = []
    warnings: list[str] = []

    def keep(path: Path | None) -> None:
        if path is not None:
            written.append(path)

    keep(
        plot_lineage_family_heatmap(
            obs,
            directory / "graded_lineage_family.png",
            group_column=LINEAGE_COLUMN,
            dpi=dpi,
        )
    )
    keep(
        plot_family_cooccurrence(
            obs,
            directory / "graded_family_cooccurrence.png",
            concern_severity=concern_severity,
            dpi=dpi,
        )
    )

    # One distribution panel per family that is actually present, because which families a
    # dataset supports is a property of the assay: snRNA has no meaningful nuclear-integrity
    # axis, and a run without doublet detection has no multiplet family.
    for family in severity.columns:
        keep(
            plot_severity_ecdf(
                obs,
                directory / f"graded_severity_{family}.png",
                group_column=sample_column,
                pair_column=pair_column,
                condition_column=condition_column,
                family=family,
                concern_severity=concern_severity,
                dpi=dpi,
            )
        )

    # Raw-metric distributions, paired by donor. One modular figure per metric rather than a
    # grid, per the spec: the point is that each is a publication panel in its own right, and a
    # grid of nine cannot be read for any of them.
    if pair_column and condition_column and pair_column in obs.columns:
        control = _control_arm(obs, condition_column)
        for metric, label, log_scale in RAW_METRICS:
            if metric not in obs.columns:
                # Not a warning. Which metrics exist is a property of the assay and the config —
                # hemoglobin on a tissue with no blood, MALAT1 on a run with the nuclear axis
                # off — and a warning per absent metric would bury the ones that matter.
                continue
            keep(
                plot_metric_rainclouds(
                    obs,
                    directory / f"qc_metric_{metric}.png",
                    metric=metric,
                    label=label,
                    log_scale=log_scale,
                    donor_column=pair_column,
                    condition_column=condition_column,
                    control_label=control,
                    dpi=dpi,
                )
            )

    if not written:
        warnings.append(
            "Graded QC figures produced nothing: the graded columns are present but no panel "
            "found the grouping columns it needs."
        )
    return written, warnings


def _control_arm(obs: pd.DataFrame, condition_column: str) -> str | None:
    """Name the arm that goes on the left, without hard-coding this study's labels.

    The house rule is "control first", and CellQuorum has to apply it on a cohort whose arms are
    called something else. So the label is recognised from a list of conventional control names
    rather than from ``"Normal"`` alone, and when none matches the first arm in natural order is
    used — deterministic, and stated in the figure title so a reader can see which way it reads.
    """
    arms = [str(value) for value in pd.unique(obs[condition_column].dropna())]
    conventional = ("normal", "control", "healthy", "wt", "wildtype", "untreated", "baseline")
    for candidate in arms:
        if candidate.strip().lower() in conventional:
            return candidate
    return None


# ─── Attribution: what drove each exclusion ─────────────────────────────────────────


def graded_attribution_table(obs: pd.DataFrame) -> pd.DataFrame:
    """Per-driver attribution of exclusions, replacing per-threshold-rule attribution.

    The threshold model attributed an exclusion to a *rule* — "failed max_mito_percent" — and
    reported each rule's gross and marginal contribution. Graded QC has no rules to attribute to:
    a verdict is a concordance of evidence families scored within a lineage. The equivalent
    question, and the one a reader actually asks, is **which family drove it**, which the
    adjudicator already records per cell as ``qc_primary_driver`` alongside the route it took in
    ``qc_state_reason``.

    Reporting both matters. The driver says *what* was wrong; the route says whether that was
    enough to act on — ``supporting_evidence_only`` and ``concordant_severe_damage`` can share a
    driver and mean opposite things.

    Args:
        obs: Observation frame with graded QC columns.

    Returns:
        One row per driver, with the exclusion routes broken out, ordered by cells excluded.
        Empty when the graded columns are absent.
    """
    driver_column, reason_column = "qc_primary_driver", REASON_COLUMN
    if driver_column not in obs.columns:
        return pd.DataFrame()

    excluded = _excluded_mask(obs)
    drivers = obs[driver_column].astype(str).replace({"": "none", "nan": "none"})
    reasons = (
        obs[reason_column].astype(str)
        if reason_column in obs.columns
        else pd.Series("unknown", index=obs.index)
    )

    rows = []
    for driver in sorted(set(drivers)):
        selected = (drivers == driver).to_numpy()
        if not selected.any():
            continue
        rows.append(
            {
                "driver": _pretty(driver).replace("\n", " ") if driver != "none" else "none",
                "n_cells": int(selected.sum()),
                "n_excluded": int(excluded.to_numpy()[selected].sum()),
                "excluded_fraction": float(excluded.to_numpy()[selected].mean()),
                # The dominant route for this driver, which is what says whether the driver was
                # acted on or merely noted.
                "leading_route": (
                    reasons[selected].value_counts().index[0] if selected.any() else ""
                ),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("n_excluded", ascending=False).set_index("driver")


def graded_policy_sentence(
    concern_severity: float,
    severe_severity: float,
    min_concordant_families: int,
    *,
    half_severity_z: float = 3.0,
) -> str:
    """One sentence describing the graded policy, for a table's source note.

    Replaces the threshold criteria sentence. A threshold policy could be written as a list of
    bounds; a graded one cannot, because the bars are on a severity scale and mean nothing
    without their sigma equivalence. So the sentence states both, and the concordance requirement
    that stops any single axis condemning a cell.
    """
    concern_z = half_severity_z * concern_severity / max(1.0 - concern_severity, 1e-9)
    severe_z = half_severity_z * severe_severity / max(1.0 - severe_severity, 1e-9)
    return (
        f"Cells were graded rather than thresholded: severity is a saturating function of a "
        f"robust deviation from each cell's own lineage, so {concern_severity:g} corresponds to "
        f"{concern_z:.1f} and {severe_severity:g} to {severe_z:.1f} robust standard deviations. "
        f"A cell reaching {concern_severity:g} on any damage family is borderline; quarantine "
        f"additionally requires {min_concordant_families} independent families to agree at "
        f"{severe_severity:g}, at least one of which can establish damage on its own. No cell is "
        f"deleted by grading — eligibility to fit, to receive a transform, and to inform an "
        f"inference are recorded separately."
    )


# ─── Calibration: raw metric distributions, paired by donor ─────────────────────────

#: The raw QC metrics the per-donor distribution figures cover, in reading order, with the
#: axis label and whether a log scale is right for them.
#:
#: Raw metrics, not severities — this is the correction the figure spec makes explicit. A
#: severity has already had a null subtracted and a saturating transform applied, so it cannot
#: be read to decide where the null should sit, and 45-65% of cells sit at exactly 0.0 on several
#: axes, which collapses any density estimate. The metric is what a calibration figure has to
#: show; the severity is what the metric was turned into.
RAW_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("total_counts", "UMIs per cell", True),
    ("n_genes_by_counts", "Genes per cell", True),
    ("pct_counts_mito", "Mitochondrial %", False),
    ("pct_counts_ribo", "Ribosomal %", False),
    ("pct_counts_in_top_20_genes", "Top-20-gene concentration (%)", False),
    ("pct_counts_hemoglobin", "Hemoglobin %", False),
    ("qc_ev_malat1_fraction_value", "MALAT1 fraction", False),
    ("qc_ev_dissociation_stress_value", "Dissociation-stress fraction", False),
    ("doublet_score", "Doublet score", False),
)


def plot_metric_rainclouds(
    obs: pd.DataFrame,
    destination: Path,
    *,
    metric: str,
    label: str,
    log_scale: bool,
    donor_column: str,
    condition_column: str,
    control_label: str | None,
    group_column: str | None = None,
    dpi: int = 300,
    max_points_per_arm: int = 300,
    seed: int = 0,
) -> Path | None:
    """One raw metric, per donor, control arm left and case arm right.

    The figure the spec puts first among the calibration outputs, and the one the previous
    per-patient boxplots could not be: those were drawn with ``showfliers`` off, so the cells a
    bound would act on were invisible, and they were drawn "after filtering", so they described
    the result of a threshold rather than the evidence for one.

    Four properties, each fixing a specific defect in what came before:

    * **paired, control first, within donor** — via
      :func:`~cellquorum.visualization.figstyle.paired_condition_order`, so this figure and every
      other one in the run agree about who is on the left;
    * **donors in natural order**, so ``P2`` precedes ``P10``;
    * **the tail is drawn**, as a half-violin plus sampled cells rather than a clipped whisker;
    * **no cell-level p-value.** Cells are not biological replicates, so a test across the
      pooled cells of two arms answers a question nobody asked. Where a statistic is warranted
      it is computed on donor medians by the caller, using
      :func:`~cellquorum.visualization.figstyle.two_group_test_on_donor_medians`.

    Args:
        obs: Observation frame from the QC object.
        destination: Output path; the vector twins are written beside it.
        metric: Column to plot.
        label: Y-axis label.
        log_scale: Use a log y-axis, right for the count-like metrics and wrong for fractions.
        donor_column: Donor column, the pairing unit.
        condition_column: Study-arm column.
        control_label: Arm to place first inside each donor.
        group_column: Optional population column. When given, one figure is drawn per
            population instead of per donor — the §4 variant of the same framework.
        dpi: Raster resolution.
        max_points_per_arm: Cells drawn per arm, deterministically subsampled.
        seed: Subsampling seed.

    Returns:
        The written path, or ``None`` when the object lacks the metric or the grouping columns.
    """
    import matplotlib.pyplot as plt

    from cellquorum.visualization.figstyle import (
        FONTSIZE,
        LE_RED,
        NORMAL_BLUE,
        TEXT,
        natural_key,
        paired_condition_order,
        raincloud,
        save_cellquorum_figure,
        set_style,
    )

    required = [metric, donor_column, condition_column]
    if any(column not in obs.columns for column in required):
        return None

    frame = obs[[*required, *([group_column] if group_column in obs.columns else [])]].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame[np.isfinite(frame[metric])]
    if frame.empty:
        return None

    # The categorical axis: donors, or populations when one is requested.
    if group_column and group_column in frame.columns:
        categories = sorted({str(value) for value in frame[group_column].dropna()}, key=natural_key)
        category_column = group_column
    else:
        order = paired_condition_order(
            frame,
            donor_col=donor_column,
            condition_col=condition_column,
            control=control_label,
        )
        categories = list(dict.fromkeys(donor for donor, _ in order))
        category_column = donor_column

    arms = paired_condition_order(
        frame, donor_col=donor_column, condition_col=condition_column, control=control_label
    )
    arm_order = list(dict.fromkeys(arm for _, arm in arms))
    if not arm_order or not categories:
        return None

    # Two arms get the house red/blue and the half-violins face away from each other; more than
    # two would make "left and right" meaningless, so they are drawn as separate offsets.
    colours = {arm: NORMAL_BLUE if index == 0 else LE_RED for index, arm in enumerate(arm_order)}
    sides = {arm: "left" if index == 0 else "right" for index, arm in enumerate(arm_order)}

    set_style()
    width = max(4.0, 0.95 * len(categories) + 1.6)
    figure, axis = plt.subplots(figsize=(width, 3.9))

    # The view ceiling is decided before anything is drawn, so the densities can be clipped to it.
    # See the annotation below for why the view is clipped at all.
    ceiling = None if log_scale else float(np.nanpercentile(frame[metric], 99.5))
    density_clip = None if ceiling is None else (float(frame[metric].min()), ceiling)

    for position, category in enumerate(categories):
        in_category = frame[category_column].astype(str) == category
        for arm in arm_order:
            values = frame.loc[in_category & (frame[condition_column].astype(str) == arm), metric]
            raincloud(
                axis,
                values.to_numpy(),
                float(position),
                color=colours[arm],
                side=sides[arm],
                points=max_points_per_arm,
                # Seeded per category and arm so the drawn cells are stable across reruns but
                # not identical between panels, which would make the jitter look like structure.
                seed=seed + position * 17 + arm_order.index(arm),
                clip=density_clip,
            )

    if log_scale:
        axis.set_yscale("log")
    else:
        # Clip the view to the bulk, and say how many cells sit above it.
        #
        # Not to hide the tail — the spec is explicit that the cells a bound acts on must be
        # visible, and hiding them is what made a mitochondrial ceiling look non-binding when it
        # accounted for essentially every removal. But a handful of barcodes at 95%
        # mitochondrial content flattens every box in the figure to a few pixels, which hides
        # the distribution instead. So the axis covers the 99.5th percentile, the count beyond
        # it is printed, and no cell goes unaccounted for.
        assert ceiling is not None  # set above whenever log_scale is False
        beyond = int((frame[metric] > ceiling).sum())
        if beyond and ceiling > 0:
            axis.set_ylim(top=ceiling * 1.08)
            axis.annotate(
                f"+{beyond:,} cells above {ceiling:,.3g}",
                xy=(0.99, 0.99),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=FONTSIZE["annotation"],
                color=TEXT,
            )
    axis.set_xticks(range(len(categories)))
    axis.set_xticklabels(categories, fontsize=FONTSIZE["tick"])
    axis.set_ylabel(label, fontsize=FONTSIZE["axis_title"], color=TEXT)
    axis.set_xlim(-0.65, len(categories) - 0.35)
    axis.tick_params(axis="y", labelsize=FONTSIZE["tick"])

    # The n per arm belongs on the figure: a raincloud over 40 cells and over 40,000 look alike.
    counts = frame[condition_column].astype(str).value_counts()
    handles = [
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            color=colours[arm],
            markersize=6,
            label=f"{arm} (n={int(counts.get(arm, 0)):,})",
        )
        for arm in arm_order
    ]
    axis.legend(handles=handles, frameon=False, fontsize=FONTSIZE["legend"], loc="best")
    axis.set_title(
        f"{label} — {'by population' if category_column == group_column else 'by donor'}, "
        f"{arm_order[0]} left",
        fontsize=FONTSIZE["title"],
        color=TEXT,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_cellquorum_figure(figure, destination, dpi=dpi, companion_formats=(".pdf", ".svg"))
    plt.close(figure)
    return destination
