"""The ligand -> receiver composite: activity, comparison activity, senders, change, targets.

What the figure must not do is answer a question by omission or by colour: a ligand the caller
named must be a row even outside the top rows, a ligand a comparison receiver never ranked must
read as absent (not as rank zero), a sender change that did not clear its FDR must not look like
one that did, and a measurement that is missing must not be drawn as no change.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.communication import ligand_receiver_composite  # noqa: E402

LIGANDS = ["TGFB1", "IL1B", "TNF", "VEGFA", "BMP4", "LIG5", "LIG6", "LIG7"]
SENDERS = ["Fibroblasts", "Macrophages", "T/NK"]


def _activity(names=LIGANDS) -> pd.DataFrame:
    scores = np.linspace(0.30, 0.05, len(names))
    return pd.DataFrame(
        {"test_ligand": names, "aupr_corrected": scores, "rank": np.arange(1, len(names) + 1)}
    )


def _expression(names=LIGANDS) -> pd.DataFrame:
    rows = [(s, g, 0.2 + 0.1 * i, 0.3) for i, s in enumerate(SENDERS) for g in names]
    return pd.DataFrame(
        rows, columns=["sender", "ligand", "mean_expression", "fraction_expressing"]
    )


def _targets(names=LIGANDS) -> pd.DataFrame:
    rows = [
        (g, f"GENE{j}", 0.05 + 0.02 * ((i + j) % 5)) for i, g in enumerate(names) for j in range(6)
    ]
    return pd.DataFrame(rows, columns=["ligand", "target", "weight"])


def _change() -> pd.DataFrame:
    rows = [(s, g, 0.1, 0.5, 5.0) for s in SENDERS for g in LIGANDS]
    frame = pd.DataFrame(rows, columns=["sender", "ligand", "effect", "fdr", "n_donors_concordant"])
    frame.loc[
        (frame.sender == "Fibroblasts") & (frame.ligand == "TGFB1"),
        ["effect", "fdr", "n_donors_concordant"],
    ] = [0.4, 0.01, 9.0]
    return frame


@pytest.fixture(autouse=True)
def _close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _axis(fig, label):
    return next(ax for ax in fig.axes if ax.get_label() == label)


def _cell(fig, panel, ligand, column):
    ax = _axis(fig, panel)
    return next(p for p in ax.patches if p.get_gid() == f"{ligand}|{column}")


def _row_labels(fig) -> list[str]:
    return [t.get_text() for t in fig.texts if t.get_gid() == "row_label"]


def _compose(**kwargs):
    base = dict(
        activity=_activity(),
        expression=_expression(),
        targets=_targets(),
        top_n=5,
        receiver_label="LEC",
    )
    base.update(kwargs)
    return ligand_receiver_composite(**base)


def test_rows_are_the_top_n_best_first() -> None:
    fig = _compose()
    assert _row_labels(fig) == LIGANDS[:5]


def test_a_named_ligand_outside_the_top_rows_is_still_a_row() -> None:
    fig = _compose(highlight=("LIG7",))
    assert _row_labels(fig)[-1] == "LIG7"
    assert len(_row_labels(fig)) == 6


def test_highlighted_ligands_are_bold() -> None:
    fig = _compose(highlight=("TGFB1",))
    weights = {t.get_text(): t.get_fontweight() for t in fig.texts if t.get_gid() == "row_label"}
    assert weights["TGFB1"] == "bold"
    assert weights["IL1B"] != "bold"


def test_a_ligand_the_comparison_receiver_never_ranked_reads_as_absent_not_rank_zero() -> None:
    comparison = _activity(names=["IL1B", "TNF"]).assign(rank=[7, 9])
    fig = _compose(comparison=comparison, comparison_label="BEC")
    ranks = {t.get_gid(): t.get_text() for t in _axis(fig, "comparison").texts}
    assert ranks["comparison_rank|TGFB1"] == "–"
    assert ranks["comparison_rank|IL1B"] == "7"


def test_only_changes_that_clear_the_fdr_are_outlined_and_carry_the_donor_count() -> None:
    fig = _compose(change=_change())
    sig = _cell(fig, "change", "TGFB1", "Fibroblasts")
    ns = _cell(fig, "change", "TGFB1", "Macrophages")
    assert sig.get_linewidth() > ns.get_linewidth()
    assert ns.get_alpha() < sig.get_alpha()
    counts = [t.get_text() for t in _axis(fig, "change").texts]
    assert counts == ["9"]


def test_a_missing_change_measurement_is_blank_grey_not_zero_change() -> None:
    change = _change()
    change = change[~((change.sender == "T/NK") & (change.ligand == "TNF"))]
    fig = _compose(change=change)
    missing = _cell(fig, "change", "TNF", "T/NK")
    zero_change = _cell(fig, "change", "TNF", "Macrophages")
    assert tuple(missing.get_facecolor())[:3] != tuple(zero_change.get_facecolor())[:3]
    assert missing.get_gid() == "TNF|T/NK"


def test_panels_without_their_inputs_are_left_out() -> None:
    fig = _compose()
    assert {ax.get_label() for ax in fig.axes if ax.get_label()} == {
        "activity",
        "expression",
        "targets",
    }


def test_no_overlap_between_the_ranking_and_the_expression_frame_is_an_error() -> None:
    with pytest.raises(ValueError, match="none of the ranked ligands"):
        ligand_receiver_composite(
            activity=_activity(),
            expression=_expression(names=["OTHER1", "OTHER2"]),
            targets=_targets(),
            receiver_label="LEC",
        )


def test_targets_are_the_strongest_for_the_drawn_ligands() -> None:
    fig = _compose(max_targets=4)
    cols = {p.get_gid().split("|")[1] for p in _axis(fig, "targets").patches}
    assert len(cols) == 4


def test_a_sender_with_no_name_is_an_error_not_a_column_called_nan() -> None:
    expression = _expression()
    expression.loc[expression.index[0], "sender"] = np.nan
    with pytest.raises(ValueError, match="sender"):
        _compose(expression=expression)


def test_the_donor_count_is_legible_on_a_saturated_cell() -> None:
    change = _change()
    change.loc[(change.sender == "Fibroblasts") & (change.ligand == "TGFB1"), "effect"] = 0.4
    fig = _compose(change=change)
    (count,) = _axis(fig, "change").texts
    assert count.get_color() == "white"
    faint = change.assign(effect=lambda d: d.effect.where(~d.fdr.lt(0.05), 0.02))
    (count,) = _axis(_compose(change=faint), "change").texts
    assert count.get_color() != "white"
