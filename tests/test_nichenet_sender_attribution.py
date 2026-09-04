"""Attribution of pooled-ranking ligands back to the senders that express them.

A multi-sender NicheNet run has one ranking and no single source, so ``source`` cannot be
filled from the ranking. It is filled from expression, and the failure modes are all about
what a wrong fill would do downstream: invent a cell type, drop a real one, or read R's
``"FALSE"`` as truthy.
"""

from __future__ import annotations

import pandas as pd

from cellquorum.stages.cell_cell_communication._nichenet_io import (
    CANONICAL_COLUMNS,
    attribute_senders,
)

POOLED = "Fib, Mac, T"


def _canonical(ligands: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": POOLED,
            "target": "LEC",
            "ligand": ligands,
            "receptor": [f"{lg}R" for lg in ligands],
            "weight": [0.3] * len(ligands),
            "sample": "",
            "condition": "Lymphedema",
        }
    )[CANONICAL_COLUMNS]


def _expression(rows: list[tuple[str, str, float, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["sender", "ligand", "fraction_expressing", "expressed"])


def test_one_edge_per_expressing_sender() -> None:
    out = attribute_senders(
        _canonical(["TGFB1"]),
        _expression(
            [
                ("Fib", "TGFB1", 0.9, "TRUE"),
                ("Mac", "TGFB1", 0.4, "TRUE"),
                ("T", "TGFB1", 0.02, "FALSE"),
            ]
        ),
        sender_label=POOLED,
    )
    assert list(out["source"]) == ["Fib", "Mac"]
    assert set(out["ligand"]) == {"TGFB1"}


def test_the_string_false_is_not_truthy() -> None:
    """``bool("FALSE")`` is ``True``, and R writes booleans as strings."""
    out = attribute_senders(
        _canonical(["TGFB1"]),
        _expression([("Fib", "TGFB1", 0.9, "TRUE"), ("Mac", "TGFB1", 0.01, "FALSE")]),
        sender_label=POOLED,
    )
    assert list(out["source"]) == ["Fib"]


def test_a_ligand_no_sender_expresses_is_dropped_not_left_pooled() -> None:
    """An edge with no source is worse than a missing edge: it draws a cell type that isn't."""
    out = attribute_senders(
        _canonical(["TGFB1", "GHOSTL"]),
        _expression([("Fib", "TGFB1", 0.9, "TRUE")]),
        sender_label=POOLED,
    )
    assert list(out["ligand"]) == ["TGFB1"]
    assert POOLED not in set(out["source"])


def test_the_activity_weight_is_the_same_for_every_sender() -> None:
    """Activity is a property of the receiver's response; expression is what differs."""
    out = attribute_senders(
        _canonical(["TGFB1"]),
        _expression([("Fib", "TGFB1", 0.9, "TRUE"), ("Mac", "TGFB1", 0.2, "TRUE")]),
        sender_label=POOLED,
    )
    assert set(out["weight"]) == {0.3}


def test_rows_that_are_not_pooled_pass_through() -> None:
    canonical = pd.concat(
        [_canonical(["TGFB1"]), _canonical(["IL1B"]).assign(source="Mac")], ignore_index=True
    )
    out = attribute_senders(
        canonical,
        _expression([("Fib", "TGFB1", 0.9, "TRUE")]),
        sender_label=POOLED,
    )
    assert sorted(out["source"]) == ["Fib", "Mac"]


def test_the_fraction_threshold_is_used_when_no_expressed_column_exists() -> None:
    expression = _expression([("Fib", "TGFB1", 0.9, "TRUE"), ("Mac", "TGFB1", 0.05, "TRUE")]).drop(
        columns=["expressed"]
    )
    out = attribute_senders(
        _canonical(["TGFB1"]), expression, sender_label=POOLED, min_fraction=0.10
    )
    assert list(out["source"]) == ["Fib"]


def test_without_a_threshold_or_a_flag_every_listed_sender_is_kept() -> None:
    """Silently applying a made-up threshold would be worse than keeping what was given."""
    expression = _expression([("Fib", "TGFB1", 0.9, "TRUE"), ("Mac", "TGFB1", 0.05, "TRUE")]).drop(
        columns=["expressed"]
    )
    out = attribute_senders(_canonical(["TGFB1"]), expression, sender_label=POOLED)
    assert sorted(out["source"]) == ["Fib", "Mac"]


def test_a_missing_expression_table_leaves_the_table_alone() -> None:
    canonical = _canonical(["TGFB1"])
    out = attribute_senders(canonical, pd.DataFrame(), sender_label=POOLED)
    assert list(out["source"]) == [POOLED]


def test_an_empty_canonical_table_keeps_the_schema() -> None:
    out = attribute_senders(pd.DataFrame(), _expression([]), sender_label=POOLED)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out.empty


def test_the_schema_survives_expansion() -> None:
    out = attribute_senders(
        _canonical(["TGFB1"]),
        _expression([("Fib", "TGFB1", 0.9, "TRUE"), ("Mac", "TGFB1", 0.4, "TRUE")]),
        sender_label=POOLED,
    )
    assert list(out.columns) == CANONICAL_COLUMNS
    assert set(out["condition"]) == {"Lymphedema"}
    assert set(out["target"]) == {"LEC"}
