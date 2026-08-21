"""Unit tests for the pseudobulk donor-pairing resolver.

``resolve_donor_pairing`` is the safety net against the silent-wrong-DE class
where a fully matched (every donor in both arms) design is analysed *unpaired*,
leaving donor baseline variance in the residual and producing false nulls
(the paired-design artifact that gated the science re-runs). The method used to
carry this logic inline, reachable only through the R backend; it is now a pure
helper so the auto-promotion and complete-pair-restriction branches are covered
without invoking edgeR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cellquorum.comparative.differential_expression.pseudobulk import (
    PseudobulkResult,
    resolve_donor_pairing,
)

_DONOR_COL = "patient_id"
_CONDITION_COL = "condition"


def _pb(samples: list[tuple[str, str]]) -> PseudobulkResult:
    """Build a PseudobulkResult from ``(donor, condition)`` pseudo-samples."""

    # Deterministic pseudo-sample index mirrors aggregate_pseudobulk's keys.
    index = [f"{donor}__{condition}" for donor, condition in samples]

    # Distinct integer counts per row so restriction is observable by value.
    counts = pd.DataFrame(
        np.arange(len(samples) * 3, dtype=np.int64).reshape(len(samples), 3),
        index=index,
        columns=["G0", "G1", "G2"],
    )
    sample_meta = pd.DataFrame(
        {
            _DONOR_COL: [donor for donor, _ in samples],
            _CONDITION_COL: [condition for _, condition in samples],
        },
        index=index,
    )
    return PseudobulkResult(counts=counts, sample_meta=sample_meta)


def _resolve(pb: PseudobulkResult, *, paired: bool):
    """Invoke the resolver with the fixed case/control labels."""

    return resolve_donor_pairing(
        pb,
        donor_col=_DONOR_COL,
        condition_col=_CONDITION_COL,
        case="LE",
        control="Normal",
        paired=paired,
    )


class TestAutoPromote:
    def test_fully_matched_unpaired_is_promoted(self):
        # Three donors, each contributing both arms, declared unpaired.
        pb = _pb(
            [
                ("d1", "LE"),
                ("d1", "Normal"),
                ("d2", "LE"),
                ("d2", "Normal"),
                ("d3", "LE"),
                ("d3", "Normal"),
            ]
        )
        decision = _resolve(pb, paired=False)
        assert decision.paired is True
        assert decision.n_complete_pairs == 3
        # Fully matched -> nothing dropped, only the promotion note.
        assert len(decision.counts) == 6
        assert len(decision.notes) == 1
        assert "Auto-promoted to PAIRED" in decision.notes[0]

    def test_single_matched_donor_is_not_promoted(self):
        # Only one donor contributes both arms: below the >= 2 promotion floor.
        pb = _pb([("d1", "LE"), ("d1", "Normal")])
        decision = _resolve(pb, paired=False)
        assert decision.paired is False
        assert decision.n_complete_pairs == 1
        assert decision.notes == []
        assert len(decision.counts) == 2

    def test_partially_matched_is_not_promoted(self):
        # d1 is matched but d2 appears only in the case arm -> not fully matched.
        pb = _pb(
            [
                ("d1", "LE"),
                ("d1", "Normal"),
                ("d2", "LE"),
            ]
        )
        decision = _resolve(pb, paired=False)
        assert decision.paired is False
        assert decision.n_complete_pairs == 1
        assert decision.notes == []
        # Unpaired stays unrestricted: every pseudo-sample is kept.
        assert len(decision.counts) == 3


class TestPairedRestriction:
    def test_declared_paired_drops_incomplete_donors(self):
        # d1, d2 matched; d3 control-only. A declared paired fit must drop d3.
        pb = _pb(
            [
                ("d1", "LE"),
                ("d1", "Normal"),
                ("d2", "LE"),
                ("d2", "Normal"),
                ("d3", "Normal"),
            ]
        )
        decision = _resolve(pb, paired=True)
        assert decision.paired is True
        assert decision.n_complete_pairs == 2
        # d3 dropped -> four surviving pseudo-samples, none belonging to d3.
        assert len(decision.counts) == 4
        assert set(decision.sample_meta[_DONOR_COL]) == {"d1", "d2"}
        assert "d3" not in set(decision.sample_meta[_DONOR_COL])
        assert len(decision.notes) == 1
        assert "restricted to 2 complete donor pairs" in decision.notes[0]
        assert "'d3'" in decision.notes[0]

    def test_declared_paired_fully_matched_is_unchanged(self):
        pb = _pb(
            [
                ("d1", "LE"),
                ("d1", "Normal"),
                ("d2", "LE"),
                ("d2", "Normal"),
            ]
        )
        decision = _resolve(pb, paired=True)
        assert decision.paired is True
        assert decision.n_complete_pairs == 2
        assert len(decision.counts) == 4
        # No promotion (already paired) and no restriction (nothing incomplete).
        assert decision.notes == []

    def test_extra_unpaired_donor_blocks_promotion(self):
        # d1, d2 matched (two complete pairs) but d3 is case-only, so the design
        # is not FULLY matched: a single unpaired donor blocks auto-promotion.
        pb = _pb(
            [
                ("d1", "LE"),
                ("d1", "Normal"),
                ("d2", "LE"),
                ("d2", "Normal"),
                ("d3", "LE"),
            ]
        )
        decision = _resolve(pb, paired=False)
        assert decision.paired is False
        assert decision.n_complete_pairs == 2
        # Unpaired stays unrestricted despite the lone unpaired donor.
        assert len(decision.counts) == 5
        assert decision.notes == []


def test_returned_frames_are_row_aligned():
    # counts and sample_meta must stay index-aligned after restriction.
    pb = _pb(
        [
            ("d1", "LE"),
            ("d1", "Normal"),
            ("d2", "LE"),
            ("d3", "Normal"),
        ]
    )
    decision = _resolve(pb, paired=True)
    assert list(decision.counts.index) == list(decision.sample_meta.index)
    assert set(decision.sample_meta[_DONOR_COL]) == {"d1"}
    assert decision.n_complete_pairs == 1
