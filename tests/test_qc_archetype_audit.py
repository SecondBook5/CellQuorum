"""The archetype audit must tell a lost population apart from debris, or stay quiet.

Clustering finds dense blobs; archetypal analysis finds extremes. That difference is the whole
reason this exists: Leiden needs density, so a fifty-cell population is exactly what it merges
into a neighbour, while a polytope vertex does not care how few cells support it.

But a vertex alone proves nothing about health, because damage is extreme too. So the audit's
real job is to separate two situations that produce an *identical* exclusion rate and demand
opposite responses:

    coherent + excluded     a real population is being removed        -> investigate
    incoherent + excluded   debris, correctly removed                 -> no action

Measured on the fixture below, coherence separates them by a factor of twenty:

    ordinary archetype   0.96
    rare population      1.00
    debris               0.04

Everything here is skipped unless the isolated partipy environment exists, because partipy is
GPL-3 and CellQuorum is BSD-3 — it is an optional backend, never a dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.qc.archetypes import ArchetypeAudit, audit_archetypes


def _partipy_or_skip():
    """Return an available partipy backend, or skip when its isolated env is absent."""
    from cellquorum.backends.partipy_backend import build_partipy_backend

    backend = build_partipy_backend()
    if not backend.status().available:
        pytest.skip("partipy environment unavailable (isolated micromamba env not built)")
    return backend


def _three_groups(seed: int = 0) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """An embedding with an ordinary bulk, a coherent rare extreme, and incoherent debris."""
    rng = np.random.default_rng(seed)
    n_ordinary, n_rare, n_debris, n_genes = 300, 40, 50, 120

    ordinary = rng.normal(0.0, 1.0, size=(n_ordinary, 8))
    rare = rng.normal(8.0, 0.4, size=(n_rare, 8))
    debris = rng.normal(-8.0, 3.0, size=(n_debris, 8))
    embedding = np.vstack([ordinary, rare, debris])

    # Counts backing the coherence measure: the rare group shares one marker programme, the
    # debris group keeps a different random handful per cell.
    counts = np.zeros((len(embedding), n_genes), dtype=np.float32)
    counts[: n_ordinary + n_rare] = rng.poisson(5.0, size=(n_ordinary + n_rare, n_genes))
    for row in range(n_ordinary + n_rare, len(counts)):
        kept = rng.choice(n_genes, size=rng.integers(8, 20), replace=False)
        counts[row, kept] = rng.poisson(2.0, size=len(kept))

    names = pd.Index([f"cell_{i}" for i in range(len(embedding))])
    return embedding, counts, names


def _excluded(names: pd.Index, start: int) -> pd.Series:
    """Mark every cell from ``start`` onward as barred from fitting."""
    flags = np.zeros(len(names), dtype=bool)
    flags[start:] = True
    return pd.Series(flags, index=names)


# ═══ Availability and graceful absence ═════════════════════════════════════════════


def test_the_audit_reports_itself_unavailable_without_the_env() -> None:
    """A missing optional backend must degrade, never raise.

    partipy cannot be a dependency, so "not installed" is the normal case and has to be a
    quiet, inspectable outcome rather than a failed run.
    """

    class _Absent:
        def status(self):  # noqa: ANN202 - a stub with the one method the audit reads
            from cellquorum.backends.base import BackendStatus

            return BackendStatus(
                name="partipy",
                kind="external",
                available=False,
                requirements=[],
                missing=["partipy"],
                warnings=[],
                details={},
            )

    embedding, counts, names = _three_groups()
    audit = audit_archetypes(embedding, names, _excluded(names, 340), counts, backend=_Absent())

    assert isinstance(audit, ArchetypeAudit)
    assert audit.available is False
    assert audit.reason and "partipy" in audit.reason
    assert audit.table.empty


# ═══ The discrimination that justifies the audit ═══════════════════════════════════


def test_debris_is_called_debris_and_not_a_lost_population() -> None:
    """An incoherent excluded vertex means QC is working, and must not raise an alarm."""
    backend = _partipy_or_skip()
    embedding, counts, names = _three_groups()

    audit = audit_archetypes(
        embedding, names, _excluded(names, 340), counts, backend=backend, n_archetypes_max=5
    )
    assert audit.available, audit.reason

    debris_rows = audit.table[audit.table["coherence"] < 0.5]
    assert not debris_rows.empty, "no incoherent archetype was found"
    assert debris_rows["probably_debris"].all()
    assert not debris_rows["losing_a_population"].any()


def test_a_coherent_excluded_population_raises_the_alarm() -> None:
    """The failure the audit exists to catch: real biology being removed.

    Here the *rare* group is the one excluded, which is the rare-population loss scenario
    rather than the debris scenario, and the verdict must flip accordingly.
    """
    backend = _partipy_or_skip()
    embedding, counts, names = _three_groups()

    # Exclude the coherent rare group (rows 300-339) instead of the debris.
    flags = np.zeros(len(names), dtype=bool)
    flags[300:340] = True
    audit = audit_archetypes(
        embedding,
        names,
        pd.Series(flags, index=names),
        counts,
        backend=backend,
        n_archetypes_max=5,
    )
    assert audit.available, audit.reason

    flagged = audit.flagged()
    assert not flagged.empty, "an excluded coherent population was not flagged"
    assert flagged["losing_a_population"].any()


def test_a_healthy_cohort_flags_nothing() -> None:
    """With nothing excluded there is nothing to report, or the audit is just noise."""
    backend = _partipy_or_skip()
    embedding, counts, names = _three_groups()

    audit = audit_archetypes(
        embedding,
        names,
        pd.Series(False, index=names),
        counts,
        backend=backend,
        n_archetypes_max=5,
    )
    assert audit.available, audit.reason
    assert audit.flagged().empty


# ═══ Shape, provenance, and the significance test ══════════════════════════════════


def test_the_polytope_significance_is_reported() -> None:
    """A polytope can always be fitted, so an unsupported one must be visible as such."""
    backend = _partipy_or_skip()
    embedding, counts, names = _three_groups()

    audit = audit_archetypes(
        embedding, names, _excluded(names, 340), counts, backend=backend, n_archetypes_max=5
    )
    assert audit.n_archetypes >= 2
    assert audit.t_ratio is not None
    assert audit.t_ratio_pvalue is not None


def test_every_cell_receives_a_dominant_archetype() -> None:
    """The per-cell label is written to obs, so it must cover the object."""
    backend = _partipy_or_skip()
    embedding, counts, names = _three_groups()

    audit = audit_archetypes(
        embedding, names, _excluded(names, 340), counts, backend=backend, n_archetypes_max=5
    )
    assert audit.dominant is not None
    assert audit.dominant.index.equals(names)
    assert audit.dominant.notna().all()


def test_the_table_has_one_row_per_archetype_with_the_expected_columns() -> None:
    """The audit is user-facing, so its shape is part of the contract."""
    backend = _partipy_or_skip()
    embedding, counts, names = _three_groups()

    audit = audit_archetypes(
        embedding, names, _excluded(names, 340), counts, backend=backend, n_archetypes_max=5
    )
    assert len(audit.table) == audit.n_archetypes
    assert set(audit.table.columns) == {
        "n_supporting",
        "excluded_fraction",
        "coherence",
        "losing_a_population",
        "probably_debris",
    }


def test_coherence_is_absent_but_harmless_without_counts() -> None:
    """Counts are optional; without them the audit still reports exclusion rates."""
    backend = _partipy_or_skip()
    embedding, _, names = _three_groups()

    audit = audit_archetypes(
        embedding, names, _excluded(names, 340), backend=backend, n_archetypes_max=5
    )
    assert audit.available, audit.reason
    assert audit.table["coherence"].isna().all()
    # Without coherence nothing can be called debris, so an excluded vertex reads as a loss.
    assert not audit.table["probably_debris"].any()


def test_the_audit_is_deterministic() -> None:
    """Same seed, same verdict — a QC audit that moves between runs is not usable."""
    backend = _partipy_or_skip()
    embedding, counts, names = _three_groups()

    first = audit_archetypes(
        embedding, names, _excluded(names, 340), counts, backend=backend, n_archetypes_max=5, seed=0
    )
    second = audit_archetypes(
        embedding, names, _excluded(names, 340), counts, backend=backend, n_archetypes_max=5, seed=0
    )
    assert first.n_archetypes == second.n_archetypes
    pd.testing.assert_series_equal(first.dominant, second.dominant)


# ═══ The significance gate ═════════════════════════════════════════════════════════


def test_nothing_is_flagged_when_the_polytope_is_unsupported() -> None:
    """A polytope can be fitted to anything, so an unsupported one must find nothing.

    Two Gaussian blobs have no simplex structure. Before this gate the audit fitted five
    archetypes to them anyway, reported a t-ratio p-value of 0.92, and still raised two "a real
    population is being removed" alarms — inventing findings out of fit artifacts. An audit that
    cries wolf on structureless data is worse than no audit, because it trains the reader to
    ignore it.
    """
    backend = _partipy_or_skip()
    rng = np.random.default_rng(0)
    embedding = np.vstack([rng.normal(0.0, 1.0, (400, 12)), rng.normal(6.0, 0.6, (80, 12))])
    names = pd.Index([f"cell_{i}" for i in range(len(embedding))])

    audit = audit_archetypes(
        embedding, names, _excluded(names, 400), backend=backend, n_archetypes_max=5
    )
    assert audit.available, audit.reason
    if audit.polytope_supported:
        pytest.skip("this fixture happened to yield a significant polytope")

    assert audit.flagged().empty
    # The table still comes back, so a reader can see what was measured and disregarded.
    assert not audit.table.empty


def test_an_unsupported_polytope_is_distinguishable_from_a_clean_result() -> None:
    """ "Nothing is wrong" and "this could not tell" must not both surface as silence."""
    backend = _partipy_or_skip()
    embedding, counts, names = _three_groups()

    audit = audit_archetypes(
        embedding, names, _excluded(names, 340), counts, backend=backend, n_archetypes_max=5
    )
    assert audit.available, audit.reason
    # The flag exists and is a real boolean, so a caller can branch on it rather than guessing
    # from an empty table.
    assert isinstance(audit.polytope_supported, bool)


def test_the_cell_cap_bounds_the_fit_and_the_audit_still_works() -> None:
    """An uncapped fit hung a 201,923-cell run for 27 minutes at 0% CPU.

    The cap is a correctness property, not a performance tweak: without it the audit can block
    a pipeline indefinitely, which is the worst behaviour available to an optional diagnostic.
    """
    backend = _partipy_or_skip()
    embedding, counts, names = _three_groups()

    audit = audit_archetypes(
        embedding,
        names,
        _excluded(names, 340),
        counts,
        backend=backend,
        n_archetypes_max=5,
        max_cells=150,
    )
    assert audit.available, audit.reason
    # Only the sampled cells receive an archetype; the rest are honestly absent rather than
    # assigned a vertex they were never compared against.
    assert audit.dominant is not None
    assert len(audit.dominant) == 150
    assert audit.dominant.index.isin(names).all()
