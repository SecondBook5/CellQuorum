"""Do these gene sets say the same thing? Set algebra with the guards that make it honest.

Any analysis that scores several gene sets eventually has to answer whether the sets are
independent readouts or restatements of each other, and the answer is always presented one
of two ways: an exclusive-membership table ("69 genes are in mitotic spindle *only*") or a
pairwise similarity. Both are easy to compute and easy to get wrong, in four specific ways
this module refuses to reproduce.

**A similarity without a test is not a result.** Jaccard 0.052 between two sets means
nothing on its own: whether it is more overlap than chance depends entirely on how large
the sets are and how many genes they were drawn from. Two real pairs can have Jaccard 0.052
at p = 0.0003 and Jaccard 0.047 at p = 0.8. Ranking by similarity therefore ranks pairs
wrong, so :func:`set_overlap_tests` always returns the test beside the coefficient.

**The test needs a universe, and no default is safe.** The hypergeometric p-value is a
function of how many genes *could* have been shared. Take the universe to be the union of
the sets themselves and every overlap looks significant, because the union is the smallest
defensible universe and the smallest universe gives the largest enrichment. So ``universe``
is a required argument with no default: the honest choice is the set of genes that were
actually testable in the experiment (detected genes, or the annotation the sets were drawn
from), and it has to be stated rather than inferred.

**Genes outside the universe are a warning, not noise.** A set whose members are mostly
absent from the universe is usually a species, alias, or annotation mismatch rather than a
biological statement. Each set is restricted to the universe before testing — a gene that
could not be drawn must not count toward the draw — and the number dropped is reported per
pair so a mismatch surfaces instead of quietly changing every p-value.

**Self-comparisons are excluded.** A set overlaps itself perfectly at p = 0. Leaving those
rows in inflates the family that multiple testing corrects over, which makes every real
pair look better.

One thing the guards cannot supply, because it is the caller's judgement
-----------------------------------------------------------------------
The hypergeometric null is "these two sets were drawn at random from the universe". That
is the right null for **independently derived** sets — two DE lists from two experiments,
a marker set against a curated module — whose members really were drawn from the detected
pool by unrelated processes, and there the p-value is the instrument to reach for. When the
two sides are two different *families* rather than one family against itself — a curated
panel against this run's leading edges — that is :func:`cross_overlap_tests`.

It is the wrong instrument for two sets **selected off one shared ordering**, even though
each is data-derived. Every leading edge in a GSEA run is a deterministic function of the
same ranked gene list, so two same-direction pathways both take the extreme end of that
one list: a gene their annotations already share and that ranks high is in both leading
edges by construction, and the right tail duly returns a very small number for a
structural fact. That question — how much of an overlap the annotation already forced —
is what :func:`selection_overlap` answers, descriptively and without a p-value.

It is close to vacuous for **hand-curated** pathway modules. Those were written by
someone reading one literature, so they overlap more than random draws by construction;
a panel of curated endothelial modules will return fold enrichments in the hundreds at
tiny FDRs, and reporting that as a finding dresses up a foregone conclusion. What is not
foregone for curated sets is *how much of each set is its own*, which is what a panel's
independence actually rests on and which needs no null at all — so read
:func:`set_sizes`' ``fraction_exclusive`` and :func:`exclusive_combinations`, and treat
the test as a floor check. A module with zero exclusive members is a restatement of its
neighbours no matter what its p-value says. And when the two sides are two *curated*
families rather than one — the identity panels a cluster was named with against the modules
whose remodeling is then tested in it — the question is how much of the naming the tested
side already contains, which is :func:`cross_membership`: the same intersection, both
directional fractions, and no null at all.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import pandas as pd

from cellquorum.stats.module_remodeling import bh_fdr

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

OVERLAP_COLUMNS: tuple[str, ...] = (
    "set_a",
    "set_b",
    "size_a",
    "size_b",
    "intersection",
    "jaccard",
    "overlap_coefficient",
    "expected",
    "fold_enrichment",
    "p_value",
    "fdr",
    "universe_size",
    "dropped_a",
    "dropped_b",
    "reason",
)

COMBINATION_COLUMNS: tuple[str, ...] = ("combination", "n_sets", "size", "elements")

CROSS_COLUMNS: tuple[str, ...] = (
    "left",
    "right",
    "size_left",
    "size_right",
    "intersection",
    "jaccard",
    "overlap_coefficient",
    "expected",
    "fold_enrichment",
    "p_value",
    "fdr",
    "universe_size",
    "dropped_left",
    "dropped_right",
    "shared_elements",
    "reason",
)

SELECTION_COLUMNS: tuple[str, ...] = (
    "set_a",
    "set_b",
    "reference_a",
    "reference_b",
    "reference_intersection",
    "selected_a",
    "selected_b",
    "selected_intersection",
    "captured_a",
    "captured_b",
    "fraction_of_reference_intersection",
    "expected_if_unrelated",
    "fold_over_unrelated",
    "shared_elements",
    "reason",
)

#: Columns of :func:`cross_membership`. Deliberately no p-value column: see that function.
MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "left",
    "right",
    "size_left",
    "size_right",
    "intersection",
    "fraction_of_left",
    "fraction_of_right",
    "jaccard",
    "shared_elements",
)

# How the exclusive table reads: everything that belongs to exactly one set first, largest
# first, then the pairs, and so on. Matches how such tables are read aloud — "these are
# only in A, these are shared by A and B" — rather than by raw size across all degrees.
_COMBINATION_SORT = ["n_sets", "size"]


def _as_sets(sets: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    return {str(name): {str(element) for element in members} for name, members in sets.items()}


def set_overlap_tests(
    sets: Mapping[str, Iterable[str]],
    *,
    universe: Collection[str],
    fdr_method: str = "fdr_bh",
) -> pd.DataFrame:
    """
    Test every pair of sets for more overlap than chance, against a stated universe.

    Args:
        sets: Name to members. Order is preserved in the output's pair ordering.
        universe: The elements that could have been shared — genes detected in the
            experiment, or the annotation the sets were drawn from. Required, because the
            p-value is a function of it and the convenient default (the union of the sets)
            is the one that makes every overlap look significant.
        fdr_method: Passed to :func:`cellquorum.stats.module_remodeling.bh_fdr`.

    Returns:
        One row per unordered pair, with the intersection, two similarity coefficients,
        the overlap expected under random draws, the fold enrichment, a one-sided
        hypergeometric p-value (the right tail — the question is always "more than
        chance"), and a BH-corrected FDR over the pairs. ``dropped_a`` / ``dropped_b``
        count members that were not in the universe, and ``reason`` explains any row
        whose test was not attempted.

    Raises:
        ValueError: Fewer than two sets, or an empty universe.
    """
    members = _as_sets(sets)
    if len(members) < 2:
        raise ValueError(
            f"need at least 2 sets to compare, got {len(members)}: "
            "a pairwise overlap table of one set is empty"
        )

    background = {str(element) for element in universe}
    if not background:
        raise ValueError(
            "the universe is empty; pass the elements that could have been shared "
            "(the detected genes, or the annotation the sets came from)"
        )
    universe_size = len(background)

    restricted = {name: genes & background for name, genes in members.items()}
    dropped = {name: len(members[name]) - len(restricted[name]) for name in members}

    rows: list[dict[str, object]] = []
    for name_a, name_b in combinations(list(members), 2):
        a, b = restricted[name_a], restricted[name_b]
        row: dict[str, object] = {
            "set_a": name_a,
            "set_b": name_b,
            **_overlap_row(a, b, universe_size),
            "dropped_a": dropped[name_a],
            "dropped_b": dropped[name_b],
        }
        if not a or not b:
            empty = name_a if not a else name_b
            row["reason"] = (
                f"{empty!r} has no members inside the universe, so there was nothing to "
                "draw; check that the sets and the universe use the same gene naming"
            )
        rows.append(row)

    table = pd.DataFrame(rows, columns=list(OVERLAP_COLUMNS))
    testable = table["p_value"].notna()
    if testable.any():
        table.loc[testable, "fdr"] = bh_fdr(
            table.loc[testable, "p_value"].to_numpy(dtype=float), method=fdr_method
        )
    return table


def cross_overlap_tests(
    left: Mapping[str, Iterable[str]],
    right: Mapping[str, Iterable[str]],
    *,
    universe: Collection[str],
    fdr_method: str = "fdr_bh",
    list_elements: bool = True,
) -> pd.DataFrame:
    """
    Test every set in one family against every set in another, against a stated universe.

    The question :func:`set_overlap_tests` cannot ask, because its family is one mapping
    compared with itself: *does this curated panel sit where the data moved?* One family is
    written from the literature, the other is derived from the experiment — a curated module
    against a GSEA leading edge, a marker panel against a DE list — and because the two were
    produced by unrelated processes, the hypergeometric null ("both were drawn at random
    from the detected pool") is the right one and its p-value means something.

    That is the whole condition on using this function, and it is worth stating in the
    negative. Two curated families overlap by construction, and the test returns fold
    enrichments in the hundreds for a foregone conclusion (see this module's header).
    Two families both selected off *one* ranking are deterministic, not sampled, and belong
    in :func:`selection_overlap`, which reports the same overlap with no p-value. This
    function is for the mixed case: one side data-derived, one side not.

    ``shared_elements`` is reported because on a hit it is the finding. A curated
    actomyosin module overlapping a pathway labelled "mitotic spindle" is only interpretable
    once you can see that the shared genes are the myosins and the filamins, and a
    fold enrichment on its own would leave a reader to trust the pathway's label.

    Args:
        left: Name to members, for the first family — conventionally the curated one.
        right: Name to members, for the second family.
        universe: The elements that could have been shared. Required, for the reason given
            in :func:`set_overlap_tests`.
        fdr_method: Passed to :func:`cellquorum.stats.module_remodeling.bh_fdr`. BH is
            corrected over the full ``left`` x ``right`` family; its members are not
            independent of each other, which BH tolerates.
        list_elements: Fill ``shared_elements``. ``False`` for a compact table.

    Returns:
        One row per (left, right) pair, in the order the two mappings were given, with the
        same counted quantities, similarity coefficients, right-tail p-value and BH FDR
        that :func:`set_overlap_tests` reports, plus the shared elements.

    Raises:
        ValueError: Either family is empty, or the universe is empty.
    """
    left_sets, right_sets = _as_sets(left), _as_sets(right)
    if not left_sets or not right_sets:
        empty = "left" if not left_sets else "right"
        raise ValueError(f"the {empty} family is empty; there is nothing to compare against")

    background = {str(element) for element in universe}
    if not background:
        raise ValueError(
            "the universe is empty; pass the elements that could have been shared "
            "(the detected genes, or the annotation the sets came from)"
        )

    restricted_left = {name: genes & background for name, genes in left_sets.items()}
    restricted_right = {name: genes & background for name, genes in right_sets.items()}

    rows: list[dict[str, object]] = []
    for name_left, a in restricted_left.items():
        for name_right, b in restricted_right.items():
            row = _overlap_row(a, b, len(background))
            row = {
                "left": name_left,
                "right": name_right,
                "size_left": row.pop("size_a"),
                "size_right": row.pop("size_b"),
                **row,
                "dropped_left": len(left_sets[name_left]) - len(a),
                "dropped_right": len(right_sets[name_right]) - len(b),
                "shared_elements": ", ".join(sorted(a & b)) if list_elements else "",
            }
            if not a or not b:
                empty = name_left if not a else name_right
                row["reason"] = (
                    f"{empty!r} has no members inside the universe, so there was nothing "
                    "to draw; check that the sets and the universe use the same gene naming"
                )
            rows.append(row)

    table = pd.DataFrame(rows, columns=list(CROSS_COLUMNS))
    testable = table["p_value"].notna()
    if testable.any():
        table.loc[testable, "fdr"] = bh_fdr(
            table.loc[testable, "p_value"].to_numpy(dtype=float), method=fdr_method
        )
    return table


def _overlap_row(a: set[str], b: set[str], universe_size: int) -> dict[str, object]:
    """The counted quantities and the test for one pair, already restricted to the universe.

    Shared by the within-family and cross-family tables so the two cannot drift apart on
    what "fold enrichment" means. A pair with nothing inside the universe gets NaNs and no
    test; the caller says which set was empty, since only it knows the family names.
    """
    shared = len(a & b)
    union = len(a | b)
    smaller = min(len(a), len(b))
    row: dict[str, object] = {
        "size_a": len(a),
        "size_b": len(b),
        "intersection": shared,
        "jaccard": (shared / union) if union else float("nan"),
        "overlap_coefficient": (shared / smaller) if smaller else float("nan"),
        "expected": float("nan"),
        "fold_enrichment": float("nan"),
        "p_value": float("nan"),
        "fdr": float("nan"),
        "universe_size": universe_size,
        "reason": "",
    }
    if a and b:
        expected = len(a) * len(b) / universe_size
        row["expected"] = expected
        row["fold_enrichment"] = (shared / expected) if expected > 0 else float("nan")
        row["p_value"] = _hypergeometric_right_tail(shared, universe_size, len(a), len(b))
    return row


def _hypergeometric_right_tail(shared: int, universe: int, size_a: int, size_b: int) -> float:
    """P(overlap >= observed) for two independent draws from ``universe``.

    The right tail, not two-sided: the question a concordance table asks is whether two
    sets share *more* than chance. A two-sided p-value would also flag pairs that are
    suspiciously disjoint, which is a different claim and would be read as concordance.
    """
    from scipy.stats import hypergeom

    # sf(k-1) rather than 1 - cdf(k-1): the survival function keeps its precision in the
    # far tail, where these p-values live.
    return float(hypergeom.sf(shared - 1, universe, size_a, size_b))


def exclusive_combinations(
    sets: Mapping[str, Iterable[str]],
    *,
    min_size: int = 1,
    list_elements: bool = True,
) -> pd.DataFrame:
    """
    Every observed combination of sets, with the elements belonging to exactly that group.

    This is the table that reads "69 genes are in mitotic spindle only, 5 are shared by
    apical junction and mitotic spindle" — the exclusive intersections behind an UpSet
    plot. Combinations that no element occupies are omitted rather than listed as zero,
    because the number of possible combinations grows as 2^n and a table of mostly zeros
    hides the ones that matter.

    Args:
        sets: Name to members. Order fixes the order names appear within a combination
            label, so the labels are stable across runs.
        min_size: Drop combinations occupied by fewer than this many elements. The
            default keeps everything observed.
        list_elements: Write the members into an ``elements`` column, comma-separated.
            Turn off for a size-only table.

    Returns:
        One row per occupied combination: the ``combination`` label, how many sets it
        spans (``n_sets``), its ``size``, and the ``elements`` themselves. Sorted by
        ``n_sets`` ascending then ``size`` descending — exclusive members first, largest
        first.

    Raises:
        ValueError: No sets were given.
    """
    members = _as_sets(sets)
    if not members:
        raise ValueError("no sets given; there are no combinations to enumerate")

    order = list(members)
    occupancy: dict[tuple[str, ...], list[str]] = {}
    for element in sorted(set().union(*members.values()) if members else []):
        key = tuple(name for name in order if element in members[name])
        if key:
            occupancy.setdefault(key, []).append(element)

    rows = [
        {
            "combination": " & ".join(key),
            "n_sets": len(key),
            "size": len(elements),
            "elements": ", ".join(elements) if list_elements else "",
        }
        for key, elements in occupancy.items()
        if len(elements) >= min_size
    ]
    table = pd.DataFrame(rows, columns=list(COMBINATION_COLUMNS))
    if table.empty:
        return table
    return table.sort_values(_COMBINATION_SORT, ascending=[True, False], kind="stable").reset_index(
        drop=True
    )


def selection_overlap(
    selected: Mapping[str, Iterable[str]],
    reference: Mapping[str, Iterable[str]],
    *,
    universe: Collection[str] | None = None,
    list_elements: bool = True,
) -> pd.DataFrame:
    """
    When two selections overlap, how much of that did the annotation already force?

    Each ``selected[name]`` was picked out of ``reference[name]`` — a GSEA leading edge out
    of its pathway, the significant genes of a module out of the module, the markers kept
    for a lineage out of the candidates. Two such selections sharing genes is routinely
    read as independent convergence, and usually is not: if the two reference sets already
    shared those genes, the selections had no choice. This table puts both numbers on the
    same row so the reader can tell which they are looking at.

    **There is deliberately no p-value here, and that is the point of the function.** For
    selections made off one shared ordering — every leading edge in a GSEA run is a
    function of the same ranked gene list — there is no sampling under which to place a
    null. Two same-direction pathways both take the extreme end of that one list, so a gene
    their annotations share and that ranks high is in both leading edges *by construction*.
    A hypergeometric right tail applied here returns a very small number for a structural
    fact, which is why :func:`set_overlap_tests` is the wrong instrument for this question
    even though a leading edge is data-derived.

    Args:
        selected: Name to the elements actually picked.
        reference: Name to the set each selection was picked from. Must cover every key in
            ``selected``.
        universe: Optional; both mappings are restricted to it before counting, so a
            selection and its reference are compared over the same pool.
        list_elements: Write the shared selected elements into ``shared_elements``,
            comma-separated. Turn off for a counts-only table.

    Returns:
        One row per unordered pair. ``reference_intersection`` is what the annotation
        shared; ``selected_intersection`` is what both selections actually took;
        ``captured_a`` / ``captured_b`` are how many of the shared reference elements each
        selection took on its own; ``fraction_of_reference_intersection`` is the share of
        the annotation overlap that both took. ``expected_if_unrelated`` is
        ``captured_a * captured_b / reference_intersection`` — a descriptive reference
        point for reading ``selected_intersection``, not a null, for the reason above.

    Raises:
        ValueError: Fewer than two selections, a selection with no reference set, or a
            selection that is not contained in its reference set (which means the two
            mappings were not built from the same source).
    """
    picked = _as_sets(selected)
    from_sets = _as_sets(reference)
    if len(picked) < 2:
        raise ValueError(
            f"need at least 2 selections to compare, got {len(picked)}: "
            "a pairwise table of one selection is empty"
        )

    unreferenced = [name for name in picked if name not in from_sets]
    if unreferenced:
        raise ValueError(
            f"no reference set for {unreferenced}: every selection must name the set it "
            "was picked from, or the overlap it shares cannot be attributed"
        )

    background = {str(element) for element in universe} if universe is not None else None
    if background is not None:
        picked = {name: genes & background for name, genes in picked.items()}
        from_sets = {name: genes & background for name, genes in from_sets.items()}

    escaped = {
        name: sorted(genes - from_sets[name])
        for name, genes in picked.items()
        if genes - from_sets[name]
    }
    if escaped:
        raise ValueError(
            f"selection is not contained in its reference set: {escaped} — the two mappings "
            "were not built from the same source, so no overlap here is attributable"
        )

    rows: list[dict[str, object]] = []
    for name_a, name_b in combinations(list(picked), 2):
        selected_a, selected_b = picked[name_a], picked[name_b]
        reference_shared = from_sets[name_a] & from_sets[name_b]
        selected_shared = selected_a & selected_b
        captured_a = len(selected_a & reference_shared)
        captured_b = len(selected_b & reference_shared)
        shared_count = len(reference_shared)
        expected = (captured_a * captured_b / shared_count) if shared_count else float("nan")
        rows.append(
            {
                "set_a": name_a,
                "set_b": name_b,
                "reference_a": len(from_sets[name_a]),
                "reference_b": len(from_sets[name_b]),
                "reference_intersection": shared_count,
                "selected_a": len(selected_a),
                "selected_b": len(selected_b),
                "selected_intersection": len(selected_shared),
                "captured_a": captured_a,
                "captured_b": captured_b,
                "fraction_of_reference_intersection": (
                    len(selected_shared) / shared_count if shared_count else float("nan")
                ),
                "expected_if_unrelated": expected,
                "fold_over_unrelated": (
                    len(selected_shared) / expected if expected and expected > 0 else float("nan")
                ),
                "shared_elements": (", ".join(sorted(selected_shared)) if list_elements else ""),
                "reason": (
                    ""
                    if shared_count
                    else "the reference sets are disjoint, so no overlap was available to take"
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(SELECTION_COLUMNS))


def set_sizes(
    sets: Mapping[str, Iterable[str]], *, universe: Collection[str] | None = None
) -> pd.DataFrame:
    """
    Each set's size, and how much of it is exclusive to it.

    The sidebar of an UpSet plot, and on its own the quickest read on whether a panel of
    sets is a panel or a family: a set with 40 members of which 4 are exclusive is not an
    independent readout of anything.

    Args:
        sets: Name to members.
        universe: Optional; when given, sizes are counted after restricting to it and
            ``outside_universe`` reports what that cost.

    Returns:
        One row per set with ``size``, ``exclusive`` (members in no other set),
        ``shared``, ``fraction_exclusive``, and ``outside_universe``.
    """
    members = _as_sets(sets)
    background = {str(element) for element in universe} if universe is not None else None

    rows = []
    for name, genes in members.items():
        inside = genes & background if background is not None else genes
        others: set[str] = set()
        for other, other_genes in members.items():
            if other != name:
                others |= other_genes & background if background is not None else other_genes
        exclusive = inside - others
        rows.append(
            {
                "set": name,
                "size": len(inside),
                "exclusive": len(exclusive),
                "shared": len(inside) - len(exclusive),
                "fraction_exclusive": (len(exclusive) / len(inside)) if inside else float("nan"),
                "outside_universe": len(genes) - len(inside),
            }
        )
    return pd.DataFrame(rows)


def cross_membership(
    left: Mapping[str, Iterable[str]], right: Mapping[str, Iterable[str]]
) -> pd.DataFrame:
    """Shared membership between two families, both directions, and no p-value.

    The circularity check. Two curated families — the identity panels a cluster was
    *named* with and the modules whose remodeling is then *tested* in it, or a QC panel
    against an outcome module — routinely share genes, and if they share enough then the
    second analysis is partly a restatement of the first. That is a question about
    membership, not about chance, and none of the tested functions here will answer it
    honestly: :func:`set_overlap_tests` compares one family with itself,
    :func:`cross_overlap_tests` needs one side to be data-derived for its null to mean
    anything (two curated families overlap by construction and it returns fold
    enrichments in the hundreds for a foregone conclusion), and
    :func:`selection_overlap` assumes each selection was drawn out of its own reference.

    So this reports the intersection and nothing inferential, with both directional
    fractions because they carry different consequences. ``fraction_of_left`` is how much
    of the naming panel the tested module already contains; ``fraction_of_right`` is how
    much of the tested module the naming panel does. A three-gene panel wholly inside a
    forty-gene module is a naming decision that leans entirely on the module, even though
    the module is barely touched.

    Args:
        left: First family, name to members. Conventionally the naming side.
        right: Second family, name to members. Conventionally the tested side.

    Returns:
        One row per pair that shares at least one member, with ``left``, ``right``,
        ``size_left``, ``size_right``, ``intersection``, ``fraction_of_left``,
        ``fraction_of_right``, ``jaccard`` and ``shared_elements``; sorted by descending
        ``fraction_of_left`` so the most compromised naming decision reads first. Pairs
        sharing nothing are omitted — the absence of a row is the clean case.
    """
    left_sets, right_sets = _as_sets(left), _as_sets(right)
    rows = []
    for left_name, left_genes in left_sets.items():
        for right_name, right_genes in right_sets.items():
            shared = left_genes & right_genes
            if not shared:
                continue
            union = left_genes | right_genes
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "size_left": len(left_genes),
                    "size_right": len(right_genes),
                    "intersection": len(shared),
                    "fraction_of_left": len(shared) / len(left_genes),
                    "fraction_of_right": len(shared) / len(right_genes),
                    "jaccard": len(shared) / len(union),
                    "shared_elements": ";".join(sorted(shared)),
                }
            )
    table = pd.DataFrame(rows, columns=list(MEMBERSHIP_COLUMNS))
    return table.sort_values(
        ["fraction_of_left", "intersection"], ascending=False, kind="stable"
    ).reset_index(drop=True)


__all__ = [
    "COMBINATION_COLUMNS",
    "CROSS_COLUMNS",
    "MEMBERSHIP_COLUMNS",
    "OVERLAP_COLUMNS",
    "SELECTION_COLUMNS",
    "cross_membership",
    "cross_overlap_tests",
    "exclusive_combinations",
    "selection_overlap",
    "set_overlap_tests",
    "set_sizes",
]
