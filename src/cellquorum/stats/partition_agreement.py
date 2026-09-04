"""Do two clusterings of the same cells say the same thing? Agreement with its direction.

Every project accumulates more than one partition of the same cells — a Leiden run at one
resolution, a second at another, a CHOIR or sc-SHC partition, a hand-curated subtype label,
the labels a previous analysis was keyed on. Results get written against whichever partition
was current, and sooner or later a table keyed on one has to be read beside a figure keyed
on another. This module is for that moment, and it refuses three specific ways it usually
goes wrong.

**A single agreement index cannot distinguish disagreement from refinement.** An adjusted
Rand index near 0.4 is returned both by two partitions that genuinely cut the data
differently and by an eight-cluster partition that nests almost perfectly inside a
three-label one. Those are opposite conclusions: the first says the two analyses are about
different things, the second says one is the other at higher resolution and the tables can
be mapped. So every pair carries the two directional purities beside the symmetric indices.
A partition that is a strict refinement of another has purity 1.0 in that direction and
much less than 1.0 in the reverse, and the asymmetry is the answer.

**Partitions of "the same cells" usually are not.** They come from different objects with
different QC, different subset rules, or different runs, so the barcode sets overlap without
matching. Aligning two such labels with pandas silently drops the difference and reports an
index over whatever survived. :func:`align_partitions` therefore returns a coverage table
alongside the aligned frame: how many cells each partition labelled, how many are shared,
and how many are its own. An agreement index computed over 60% of the cells is a different
statement from one computed over all of them, and the reader has to be told which.

**Unlabelled cells are not a cluster.** A partition read off ``obs`` routinely carries NaN
for cells the labelling did not reach — a gated subcluster, a cell below a margin threshold.
Treating those as a shared category invents agreement wherever both partitions failed to
label the same cell. Cells missing from either side of a pair are dropped from that pair,
per pair rather than globally, so one sparse partition does not shrink the others.

What this module does not do is decide which partition is right. That is a question about
the data and the claim, not about the two label vectors: a partition rejected by a formal
significance test can still be the one a per-cluster effect size was computed on, and the
consequence is a claim hierarchy rather than a number.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Columns of :func:`align_partitions`' coverage table.
COVERAGE_COLUMNS = [
    "partition",
    "n_labelled",
    "n_shared",
    "n_only_here",
    "n_unlabelled_in_shared",
    "n_clusters",
]

#: Columns of :func:`partition_agreement`.
AGREEMENT_COLUMNS = [
    "partition_a",
    "partition_b",
    "n_cells",
    "n_clusters_a",
    "n_clusters_b",
    "ari",
    "ami",
    "purity_a_in_b",
    "purity_b_in_a",
]

#: Columns of :func:`partition_crosstab`.
CROSSTAB_COLUMNS = [
    "partition_a",
    "partition_b",
    "cluster_a",
    "cluster_b",
    "n_cells",
    "frac_of_a",
    "frac_of_b",
]


def _as_labels(values: pd.Series) -> pd.Series:
    """A label vector as nullable strings, with the engine's absent markers as NA.

    Categorical, integer and object columns all turn up as partitions, and comparing them
    requires one representation. The empty string and ``"nan"`` are treated as absent
    because a partition round-tripped through CSV brings them back that way, and a cell
    whose label is the string ``"nan"`` is an unlabelled cell, not a cluster named nan.
    """
    labels = pd.Series(values).astype("object")
    labels = labels.where(~pd.isna(labels), other=None)
    labels = labels.map(lambda v: None if v is None else str(v))
    return labels.replace({"": None, "nan": None, "NaN": None, "None": None})


def align_partitions(
    partitions: Mapping[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Line up several partitions of the same cells on the cells they share.

    Args:
        partitions: Partition name -> per-cell label, each indexed by cell id (barcode).
            Indices need not match; the intersection is what agreement is computed over.

    Returns:
        ``(aligned, coverage)``. ``aligned`` is a frame indexed by the shared cell ids with
        one column per partition, labels as strings and unlabelled cells as ``None``.
        ``coverage`` has :data:`COVERAGE_COLUMNS`, one row per partition, and is the record
        of what the intersection cost — ``n_only_here`` is the cells that partition labels
        and no comparison can use.

    Raises:
        ValueError: If fewer than two partitions are given, or the intersection is empty.
    """
    if len(partitions) < 2:
        raise ValueError("partition agreement needs at least two partitions")

    labelled = {name: _as_labels(values) for name, values in partitions.items()}
    shared = None
    for series in labelled.values():
        index = pd.Index(series.index)
        shared = index if shared is None else shared.intersection(index)
    if shared is None or len(shared) == 0:
        raise ValueError(
            "the partitions share no cell ids — check that they came from objects with "
            "the same barcode convention before reading any agreement index"
        )

    aligned = pd.DataFrame({name: series.reindex(shared) for name, series in labelled.items()})
    coverage = pd.DataFrame(
        [
            {
                "partition": name,
                "n_labelled": int(series.notna().sum()),
                "n_shared": int(len(shared)),
                "n_only_here": int(len(series.index.difference(shared))),
                "n_unlabelled_in_shared": int(aligned[name].isna().sum()),
                "n_clusters": int(aligned[name].dropna().nunique()),
            }
            for name, series in labelled.items()
        ],
        columns=COVERAGE_COLUMNS,
    )
    return aligned, coverage


def _purity(a: pd.Series, b: pd.Series) -> float:
    """Weighted mean, over clusters of ``a``, of the fraction in that cluster's modal ``b``.

    One if every cluster of ``a`` sits entirely inside a single cluster of ``b`` — i.e. if
    ``a`` is a refinement of ``b`` — and it is deliberately not symmetric, because that
    asymmetry is what tells a refinement from a disagreement.
    """
    counts = pd.crosstab(a, b)
    if counts.empty:
        return float("nan")
    return float(counts.max(axis=1).sum() / counts.to_numpy().sum())


def partition_agreement(aligned: pd.DataFrame) -> pd.DataFrame:
    """All-pairs agreement over the columns of an aligned partition frame.

    Args:
        aligned: Output of :func:`align_partitions` (or any frame of per-cell labels).

    Returns:
        One row per unordered pair with :data:`AGREEMENT_COLUMNS`. ``ari`` and ``ami`` are
        the symmetric indices; read them with the two purities, since a high
        ``purity_a_in_b`` beside a low ``purity_b_in_a`` is a refinement rather than a
        disagreement however modest the ARI. ``n_cells`` is the cells labelled by *both*
        members of that pair, which is not in general the height of ``aligned``.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    rows = []
    for name_a, name_b in combinations(aligned.columns, 2):
        pair = aligned[[name_a, name_b]].dropna()
        a, b = pair[name_a], pair[name_b]
        if len(pair) == 0 or a.nunique() == 0 or b.nunique() == 0:
            rows.append(
                {
                    "partition_a": name_a,
                    "partition_b": name_b,
                    "n_cells": int(len(pair)),
                    "n_clusters_a": int(a.nunique()),
                    "n_clusters_b": int(b.nunique()),
                    "ari": float("nan"),
                    "ami": float("nan"),
                    "purity_a_in_b": float("nan"),
                    "purity_b_in_a": float("nan"),
                }
            )
            continue
        codes_a = pd.factorize(a, sort=True)[0]
        codes_b = pd.factorize(b, sort=True)[0]
        rows.append(
            {
                "partition_a": name_a,
                "partition_b": name_b,
                "n_cells": int(len(pair)),
                "n_clusters_a": int(a.nunique()),
                "n_clusters_b": int(b.nunique()),
                "ari": float(adjusted_rand_score(codes_a, codes_b)),
                "ami": float(adjusted_mutual_info_score(codes_a, codes_b)),
                "purity_a_in_b": _purity(a, b),
                "purity_b_in_a": _purity(b, a),
            }
        )
    return pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)


def partition_crosstab(aligned: pd.DataFrame, partition_a: str, partition_b: str) -> pd.DataFrame:
    """The cell-by-cell contingency of two partitions, long form with both fractions.

    Both fractions are returned because they answer different questions and a table with
    one of them invites the other to be guessed. ``frac_of_a`` says where a cluster of
    ``partition_a`` went — the mapping direction — while ``frac_of_b`` says what a cluster
    of ``partition_b`` is made of. A cell block that is 100% of a small cluster of ``a``
    and 4% of a large cluster of ``b`` is a very different fact under the two readings.

    Args:
        aligned: Output of :func:`align_partitions`.
        partition_a: Column read down the rows.
        partition_b: Column read across the columns.

    Returns:
        One row per non-empty ``(cluster_a, cluster_b)`` cell with
        :data:`CROSSTAB_COLUMNS`, sorted by ``cluster_a`` then descending ``n_cells`` so
        each source cluster's dominant destination reads first.
    """
    pair = aligned[[partition_a, partition_b]].dropna()
    counts = pd.crosstab(pair[partition_a], pair[partition_b])
    if counts.empty:
        return pd.DataFrame(columns=CROSSTAB_COLUMNS)

    row_totals = counts.sum(axis=1)
    col_totals = counts.sum(axis=0)
    rows = []
    for cluster_a in counts.index:
        for cluster_b in counts.columns:
            n = int(counts.loc[cluster_a, cluster_b])
            if n == 0:
                continue
            rows.append(
                {
                    "partition_a": partition_a,
                    "partition_b": partition_b,
                    "cluster_a": cluster_a,
                    "cluster_b": cluster_b,
                    "n_cells": n,
                    "frac_of_a": float(n / row_totals[cluster_a]),
                    "frac_of_b": float(n / col_totals[cluster_b]),
                }
            )
    table = pd.DataFrame(rows, columns=CROSSTAB_COLUMNS)
    return table.sort_values(
        ["cluster_a", "n_cells"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)


def cluster_group_support(
    cluster_labels: pd.Series,
    groups: pd.Series,
    *,
    min_cells_per_group: int = 10,
) -> pd.DataFrame:
    """How many independent units each cluster is actually built from.

    A per-cluster effect size is only as replicated as the number of donors (or samples,
    or batches) contributing cells to that cluster, and a cluster's cell count says
    nothing about that: two thousand cells from two donors support a weaker claim than
    two hundred from nine. ``min_cells_per_group`` is applied because a group
    contributing three cells is not a unit a mixed model can estimate a random intercept
    from, so it is counted separately rather than in the headline number.

    Args:
        cluster_labels: Per-cell cluster id.
        groups: Per-cell replication unit (donor, sample, batch), aligned to
            ``cluster_labels``.
        min_cells_per_group: Cells a group needs in a cluster to count as supporting it.

    Returns:
        One row per cluster with ``cluster``, ``n_cells``, ``n_groups``,
        ``n_groups_supporting``, ``max_group_frac`` and ``groups_supporting`` (a
        semicolon-joined list). ``max_group_frac`` near one is a cluster that is one
        donor's cells wearing a cluster's name.
    """
    frame = pd.DataFrame(
        {"cluster": _as_labels(cluster_labels).to_numpy(), "group": _as_labels(groups).to_numpy()}
    ).dropna()
    rows = []
    for cluster, block in frame.groupby("cluster", sort=True):
        per_group = block["group"].value_counts()
        supporting = per_group[per_group >= min_cells_per_group]
        rows.append(
            {
                "cluster": cluster,
                "n_cells": int(len(block)),
                "n_groups": int(len(per_group)),
                "n_groups_supporting": int(len(supporting)),
                "max_group_frac": float(per_group.max() / len(block)),
                "groups_supporting": "; ".join(map(str, sorted(supporting.index))),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "cluster",
            "n_cells",
            "n_groups",
            "n_groups_supporting",
            "max_group_frac",
            "groups_supporting",
        ],
    )


def label_composition(
    cluster_labels: pd.Series, condition: pd.Series, *, order: list[str] | None = None
) -> pd.DataFrame:
    """Per-cluster cell counts split by condition, with both directions' fractions.

    The counterpart to :func:`cluster_group_support` for the other design column: a
    subtype-count table of the kind a reference paper prints is exactly this, and the
    fraction it omits is usually the one that matters. ``frac_within_cluster`` is the
    cluster's own composition; ``frac_of_condition`` is what share of that condition's
    cells the cluster holds, which is the abundance reading.

    Args:
        cluster_labels: Per-cell cluster id.
        condition: Per-cell condition, aligned to ``cluster_labels``.
        order: Condition values, in the order the columns should read. Defaults to
            sorted order. Values not present are kept as zero columns so two arms'
            tables stay column-compatible.

    Returns:
        One row per cluster: ``cluster``, ``n_cells``, then ``n_<condition>``,
        ``frac_within_cluster_<condition>`` and ``frac_of_condition_<condition>`` per
        condition value.
    """
    frame = pd.DataFrame(
        {
            "cluster": _as_labels(cluster_labels).to_numpy(),
            "condition": _as_labels(condition).to_numpy(),
        }
    ).dropna()
    values = order if order is not None else sorted(frame["condition"].unique())
    counts = pd.crosstab(frame["cluster"], frame["condition"]).reindex(columns=values, fill_value=0)
    out = pd.DataFrame({"cluster": counts.index, "n_cells": counts.sum(axis=1).to_numpy()})
    totals = counts.sum(axis=0)
    for value in values:
        out[f"n_{value}"] = counts[value].to_numpy()
        out[f"frac_within_cluster_{value}"] = np.where(
            out["n_cells"].to_numpy() > 0, counts[value].to_numpy() / out["n_cells"].to_numpy(), 0.0
        )
        out[f"frac_of_condition_{value}"] = (
            counts[value].to_numpy() / totals[value] if totals[value] else 0.0
        )
    return out.reset_index(drop=True)


__all__ = [
    "AGREEMENT_COLUMNS",
    "COVERAGE_COLUMNS",
    "CROSSTAB_COLUMNS",
    "align_partitions",
    "cluster_group_support",
    "label_composition",
    "partition_agreement",
    "partition_crosstab",
]
