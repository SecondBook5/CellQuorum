"""Cross-group ambient specificity: is a gene's presence here real, or leaked-in soup?

SoupX (``ambient_correction``, order=10) runs per LIBRARY, before any cell-type label
exists, so it estimates and removes one contamination fraction per library and cannot
see that a gene is only ever real in ONE cell type and is soup everywhere else. A
keratin gene peaking in keratinocytes at high, specific expression can still show up
at a low, non-specific level in fibroblasts from the same library — SoupX has no way to
attribute that residual to the wrong compartment, because compartments don't exist yet
at the point it runs.

``ambient_gene_mask`` answers the question at the only point it can be answered: once
cell-type labels exist, for a per-cell-type analysis (typically pseudobulk DE), a gene
is ambient for a given group if it BOTH (a) peaks in some OTHER group and (b) is
expressed in the target group at a low fraction of that peak. Guard (a) alone is not
enough: many genes legitimately peak in a closely related cell type and are still real
biology in the target group (ACTA2 peaks in pericytes/SMCs but is genuine myofibroblast
signal in fibroblasts) — guard (b) is what tells the difference, since ambient soup is
both foreign AND weak, whereas shared real biology is foreign but strong.

An optional ``group_map`` coarsens raw groups (cell types) into broader compartments
(e.g. "Fibroblasts" and "Pericyte/SMC" -> "mesenchymal") so a gene peaking in a
biologically related raw group is never even considered foreign in the first place.
There is no default grouping and no built-in vocabulary of compartments — which cell
types are "related enough to share real biology" is a domain judgment call for the
caller to supply, never a hardcoded biology.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def ambient_gene_mask(
    profile: pd.DataFrame,
    *,
    target_group: str,
    spec_threshold: float = 0.25,
    group_map: Mapping[str, str] | None = None,
) -> pd.Series:
    """
    Flag genes to KEEP (True) versus drop as ambient (False) for one target group.

    Args:
        profile: Genes (index) x groups (columns) mean-expression table, non-negative
            (e.g. mean of a normalized expression layer per cell type across a wider
            reference cohort — never just the group being tested, since a gene cannot
            be judged foreign relative to itself).
        target_group: The column in ``profile`` the returned mask applies to.
        spec_threshold: A gene peaking in a foreign group is ambient here only when its
            value in ``target_group`` is strictly below this fraction of its peak.
        group_map: Optional raw-group -> coarse-group mapping. A group absent from the
            mapping is left as its own group (identity fallback). When None, every raw
            group in ``profile`` is its own group.

    Returns:
        Boolean Series indexed like ``profile``: True = keep, False = ambient (drop).

    Raises:
        ValueError: If ``target_group`` is not a column of ``profile``.
    """

    if target_group not in profile.columns:
        raise ValueError(
            f"target_group '{target_group}' is not a column of profile "
            f"(columns: {list(profile.columns)})."
        )

    def resolve(raw_group: str) -> str:
        return group_map.get(raw_group, raw_group) if group_map else raw_group

    peak_group = profile.idxmax(axis=1)
    peak_value = profile.max(axis=1)
    target_value = profile[target_group]

    # 0/0 (a gene undetected in every group) has no peak to be foreign to or
    # specific relative to — never ambient, just uninformative.
    ratio = pd.Series(
        np.divide(
            target_value.to_numpy(),
            peak_value.to_numpy(),
            out=np.full(len(profile), np.nan),
            where=peak_value.to_numpy() > 0,
        ),
        index=profile.index,
    )

    target_resolved = resolve(target_group)
    foreign = peak_group.map(resolve) != target_resolved
    ambient = foreign & ratio.notna() & (ratio < spec_threshold)

    return ~ambient


__all__ = ["ambient_gene_mask"]
