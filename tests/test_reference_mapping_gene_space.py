"""A reference and a query must be put into the same gene naming before intersecting.

Cell Ranger writes gene symbols into ``var_names``. CellxGene writes Ensembl IDs and demotes
the symbol to ``var['feature_name']``. Intersecting the two directly gives **zero** genes.

This stage used to answer that with a ``MethodSkip``: the 2.3 GB atlas loaded, matched nothing,
the stage reported "no shared genes" in one log line, and the run continued to the confirmation
stages as though mapping had happened. A run whose entire purpose was atlas mapping therefore
did no atlas mapping, and the first visible symptom was a *later* stage failing on the missing
``ref_cell_type_granular`` column.

So two properties are pinned here: the identifier is resolved by evidence rather than
convention, and a genuinely non-overlapping pair fails loudly instead of opting out.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.annotation.reference_mapping.scarches import _align_atlas_gene_space

_SYMBOLS = ["PROX1", "PDPN", "LYVE1", "PECAM1", "COL1A1", "KRT14"]
_ENSEMBL = [f"ENSG{i:011d}" for i in range(len(_SYMBOLS))]


def _cellxgene_style_atlas() -> ad.AnnData:
    """An atlas indexed the way CellxGene actually ships one."""

    var = pd.DataFrame({"feature_name": _SYMBOLS}, index=pd.Index(_ENSEMBL))
    return ad.AnnData(X=np.ones((4, len(_SYMBOLS)), dtype="float32"), var=var)


# ═══ Resolving the identifier ══════════════════════════════════════════════════════


def test_an_ensembl_indexed_atlas_is_re_indexed_onto_symbols() -> None:
    """The production case: 0 shared genes becomes all of them."""

    atlas = _cellxgene_style_atlas()
    query = pd.Index(_SYMBOLS)
    assert len(set(atlas.var_names) & set(query)) == 0, "fixture is not the failing case"

    aligned, column, n_shared = _align_atlas_gene_space(atlas, query, symbol_column=None)

    assert column == "feature_name"
    assert n_shared == len(_SYMBOLS)
    assert list(aligned.var_names) == _SYMBOLS


def test_the_original_identifier_is_preserved_for_provenance() -> None:
    """Which Ensembl ID a symbol came from must remain recoverable."""

    aligned, _, _ = _align_atlas_gene_space(
        _cellxgene_style_atlas(), pd.Index(_SYMBOLS), symbol_column=None
    )
    assert list(aligned.var["_original_var_names"]) == _ENSEMBL


def test_an_already_matching_atlas_is_left_alone() -> None:
    """No re-index when var_names is already the shared identifier."""

    var = pd.DataFrame({"feature_name": [f"other_{s}" for s in _SYMBOLS]}, index=pd.Index(_SYMBOLS))
    atlas = ad.AnnData(X=np.ones((4, len(_SYMBOLS)), dtype="float32"), var=var)

    aligned, column, n_shared = _align_atlas_gene_space(
        atlas, pd.Index(_SYMBOLS), symbol_column=None
    )

    assert column is None
    assert n_shared == len(_SYMBOLS)
    assert list(aligned.var_names) == _SYMBOLS


def test_the_column_is_chosen_by_measured_overlap_not_by_name() -> None:
    """Two plausible columns, and the one that actually matches wins.

    Preferring a conventional name would pick ``feature_name`` here and share one gene. The
    choice has to come from the data, because a reference is someone else's file.
    """

    var = pd.DataFrame(
        {
            "feature_name": ["PROX1"] + [f"junk{i}" for i in range(len(_SYMBOLS) - 1)],
            "gene_symbol": _SYMBOLS,
        },
        index=pd.Index(_ENSEMBL),
    )
    atlas = ad.AnnData(X=np.ones((4, len(_SYMBOLS)), dtype="float32"), var=var)

    _, column, n_shared = _align_atlas_gene_space(atlas, pd.Index(_SYMBOLS), symbol_column=None)

    assert column == "gene_symbol"
    assert n_shared == len(_SYMBOLS)


def test_duplicate_symbols_collapse_to_one_gene_each() -> None:
    """Several Ensembl IDs share a symbol, and CellxGene keeps them all.

    A duplicated index makes the later ``atlas[:, shared]`` subset ambiguous, so the first
    occurrence wins and the rest are dropped deterministically.
    """

    ids = ["ENSG1", "ENSG2", "ENSG3"]
    var = pd.DataFrame({"feature_name": ["PROX1", "PROX1", "PDPN"]}, index=pd.Index(ids))
    atlas = ad.AnnData(X=np.arange(9, dtype="float32").reshape(3, 3), var=var)

    aligned, column, n_shared = _align_atlas_gene_space(
        atlas, pd.Index(["PROX1", "PDPN"]), symbol_column=None
    )

    assert column == "feature_name"
    assert list(aligned.var_names) == ["PROX1", "PDPN"]
    assert n_shared == 2
    assert aligned.var["_original_var_names"].tolist() == ["ENSG1", "ENSG3"]


# ═══ Explicit requests ═════════════════════════════════════════════════════════════


def test_an_explicit_column_is_used() -> None:
    aligned, column, _ = _align_atlas_gene_space(
        _cellxgene_style_atlas(), pd.Index(_SYMBOLS), symbol_column="feature_name"
    )
    assert column == "feature_name"
    assert list(aligned.var_names) == _SYMBOLS


def test_an_explicit_column_that_does_not_exist_is_refused() -> None:
    """A named column that silently fell back to detection would hide the misconfiguration."""

    with pytest.raises(CellQuorumDataError, match="not a column of the atlas var"):
        _align_atlas_gene_space(_cellxgene_style_atlas(), pd.Index(_SYMBOLS), symbol_column="nope")


# ═══ Genuinely disjoint ════════════════════════════════════════════════════════════


def test_no_candidate_column_leaves_the_atlas_untouched() -> None:
    """Detection failing must return the truth (0 shared) for the caller to raise on.

    The caller turns this into a hard error naming both identifier styles. Returning a
    re-indexed atlas here, or skipping, is what produced a silent no-op run.
    """

    atlas = ad.AnnData(
        X=np.ones((4, 3), dtype="float32"),
        var=pd.DataFrame(index=pd.Index(["ENSG1", "ENSG2", "ENSG3"])),
    )

    aligned, column, n_shared = _align_atlas_gene_space(
        atlas, pd.Index(_SYMBOLS), symbol_column=None
    )

    assert column is None
    assert n_shared == 0
    assert list(aligned.var_names) == ["ENSG1", "ENSG2", "ENSG3"]


# ═══ Against the real atlas ════════════════════════════════════════════════════════


@pytest.mark.integration
def test_the_real_skin_atlas_aligns_onto_cell_ranger_symbols() -> None:
    """The actual reference, against actual Cell Ranger gene symbols.

    The unit fixtures encode what CellxGene is believed to do; this checks it. Without the
    alignment the intersection is literally zero, so this is the test that would have caught
    the 3-hour no-op run.
    """
    from _external_data import require_external_file

    path = require_external_file(
        "CELLQUORUM_TEST_SKIN_ATLAS",
        what="the CellxGene atopic-dermatitis skin atlas .h5ad",
    )
    atlas = ad.read_h5ad(path, backed="r")

    # A handful of real Cell Ranger symbols, including the four lineages of interest.
    query = pd.Index(["PROX1", "PDPN", "LYVE1", "PECAM1", "CLDN5", "COL1A1", "DCN", "KRT14"])

    assert len(set(atlas.var_names) & set(query)) == 0, "atlas is no longer Ensembl-indexed"

    _, column, n_shared = _align_atlas_gene_space(atlas.to_memory(), query, symbol_column=None)

    assert column == "feature_name"
    assert n_shared == len(query), "not every marker survived the re-index"
