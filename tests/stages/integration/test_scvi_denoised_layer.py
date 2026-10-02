"""The scVI denoised layer must survive HVG-restricted training.

scVI's decoder can only reconstruct genes the model was trained on. Once training was
restricted to ~2,000 highly variable genes, a requested marker outside that set became
undecodable — and the write did not degrade gracefully, it crashed:

    shape mismatch: value array of shape (201871,17) could not be broadcast to
    indexing result of shape (201871,20)

Three of twenty requested markers were not HVGs, the decoder returned the seventeen it knew,
and the destination was sized for twenty. That surfaced after six minutes of GPU training, on
the full 202,000-cell cohort, with no test covering the denoised layer at all.

The fix has two halves and both are pinned here: the training set gains the requested genes so
they are decodable, and the writer indexes by what the model actually knows so an undecodable
gene is reported rather than fatal.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.integration._hvg_selection import _add_decodable_genes
from cellquorum.stages.integration.scvi_methods import _write_denoised_layer

_GENES = ["HVG1", "HVG2", "HVG3", "PECAM1", "COL1A1", "PROX1"]


def _column(adata: ad.AnnData, layer: str, gene: str) -> np.ndarray:
    """One gene's column from the layer, dense, whatever the layer's storage is."""

    block = adata.layers[layer][:, adata.var_names.get_loc(gene)]
    return np.asarray(block.todense() if hasattr(block, "todense") else block).ravel()


def _adata(n_obs: int = 12) -> ad.AnnData:
    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, size=(n_obs, len(_GENES))).astype("float32")
    adata = ad.AnnData(
        X=counts,
        var=pd.DataFrame(index=pd.Index(_GENES)),
        obs=pd.DataFrame(index=pd.Index([f"cell{i}" for i in range(n_obs)])),
    )
    adata.layers["counts"] = counts.copy()
    # HVG1-3 are variable; the three markers are not — the realistic case, since broadly
    # expressed lineage markers have high means and low relative variance.
    adata.var["highly_variable"] = [True, True, True, False, False, False]
    return adata


class _FakeSCVI:
    """A decoder that, like the real one, only knows the genes it was trained on."""

    def __init__(self, trained_genes: list[str]) -> None:
        self._trained = trained_genes

    def get_normalized_expression(self, work, *, gene_list, library_size, return_mean):  # noqa: ANN001, ANN003, ARG002
        usable = [gene for gene in gene_list if gene in self._trained]
        assert usable == list(gene_list), (
            "the caller asked the decoder for genes it was never trained on: "
            f"{[g for g in gene_list if g not in self._trained]}"
        )
        values = np.ones((work.n_obs, len(usable)), dtype="float32")
        return pd.DataFrame(values, index=work.obs_names, columns=usable)


# ═══ Half one: the training set must cover the panel ═══════════════════════════════


def test_requested_markers_are_added_to_the_training_set() -> None:
    """A marker that missed the HVG cut is added, because otherwise it is undecodable."""

    adata = _adata()
    mask = adata.var["highly_variable"].to_numpy(dtype=bool)

    extended, added = _add_decodable_genes(adata, mask, ["PECAM1", "PROX1"])

    assert added == ["PECAM1", "PROX1"]
    assert int(extended.sum()) == 5
    for gene in ("HVG1", "HVG2", "HVG3", "PECAM1", "PROX1"):
        assert extended[adata.var_names.get_loc(gene)]
    assert not extended[adata.var_names.get_loc("COL1A1")], "an unrequested gene was added"


def test_the_original_mask_is_not_mutated() -> None:
    """The HVG flag is shared state; scVI extending it must not change what PCA sees."""

    adata = _adata()
    mask = adata.var["highly_variable"].to_numpy(dtype=bool)
    before = mask.copy()

    _add_decodable_genes(adata, mask, ["PECAM1"])

    assert np.array_equal(mask, before)


def test_a_marker_that_is_already_variable_is_not_reported_as_added() -> None:
    """The note names what training actually gained, so it must not inflate."""

    adata = _adata()
    mask = adata.var["highly_variable"].to_numpy(dtype=bool)

    extended, added = _add_decodable_genes(adata, mask, ["HVG1", "PECAM1"])

    assert added == ["PECAM1"]
    assert int(extended.sum()) == 4


def test_no_panel_means_no_change() -> None:
    adata = _adata()
    mask = adata.var["highly_variable"].to_numpy(dtype=bool)

    extended, added = _add_decodable_genes(adata, mask, [])

    assert added == []
    assert np.array_equal(extended, mask)


# ═══ Half two: the writer must index by what the model knows ═══════════════════════


def test_the_shape_mismatch_does_not_recur() -> None:
    """The regression, at unit scale: 3 of 6 requested genes outside the trained set.

    Before the fix the writer built its destination from ``adata.var_names`` while the decoder
    answered from the trained subset, so the assignment was 3-wide into a 6-wide slot.
    """

    adata = _adata()
    work = adata[:, adata.var["highly_variable"].to_numpy(dtype=bool)].copy()
    model = _FakeSCVI(list(work.var_names))

    notes = _write_denoised_layer(
        adata,
        work,
        model=model,
        layer="denoised",
        genes=_GENES,
        library_size=1e4,
    )

    assert adata.layers["denoised"].shape == (adata.n_obs, adata.n_vars)
    # The three trained genes carry decoded values; the untrained three stay zero.
    for gene in ("HVG1", "HVG2", "HVG3"):
        assert np.all(_column(adata, "denoised", gene) == 1.0)
    for gene in ("PECAM1", "COL1A1", "PROX1"):
        assert np.all(_column(adata, "denoised", gene) == 0.0)
    # And it says so, rather than leaving a silently blank panel row.
    assert any("not among the genes scVI trained on" in note for note in notes)
    assert any("PECAM1" in note for note in notes)


def test_the_whole_panel_decodes_when_training_covered_it() -> None:
    """The two halves together: extend the mask, then every requested gene is filled."""

    adata = _adata()
    mask = adata.var["highly_variable"].to_numpy(dtype=bool)
    extended, _ = _add_decodable_genes(adata, mask, _GENES)
    work = adata[:, extended].copy()
    model = _FakeSCVI(list(work.var_names))

    notes = _write_denoised_layer(
        adata, work, model=model, layer="denoised", genes=_GENES, library_size=1e4
    )

    for gene in _GENES:
        assert np.all(_column(adata, "denoised", gene) == 1.0), gene
    assert not any("not among the genes" in note for note in notes)


def test_the_layer_is_sparse(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Dense, this layer is 27 GB on the real cohort to hold 20 genes.

    The layer spans the whole gene space so a figure can address a marker by name, but only the
    named genes carry values. Materializing that densely made the post-annotation checkpoint
    41.6 GB and left the process holding ~35 GB right before reference_mapping tried to load a
    24.8 GB atlas — on a 54 GB machine, which is where the run died.
    """
    import scipy.sparse as sp

    adata = _adata()
    work = adata.copy()
    model = _FakeSCVI(list(work.var_names))

    _write_denoised_layer(
        adata, work, model=model, layer="denoised", genes=["HVG1", "PROX1"], library_size=1e4
    )

    layer = adata.layers["denoised"]
    assert sp.issparse(layer), "a dense layer here is 27 GB on the real cohort"
    # Only the two named genes are stored, not n_obs x n_vars.
    assert layer.nnz <= adata.n_obs * 2
    assert np.all(_column(adata, "denoised", "HVG1") == 1.0)
    assert np.all(_column(adata, "denoised", "PROX1") == 1.0)
    assert np.all(_column(adata, "denoised", "COL1A1") == 0.0)


def test_the_sparse_layer_survives_a_round_trip(tmp_path: Path) -> None:
    """It has to still be sparse after h5ad write/read, since that is what blew up the disk."""
    import scipy.sparse as sp

    adata = _adata()
    _write_denoised_layer(
        adata,
        adata.copy(),
        model=_FakeSCVI(list(adata.var_names)),
        layer="denoised",
        genes=["HVG1", "PROX1"],
        library_size=1e4,
    )
    path = tmp_path / "round_trip.h5ad"
    adata.write_h5ad(path)

    back = ad.read_h5ad(path)
    assert sp.issparse(back.layers["denoised"])
    assert np.all(_column(back, "denoised", "HVG1") == 1.0)


def test_the_layer_is_tagged_imputed() -> None:
    """Decoder output is model output; the tag is what keeps it out of DE and abundance."""

    from cellquorum.core.contracts.magic_guard import assert_not_imputed

    adata = _adata()
    work = adata.copy()
    model = _FakeSCVI(list(work.var_names))

    _write_denoised_layer(
        adata, work, model=model, layer="denoised", genes=["HVG1"], library_size=1e4
    )

    with pytest.raises(Exception, match="imputed"):
        assert_not_imputed(adata, "denoised")


def test_a_gene_absent_from_the_object_is_named() -> None:
    """A typo in the panel must be visible, and distinguishable from an untrained gene."""

    adata = _adata()
    work = adata.copy()
    model = _FakeSCVI(list(work.var_names))

    notes = _write_denoised_layer(
        adata,
        work,
        model=model,
        layer="denoised",
        genes=["HVG1", "NOT_A_GENE"],
        library_size=1e4,
    )

    assert any("absent from the object" in note and "NOT_A_GENE" in note for note in notes)


def test_nothing_decodable_is_refused_with_both_counts() -> None:
    """The error separates "you misspelled it" from "the model never saw it"."""

    adata = _adata()
    work = adata[:, ["HVG1"]].copy()
    model = _FakeSCVI(["HVG1"])

    with pytest.raises(CellQuorumDataError, match="absent from the object"):
        _write_denoised_layer(
            adata,
            work,
            model=model,
            layer="denoised",
            genes=["PECAM1", "NOT_A_GENE"],
            library_size=1e4,
        )


def test_an_empty_panel_is_refused() -> None:
    """Decoding every gene would be ~26 GB dense on the real cohort."""

    adata = _adata()
    with pytest.raises(CellQuorumDataError, match="denoised_genes"):
        _write_denoised_layer(
            adata,
            adata.copy(),
            model=_FakeSCVI(list(adata.var_names)),
            layer="denoised",
            genes=[],
            library_size=1e4,
        )
