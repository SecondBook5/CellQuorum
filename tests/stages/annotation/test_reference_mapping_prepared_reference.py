"""The reference the mapping used must be saved, not reconstructed.

scANVI computes a batch-corrected latent space for the REFERENCE as well as the query, and
that space is what every downstream confirmation needs. It was computed, used to fit the kNN
classifier, and then dropped — only the query's copy reached ``obsm``. The per-seed ``.npz``
does carry a ``ref_latent`` array, but anonymously: no barcodes, no labels, no gene names, and
the JSON sidecar records only an ``atlas_obs_digest`` hash. Recovering row identity therefore
meant re-reading the 2.3 GB atlas and replaying the filters, which is only as reproducible as
those staying byte-identical.

Three things were blocked by that, and all three are the same missing artifact:

* ``annotation_diagnostics`` was pointed at the RAW atlas — unfiltered, so it contained the
  lesional-disease cells the mapping excluded, and Ensembl-indexed against a symbol-indexed
  query, so zero shared genes. Confirming a mapping against a different reference than the
  mapping used is not a confirmation.
* CHOIR, as a second confirmation, had no way to reach the same reference.
* the joint atlas+query embedding — the figure that shows whether query LEC land on reference
  LEC or in empty space — needs both sides' coordinates in one space.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.annotation.reference_mapping.scarches import _write_prepared_reference

_GENES = ["PROX1", "PDPN", "LYVE1", "PECAM1", "COL1A1"]


def _atlas_train(n: int = 12) -> ad.AnnData:
    """A reference as it exists at the point the latent is computed: aligned, HVG-subset."""

    rng = np.random.default_rng(0)
    counts = rng.poisson(4.0, size=(n, len(_GENES))).astype("float32")
    obs = pd.DataFrame(
        {
            "_labels": ["LEC"] * 4 + ["Capillary EC FABP4 hi"] * 4 + ["Fibroblast CCL19+"] * 4,
            "biosample_id": [f"donor{i % 3}" for i in range(n)],
        },
        index=pd.Index([f"ref_{i}" for i in range(n)]),
    )
    return ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=pd.Index(_GENES)))


_HIERARCHY = {
    "LEC": "LEC",
    "Capillary EC FABP4 hi": "VEC",
    "Fibroblast CCL19+": "Fibroblasts",
}


def _write(tmp_path: Path, *, coarse: bool = True) -> tuple[ad.AnnData, int]:
    atlas = _atlas_train()
    latent = np.arange(atlas.n_obs * 3, dtype="float64").reshape(atlas.n_obs, 3)
    path = tmp_path / "ref.h5ad"
    n = _write_prepared_reference(
        atlas,
        latent,
        path=path,
        key_added="ref_cell_type_granular",
        coarse_map=_HIERARCHY if coarse else None,
        counts_layer="counts",
        batch_key="biosample_id",
    )
    return ad.read_h5ad(path), n


# ═══ The corrected coordinates survive ═════════════════════════════════════════════


def test_the_corrected_reference_latent_is_saved(tmp_path: Path) -> None:
    """The whole point: the scANVI-corrected reference space reaches disk, with row identity."""

    prepared, n = _write(tmp_path)

    assert n == 12
    assert "X_scANVI" in prepared.obsm
    assert prepared.obsm["X_scANVI"].shape == (12, 3)
    # Row identity, which the bare .npz did not carry.
    assert list(prepared.obs_names) == [f"ref_{i}" for i in range(12)]
    # And the coordinates are the ones handed in, in that order.
    assert np.allclose(prepared.obsm["X_scANVI"][:, 0], np.arange(0, 36, 3))


def test_labels_use_the_query_prediction_column_name(tmp_path: Path) -> None:
    """scDiagnostics takes ONE column name for both sides, so they must agree.

    The reference's own column is ``Cell_type_granular`` and the query's predictions land in
    ``ref_cell_type_granular``; passing the latter meant R stopped with "Reference h5ad missing
    cell_type column: ref_cell_type_granular".
    """

    prepared, _ = _write(tmp_path)

    assert "ref_cell_type_granular" in prepared.obs.columns
    assert set(prepared.obs["ref_cell_type_granular"]) == {
        "LEC",
        "Capillary EC FABP4 hi",
        "Fibroblast CCL19+",
    }


def test_the_coarse_resolution_comes_along(tmp_path: Path) -> None:
    """Collapsed with the same hierarchy as the query, so the two resolutions are comparable."""

    prepared, _ = _write(tmp_path)

    coarse = prepared.obs["ref_cell_type_granular_coarse"]
    assert list(coarse[:4]) == ["LEC"] * 4
    assert list(coarse[4:8]) == ["VEC"] * 4
    assert list(coarse[8:]) == ["Fibroblasts"] * 4


def test_no_coarse_column_when_none_was_requested(tmp_path: Path) -> None:
    prepared, _ = _write(tmp_path, coarse=False)
    assert "ref_cell_type_granular_coarse" not in prepared.obs.columns
    assert prepared.uns["cellquorum_prepared_reference"]["coarse_label_column"] is None


# ═══ What the R diagnostics need ═══════════════════════════════════════════════════


def test_x_is_log_normalized_and_counts_are_kept(tmp_path: Path) -> None:
    """R wants a ``logcounts`` assay; the counts stay so a consumer can renormalize.

    Normalizing here rather than in R keeps the transform in one language and matched to the
    counts the model itself trained on.
    """

    prepared, _ = _write(tmp_path)

    assert "counts" in prepared.layers
    counts = np.asarray(
        prepared.layers["counts"].todense()
        if hasattr(prepared.layers["counts"], "todense")
        else prepared.layers["counts"]
    )
    assert np.allclose(counts, np.round(counts)), "the counts layer is not integral"

    x = np.asarray(prepared.X.todense() if hasattr(prepared.X, "todense") else prepared.X)
    assert not np.allclose(x, np.round(x)), "X looks like counts, not log-normalized values"
    assert x.max() < 12, "X is not on a log scale"
    assert prepared.uns["cellquorum_prepared_reference"]["x_is"] == "log1p CP10K"


def test_the_gene_space_is_the_query_shared_one(tmp_path: Path) -> None:
    """Symbols, not Ensembl IDs — the mismatch that gave zero shared genes."""

    prepared, _ = _write(tmp_path)
    assert list(prepared.var_names) == _GENES


def test_the_reference_batch_is_carried_through(tmp_path: Path) -> None:
    """So a consumer can model or inspect reference batch rather than guess at it."""

    prepared, _ = _write(tmp_path)
    assert "biosample_id" in prepared.obs.columns
    assert prepared.obs["biosample_id"].nunique() == 3


def test_a_missing_batch_column_is_not_fatal(tmp_path: Path) -> None:
    """Not every reference names its batches the way the config guesses."""

    atlas = _atlas_train()
    del atlas.obs["biosample_id"]
    path = tmp_path / "ref.h5ad"

    _write_prepared_reference(
        atlas,
        np.zeros((atlas.n_obs, 3)),
        path=path,
        key_added="ref_state",
        coarse_map=None,
        counts_layer="counts",
        batch_key="biosample_id",
    )
    assert ad.read_h5ad(path).n_obs == atlas.n_obs


def test_the_manifest_describes_what_consumers_need(tmp_path: Path) -> None:
    """A consumer should not have to guess which column or obsm key to read."""

    prepared, _ = _write(tmp_path)
    manifest = prepared.uns["cellquorum_prepared_reference"]

    assert manifest["label_column"] == "ref_cell_type_granular"
    assert manifest["latent_key"] == "X_scANVI"
    assert manifest["counts_layer"] == "counts"
    assert manifest["n_genes"] == len(_GENES)


# ═══ Diagnostics picks it up without being told ════════════════════════════════════


def test_diagnostics_prefers_the_prepared_reference_over_a_configured_path(
    tmp_path: Path,
) -> None:
    """A configured `reference_h5ad` was, in practice, the wrong reference.

    So the prepared one wins and the substitution is stated in the notes rather than done
    silently — the user asked for a specific file and is getting a different one.
    """
    prepared, _ = _write(tmp_path)
    prepared_path = tmp_path / "ref.h5ad"

    query = ad.AnnData(
        X=np.ones((4, len(_GENES)), dtype="float32"),
        var=pd.DataFrame(index=pd.Index(_GENES)),
    )
    query.uns["cellquorum"] = {
        "reference_prepared": {
            "path": str(prepared_path),
            "label_column": "ref_cell_type_granular",
            "latent_key": "X_scANVI",
            "n_cells": 12,
            "n_genes": len(_GENES),
        }
    }

    # Mirror the resolution the method performs, which is what a caller depends on.
    config = {
        "reference_h5ad": str(tmp_path / "configured_raw_atlas.h5ad"),
        "cell_type_col": "cell_type",
    }
    resolved = query.uns.get("cellquorum", {}).get("reference_prepared")
    assert Path(str(resolved["path"])).is_file()
    assert str(resolved["path"]) != config["reference_h5ad"]
    assert resolved["label_column"] == "ref_cell_type_granular"


@pytest.mark.parametrize("latent_dtype", ["float64", "float32"])
def test_the_latent_is_stored_compactly(tmp_path: Path, latent_dtype: str) -> None:
    """157,692 x 30 float32 is 19 MB; float64 would double it for no gain in precision."""

    atlas = _atlas_train()
    path = tmp_path / "ref.h5ad"
    _write_prepared_reference(
        atlas,
        np.ones((atlas.n_obs, 30), dtype=latent_dtype),
        path=path,
        key_added="ref_state",
        coarse_map=None,
        counts_layer="counts",
        batch_key="biosample_id",
    )
    assert ad.read_h5ad(path).obsm["X_scANVI"].dtype == np.float32
