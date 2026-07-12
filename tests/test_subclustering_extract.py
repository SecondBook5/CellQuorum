"""Tests for subclustering focus extraction and group filtering."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.subclustering.extract import apply_group_filter, extract_focus


def make_synthetic_adata() -> ad.AnnData:
    """
    Build a synthetic AnnData for extraction tests.

    Returns:
        AnnData with 200 cells, obs[cell_type] in {A,B},
        obs[donor] in {d1,d2,d3}, layers["counts"], obsm["X_pca"].
    """
    # Build 200-cell count matrix.
    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=5.0, size=(200, 50))

    # Build obs: 100 A cells, 100 B cells.
    obs = pd.DataFrame(
        {
            "cell_type": ["A"] * 100 + ["B"] * 100,
            "donor": (
                ["d1"] * 40
                + ["d2"] * 35
                + ["d3"] * 25  # A cells
                + ["d1"] * 50
                + ["d2"] * 40
                + ["d3"] * 10  # B cells
            ),
        },
        index=[f"cell_{i}" for i in range(200)],
    )

    # Build var.
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(50)])

    # Build AnnData with a stale obsm["X_pca"].
    adata = ad.AnnData(X=counts.astype(float), obs=obs, var=var)
    adata.layers["counts"] = counts
    adata.obsm["X_pca"] = rng.normal(size=(200, 10))

    return adata


def test_extract_focus_subset_to_labels() -> None:
    """Verify extract_focus subsets to focus labels."""
    # Build synthetic adata.
    adata = make_synthetic_adata()

    # Import FocusConfig for the test.
    from cellquorum.subclustering.config import FocusConfig

    focus = FocusConfig(label_key="cell_type", labels=["A"])

    # Extract focus (A cells only).
    result = extract_focus(adata, focus, counts_layer="counts")

    # Verify only A cells remain.
    assert result.n_obs == 100
    assert (result.obs["cell_type"] == "A").all()

    # Verify input unchanged.
    assert adata.n_obs == 200


def test_extract_focus_restores_counts_to_X() -> None:
    """Verify extract_focus restores counts to X."""
    # Build synthetic adata.
    adata = make_synthetic_adata()

    # Import FocusConfig.
    from cellquorum.subclustering.config import FocusConfig

    focus = FocusConfig(label_key="cell_type", labels=["A"])

    # Extract focus.
    result = extract_focus(adata, focus, counts_layer="counts")

    # Verify X == counts.
    assert np.array_equal(result.X, result.layers["counts"])


def test_extract_focus_clears_stale_obsm() -> None:
    """Verify extract_focus deletes stale embeddings."""
    # Build synthetic adata.
    adata = make_synthetic_adata()

    # Import FocusConfig.
    from cellquorum.subclustering.config import FocusConfig

    focus = FocusConfig(label_key="cell_type", labels=["A"])

    # Verify X_pca exists before extraction.
    assert "X_pca" in adata.obsm

    # Extract focus.
    result = extract_focus(adata, focus, counts_layer="counts")

    # Verify X_pca is deleted.
    assert "X_pca" not in result.obsm


def test_extract_focus_records_provenance() -> None:
    """Verify extract_focus records provenance in uns."""
    # Build synthetic adata.
    adata = make_synthetic_adata()

    # Import FocusConfig.
    from cellquorum.subclustering.config import FocusConfig

    focus = FocusConfig(label_key="cell_type", labels=["A"])

    # Extract focus.
    result = extract_focus(adata, focus, counts_layer="counts")

    # Verify provenance recorded.
    assert "subcluster_extraction" in result.uns
    prov = result.uns["subcluster_extraction"]
    assert prov["label_key"] == "cell_type"
    assert prov["labels"] == ["A"]
    assert prov["n_cells_total"] == 200
    assert prov["n_cells_kept"] == 100


def test_extract_focus_empty_labels_no_op() -> None:
    """Verify empty labels list keeps all cells."""
    # Build synthetic adata.
    adata = make_synthetic_adata()

    # Import FocusConfig.
    from cellquorum.subclustering.config import FocusConfig

    focus = FocusConfig(label_key="cell_type", labels=[])

    # Extract focus (no-op).
    result = extract_focus(adata, focus, counts_layer="counts")

    # Verify all cells kept.
    assert result.n_obs == 200


def test_apply_group_filter_drops_low_groups() -> None:
    """Verify apply_group_filter drops groups below min_cells."""
    # Build synthetic adata.
    adata = make_synthetic_adata()

    # Import FocusConfig and extract A cells first.
    from cellquorum.subclustering.config import FocusConfig

    focus = FocusConfig(label_key="cell_type", labels=["A"])
    adata_a = extract_focus(adata, focus, counts_layer="counts")

    # A cells: d1=40, d2=35, d3=25.
    # Apply group filter with min_cells=30 (should drop d3).
    filtered, provenance = apply_group_filter(adata_a, "donor", 30)

    # Verify d3 dropped.
    assert filtered.n_obs == 75  # 40 + 35
    assert "d3" not in filtered.obs["donor"].values
    assert "d1" in filtered.obs["donor"].values
    assert "d2" in filtered.obs["donor"].values

    # Verify provenance.
    assert provenance["applied"] is True
    assert provenance["group_key"] == "donor"
    assert provenance["min_cells"] == 30
    assert provenance["counts"]["d1"] == 40
    assert provenance["counts"]["d2"] == 35
    assert provenance["counts"]["d3"] == 25
    assert set(provenance["kept"]) == {"d1", "d2"}
    assert provenance["dropped"] == ["d3"]

    # Verify input unchanged.
    assert adata_a.n_obs == 100


def test_apply_group_filter_no_op_when_none() -> None:
    """Verify apply_group_filter no-op when group_key is None."""
    # Build synthetic adata.
    adata = make_synthetic_adata()

    # Apply group filter with None group_key.
    filtered, provenance = apply_group_filter(adata, None, 50)

    # Verify no-op.
    assert filtered.n_obs == 200
    assert provenance["applied"] is False

    # Verify input unchanged.
    assert adata.n_obs == 200


def test_apply_group_filter_no_op_when_min_cells_none() -> None:
    """Verify apply_group_filter no-op when min_cells is None."""
    # Build synthetic adata.
    adata = make_synthetic_adata()

    # Apply group filter with None min_cells.
    filtered, provenance = apply_group_filter(adata, "donor", None)

    # Verify no-op.
    assert filtered.n_obs == 200
    assert provenance["applied"] is False


def test_extract_focus_input_immutable() -> None:
    """Verify extract_focus never mutates input."""
    # Build synthetic adata.
    adata = make_synthetic_adata()
    original_obs_size = adata.n_obs
    original_has_pca = "X_pca" in adata.obsm

    # Import FocusConfig.
    from cellquorum.subclustering.config import FocusConfig

    focus = FocusConfig(label_key="cell_type", labels=["A"])

    # Extract focus.
    result = extract_focus(adata, focus, counts_layer="counts")

    # Verify input unchanged.
    assert adata.n_obs == original_obs_size
    assert ("X_pca" in adata.obsm) == original_has_pca
    assert result is not adata


def test_apply_group_filter_input_immutable() -> None:
    """Verify apply_group_filter never mutates input."""
    # Build synthetic adata.
    adata = make_synthetic_adata()
    original_obs_size = adata.n_obs

    # Apply group filter.
    filtered, _ = apply_group_filter(adata, "donor", 50)

    # Verify input unchanged.
    assert adata.n_obs == original_obs_size
    assert filtered is not adata
