"""The whole fitted chain must be core-fitted together, not stage by stage.

Each stage has its own proof that it honours the fit population. This file asserts the
property that only shows up when they run in sequence, and that no single-stage test can
establish: running the chain on a cohort with non-core cells present must give the core cells
exactly the result they would have had if those cells had never been in the file.

That composed claim is the one that actually matters scientifically, because the leak the
graded QC model was built to close is cumulative. HVG picks genes, PCA picks axes on those
genes, clustering partitions that space. A leak at any step is inherited by every step after
it, so protecting four stages individually is worth nothing if the composition drifts.

The chain under test is HVG → PCA → clustering. Normalization is exercised separately in
``test_normalization_fits_target_on_core`` because it needs the isolated scclr environment;
here the lognorm layer is built per-cell, which fits nothing and so cannot leak.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.clustering.neighbors_leiden import LABEL_SOURCE_COLUMN, LeidenMethod
from cellquorum.stages.preprocessing.dimensionality.pca import PCAMethod
from cellquorum.stages.preprocessing.feature_selection.hvg import HVGMethod
from cellquorum.stages.qc.eligibility import Analysis, EligibilityMasks, Permission

MANIFOLD_FIT = EligibilityMasks.column_name(Analysis.MANIFOLD, Permission.FIT)
CLUSTERING_FIT = EligibilityMasks.column_name(Analysis.CLUSTERING, Permission.FIT)
LAYER = "cellquorum_normalized"


class _Paths:
    def __init__(self, figures) -> None:
        self.figures = figures


class _Context:
    def __init__(self, figures) -> None:
        self.paths = _Paths(figures)


def _cohort(
    n_per_type: int = 150,
    n_damaged: int = 70,
    n_genes: int = 200,
    *,
    seed: int = 0,
) -> ad.AnnData:
    """Two real cell types plus damaged cells with their own private stress programme.

    The damaged cells are built to attack every link at once: their stress block is variable
    enough to win HVG slots, far enough off-axis to claim a principal component, and dense
    enough to form a Leiden community. So if the chain leaks anywhere, the core cells' result
    moves.
    """
    rng = np.random.default_rng(seed)
    stress = [f"STRESS{i}" for i in range(15)]
    genes = [*stress, *[f"G{i}" for i in range(n_genes - len(stress))]]

    # Two cell types differing in a marker block, on a near-constant background so the
    # biological signal is what HVG should find.
    baseline = rng.integers(4, 7, size=n_genes).astype(np.float32)
    type_a = np.tile(baseline, (n_per_type, 1))
    type_b = np.tile(baseline, (n_per_type, 1))
    type_a[:, 30:45] += rng.poisson(12.0, size=(n_per_type, 15))
    type_b[:, 60:75] += rng.poisson(12.0, size=(n_per_type, 15))

    # Damaged cells: the stress block only they express.
    damaged = np.tile(baseline, (n_damaged, 1))
    damaged[:, : len(stress)] += rng.poisson(20.0, size=(n_damaged, len(stress)))

    counts = np.vstack([type_a, type_b, damaged]).astype(np.float32)
    counts += rng.binomial(1, 0.05, size=counts.shape).astype(np.float32)
    counts[: 2 * n_per_type, : len(stress)] = 0.0  # only damaged cells carry stress

    n_core = 2 * n_per_type
    obs = pd.DataFrame(
        {
            MANIFOLD_FIT: [True] * n_core + [False] * n_damaged,
            CLUSTERING_FIT: [True] * n_core + [False] * n_damaged,
            "is_damaged": [False] * n_core + [True] * n_damaged,
        },
        index=[f"cell_{i}" for i in range(len(counts))],
    )
    adata = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = counts.copy()

    # Per-cell lognorm. No cohort statistic, so nothing here can leak.
    totals = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    adata.layers[LAYER] = np.log1p(counts / totals * 1e4).astype(np.float32)
    return adata


def _run_chain(adata: ad.AnnData, tmp_path) -> None:
    """HVG → PCA → clustering, as the pipeline would run them."""
    HVGMethod()._run(
        adata,
        {"method": "seurat_v3", "n_top_genes": 40, "counts_layer": "counts"},
        context=None,
    )
    PCAMethod()._run(
        adata,
        {
            "input_layer": LAYER,
            "n_pcs": 10,
            "max_pcs": 10,
            "random_state": 0,
            "use_highly_variable": True,
        },
        context=_Context(tmp_path),
    )
    LeidenMethod()._run(
        adata,
        {
            "n_neighbors": 15,
            "resolution": 1.0,
            "random_state": 0,
            "key_added": "leiden",
            "use_rep": "X_pca",
        },
        context=None,
    )


def _core_only(adata: ad.AnnData) -> ad.AnnData:
    """The same cohort as if the damaged cells had never been sequenced."""
    core = adata[adata.obs[MANIFOLD_FIT].to_numpy(bool)].copy()
    del core.obs[MANIFOLD_FIT]
    del core.obs[CLUSTERING_FIT]
    return core


# ═══ The composed property ═════════════════════════════════════════════════════════


def test_the_chain_gives_core_cells_the_result_they_would_have_had_alone(tmp_path) -> None:
    """Gene set, axes and partition must all be identical to a damaged-cell-free run.

    The single strongest statement available about the fitted chain: the presence of
    questionable cells in the file changed nothing about the reference the analysis is built
    on.
    """
    full = _cohort()
    _run_chain(full, tmp_path)

    core = _core_only(_cohort())
    _run_chain(core, tmp_path)

    # 1. The same genes.
    assert set(full.var_names[full.var["highly_variable"].to_numpy(bool)]) == set(
        core.var_names[core.var["highly_variable"].to_numpy(bool)]
    )
    # 2. The same axes.
    np.testing.assert_allclose(full.varm["PCs"], core.varm["PCs"], atol=1e-8)
    # 3. The same partition, cell for cell.
    assert (
        full.obs.loc[core.obs_names, "leiden"].astype(str).to_numpy()
        == core.obs["leiden"].astype(str).to_numpy()
    ).all()


def test_the_control_the_damaged_cells_really_do_change_all_three(tmp_path) -> None:
    """Without the masks each link moves, so the composed test above is not vacuous.

    Checked link by link, because a control that only proved one of the three had shifted
    would leave the other two assertions unsupported.
    """
    unmasked = _cohort()
    del unmasked.obs[MANIFOLD_FIT]
    del unmasked.obs[CLUSTERING_FIT]
    _run_chain(unmasked, tmp_path)

    core = _core_only(_cohort())
    _run_chain(core, tmp_path)

    leaked_genes = set(unmasked.var_names[unmasked.var["highly_variable"].to_numpy(bool)])
    clean_genes = set(core.var_names[core.var["highly_variable"].to_numpy(bool)])
    assert leaked_genes != clean_genes, "the damaged cells did not change the HVG set"

    stress_selected = {gene for gene in leaked_genes if gene.startswith("STRESS")}
    assert stress_selected, "the stress block never won an HVG slot"

    core_rows = np.arange(len(core.obs_names))
    assert not np.allclose(
        unmasked.varm["PCs"][:, 0], core.varm["PCs"][:, 0], atol=1e-6
    ), "the damaged cells did not move PC1"

    unmasked_labels = unmasked.obs["leiden"].astype(str).to_numpy()[core_rows]
    clean_labels = core.obs["leiden"].astype(str).to_numpy()
    assert not (unmasked_labels == clean_labels).all(), "the partition of core cells held"


# ═══ Nothing is lost along the way ═════════════════════════════════════════════════


def test_every_cell_comes_out_of_the_chain_with_coordinates_and_a_label(tmp_path) -> None:
    """Protecting the reference must not quietly delete the cells being protected against."""
    adata = _cohort()
    _run_chain(adata, tmp_path)

    assert adata.obsm["X_pca"].shape[0] == adata.n_obs
    assert np.isfinite(adata.obsm["X_pca"]).all()
    assert adata.obs["leiden"].notna().all()
    assert adata.n_obs == 370


def test_the_damaged_cells_are_marked_as_transferred_not_fitted(tmp_path) -> None:
    """Their labels must be traceable to transfer, so a reader can discount them."""
    adata = _cohort()
    _run_chain(adata, tmp_path)

    damaged = adata.obs["is_damaged"].to_numpy(bool)
    assert (adata.obs[LABEL_SOURCE_COLUMN][damaged] == "transferred").all()
    assert (adata.obs[LABEL_SOURCE_COLUMN][~damaged] == "fitted").all()


def test_the_stress_block_never_enters_the_gene_set(tmp_path) -> None:
    """The specific leak, checked at the end of the chain rather than at HVG alone."""
    adata = _cohort()
    _run_chain(adata, tmp_path)

    selected = adata.var_names[adata.var["highly_variable"].to_numpy(bool)]
    assert not [gene for gene in selected if gene.startswith("STRESS")]


def test_a_dataset_without_graded_qc_runs_the_chain_unchanged(tmp_path) -> None:
    """The whole chain must remain usable on data that never went through graded QC."""
    adata = _cohort()
    del adata.obs[MANIFOLD_FIT]
    del adata.obs[CLUSTERING_FIT]
    _run_chain(adata, tmp_path)

    assert adata.obsm["X_pca"].shape[0] == adata.n_obs
    assert adata.obs["leiden"].notna().all()
