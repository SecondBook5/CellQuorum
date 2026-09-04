"""Declared fit scope must change behaviour, not just appear in the registry.

`tests/test_stage_cell_scope.py` proves the *declaration* exists. This proves the
declaration is **obeyed**, which is a different claim and the one that actually matters —
a scope nobody reads is the same failure as a QC verdict nobody reads, one level up.

The case under test is HVG selection, because it is the leak that is easiest to miss:
means, variances and dispersions are cohort statistics computed upstream of PCA, so a
damaged cell pulls the gene set and excluding it from PCA afterwards cannot undo that. The
manifold is already defined on the wrong genes.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.preprocessing.feature_selection.hvg import HVGMethod
from cellquorum.stages.qc.eligibility import Analysis, EligibilityMasks, Permission

FIT_COLUMN = EligibilityMasks.column_name(Analysis.MANIFOLD, Permission.FIT)


def _cohort_with_a_distinct_damaged_signal(
    n_healthy: int = 300,
    n_damaged: int = 60,
    n_genes: int = 300,
    seed: int = 0,
) -> ad.AnnData:
    """Build a cohort where damaged cells express their own private gene block.

    The damaged cells drive ten "STRESS*" genes and nothing else touches them, so those
    genes are highly variable **only** because the damaged cells are present. Fit on every
    cell and all ten are selected; fit on core cells and none can be.

    The ordinary genes are deliberately near-deterministic. An earlier version drew them
    from the same Poisson as everything else, and with 120 identically-distributed genes the
    HVG ranking was sampling noise — the control test selected 0-2 stress genes
    non-monotonically, which would have made the masked assertion vacuous. Giving the
    ordinary genes almost no variance means a bimodal gene has to rank.

    Note also that ``seurat_v3`` variance-stabilises: it *expects* high variance at high
    mean, so a stress block expressed at Poisson(80) is not "highly variable" in its sense.
    Poisson(20) against a near-constant background is what actually leaks, verified across
    seeds.
    """
    rng = np.random.default_rng(seed)
    stress = [f"STRESS{i}" for i in range(10)]
    ordinary = [f"G{i}" for i in range(n_genes - len(stress))]
    genes = [*stress, *ordinary]

    # Near-deterministic background: a per-gene baseline plus a tiny jitter.
    baseline = rng.integers(4, 7, size=n_genes).astype(np.float32)
    matrix = np.tile(baseline, (n_healthy + n_damaged, 1))
    matrix += rng.binomial(1, 0.05, size=matrix.shape).astype(np.float32)

    # The stress block: silent in healthy cells, strongly expressed in damaged ones.
    matrix[:, : len(stress)] = 0.0
    matrix[n_healthy:, : len(stress)] = rng.poisson(20.0, size=(n_damaged, len(stress))).astype(
        np.float32
    )

    obs = pd.DataFrame(
        {
            "is_damaged": [False] * n_healthy + [True] * n_damaged,
            # Core = the healthy cells. This is what QC would have written.
            FIT_COLUMN: [True] * n_healthy + [False] * n_damaged,
        },
        index=[f"cell_{i}" for i in range(len(matrix))],
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = matrix.copy()
    return adata


def _select_hvgs(adata: ad.AnnData) -> set[str]:
    """Run HVG selection and return the selected gene names."""
    HVGMethod()._run(
        adata,
        {"method": "seurat_v3", "n_top_genes": 20, "counts_layer": "counts"},
        context=None,
    )
    return set(adata.var_names[adata.var["highly_variable"].to_numpy(dtype=bool)])


def test_damaged_cells_do_not_shape_the_hvg_set_when_excluded_from_fitting() -> None:
    """The stress block must not be selected when damaged cells may not fit."""
    adata = _cohort_with_a_distinct_damaged_signal()
    selected = _select_hvgs(adata)

    stress_selected = {gene for gene in selected if gene.startswith("STRESS")}
    assert not stress_selected, (
        f"HVG selected {sorted(stress_selected)} — genes that are variable only because the "
        f"damaged cells are present. The fit mask was declared but not honoured."
    )


def test_the_same_cells_do_shape_it_when_the_mask_is_absent() -> None:
    """The control: without the mask the leak is real, so the test above is meaningful.

    Without this, the assertion above could pass for an unrelated reason and nobody would
    know the mask was doing any work.
    """
    adata = _cohort_with_a_distinct_damaged_signal()
    del adata.obs[FIT_COLUMN]
    selected = _select_hvgs(adata)

    stress_selected = {gene for gene in selected if gene.startswith("STRESS")}
    assert len(stress_selected) == 10, (
        f"only {len(stress_selected)}/10 stress genes leaked without the mask, so the masked "
        f"assertion above proves little — strengthen the fixture"
    )


def test_hvg_results_are_written_for_every_gene_not_just_fitted_cells() -> None:
    """Fit on a cell subset, apply to all genes. `var` is gene-level, so nothing is lost."""
    adata = _cohort_with_a_distinct_damaged_signal()
    _select_hvgs(adata)

    assert "highly_variable" in adata.var.columns
    assert len(adata.var["highly_variable"]) == adata.n_vars
    assert adata.var["highly_variable"].notna().all()


def test_all_cells_survive_hvg_selection() -> None:
    """HVG flags genes and must never subset cells, masked or not."""
    adata = _cohort_with_a_distinct_damaged_signal()
    before = adata.n_obs
    _select_hvgs(adata)

    assert adata.n_obs == before


def test_an_empty_fit_population_falls_back_rather_than_fitting_on_nothing() -> None:
    """An all-False mask is a QC misconfiguration, not an instruction to fit on zero cells.

    Falling back to every cell would be a silent wrong answer, so the fallback restores the
    prior behaviour instead of inventing one.
    """
    adata = _cohort_with_a_distinct_damaged_signal()
    adata.obs[FIT_COLUMN] = False

    selected = _select_hvgs(adata)
    assert selected, "HVG produced nothing when the fit population was empty"


def test_a_dataset_without_graded_qc_behaves_as_before() -> None:
    """Absent QC columns must not become a hidden dependency."""
    adata = _cohort_with_a_distinct_damaged_signal()
    del adata.obs[FIT_COLUMN]

    selected = _select_hvgs(adata)
    assert len(selected) == 20
