"""PCA must fit its basis on core cells and project everyone else onto it.

HVG was the easy half of the cohort-derived-quantity rule: its result lands in ``var``, so
fitting on a subset and writing back to every gene loses nothing. PCA is the half that
actually needs a design, because its result is *per cell*. Fitting on core alone and stopping
there would leave every borderline cell without coordinates, which would silently drop them
from clustering, annotation and every UMAP — a worse outcome than the leak.

So the rule here is fit-then-project, and these tests pin the two claims that make it sound:

    1. Non-core cells cannot move the basis     (no leak)
    2. Non-core cells still receive coordinates (no silent loss)

The second is what distinguishes "projected into the manifold" from "excluded from it", and
it is the distinction the whole eligibility model rests on.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.stages.preprocessing.dimensionality.pca import (
    PCAMethod,
    project_onto_fitted_basis,
)
from cellquorum.stages.qc.eligibility import Analysis, EligibilityMasks, Permission

FIT_COLUMN = EligibilityMasks.column_name(Analysis.MANIFOLD, Permission.FIT)
LAYER = "cellquorum_normalized"


class _Paths:
    """Minimal stand-in for the pipeline context's paths object."""

    def __init__(self, figures: Path) -> None:
        self.figures = figures


class _Context:
    """Minimal stand-in for the pipeline context."""

    def __init__(self, figures: Path) -> None:
        self.paths = _Paths(figures)


def _cohort(
    n_core: int = 200,
    n_outlier: int = 40,
    n_genes: int = 80,
    *,
    sparse: bool = False,
    seed: int = 0,
) -> ad.AnnData:
    """A cohort whose non-core cells have a wildly different covariance structure.

    The outliers are not merely noisier — they load a private gene block hard. If they were
    allowed into the fit they would capture a leading component outright, so "the basis did
    not change" is a strong statement rather than a tolerance artifact.
    """
    rng = np.random.default_rng(seed)
    core = rng.normal(1.0, 0.35, size=(n_core, n_genes))
    # Give the core a real, learnable structure so the basis is not pure noise.
    core[:, :10] += rng.normal(0.0, 2.0, size=(n_core, 1))

    outliers = rng.normal(1.0, 0.35, size=(n_outlier, n_genes))
    outliers[:, 40:50] += 25.0  # a private block, far off the core's axes

    matrix = np.vstack([core, outliers]).astype(np.float32)
    obs = pd.DataFrame(
        {FIT_COLUMN: [True] * n_core + [False] * n_outlier},
        index=[f"cell_{i}" for i in range(len(matrix))],
    )
    adata = ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    adata.layers[LAYER] = sp.csr_matrix(matrix) if sparse else matrix.copy()
    return adata


def _run_pca(adata: ad.AnnData, tmp_path: Path, **overrides: object) -> object:
    """Run the PCA method the way the stage would."""
    config: dict[str, object] = {
        "input_layer": LAYER,
        "n_pcs": 10,
        "max_pcs": 10,
        "random_state": 0,
        "use_highly_variable": False,
    }
    config.update(overrides)
    return PCAMethod()._run(adata, config, context=_Context(tmp_path))


# ═══ 1. No leak: non-core cells cannot move the basis ═══════════════════════════════


def test_outlier_cells_do_not_move_the_basis(tmp_path: Path) -> None:
    """The fitted basis must be identical to one fitted with the outliers absent.

    This is the reference-immutability property. Anything weaker — "similar", "correlated" —
    would let a damaged cell tilt the manifold by a little, every run, invisibly.
    """
    full = _cohort()
    _run_pca(full, tmp_path)

    core_only = full[full.obs[FIT_COLUMN].to_numpy(bool)].copy()
    del core_only.obs[FIT_COLUMN]
    _run_pca(core_only, tmp_path)

    np.testing.assert_allclose(
        full.varm["PCs"], core_only.varm["PCs"], atol=1e-8, err_msg="the basis shifted"
    )


def test_the_variance_spectrum_describes_the_fit_population(tmp_path: Path) -> None:
    """Downstream reads uns['pca'] to pick n_pcs, so it must describe the core too."""
    full = _cohort()
    _run_pca(full, tmp_path)

    core_only = full[full.obs[FIT_COLUMN].to_numpy(bool)].copy()
    del core_only.obs[FIT_COLUMN]
    _run_pca(core_only, tmp_path)

    np.testing.assert_allclose(
        full.uns["pca"]["variance_ratio"],
        core_only.uns["pca"]["variance_ratio"],
        atol=1e-8,
    )


def test_the_control_outliers_really_would_have_captured_a_component(tmp_path: Path) -> None:
    """Without the mask the outliers dominate, so the test above is not vacuous.

    The private gene block is 25 units off the core's axes, so an unmasked fit must load it
    heavily on a leading component. If that were not true, "the basis did not change" would
    be trivially satisfiable and would prove nothing.
    """
    unmasked = _cohort()
    del unmasked.obs[FIT_COLUMN]
    _run_pca(unmasked, tmp_path)

    masked = _cohort()
    _run_pca(masked, tmp_path)

    block = slice(40, 50)
    leak = np.abs(unmasked.varm["PCs"][block, 0]).mean()
    clean = np.abs(masked.varm["PCs"][block, 0]).mean()
    assert leak > 10 * clean, (
        f"the outlier block loaded PC1 at {leak:.3f} unmasked vs {clean:.3f} masked — not a "
        f"large enough contrast for the immutability test above to mean anything"
    )


# ═══ 2. No silent loss: every cell still gets coordinates ═══════════════════════════


def test_every_cell_receives_an_embedding(tmp_path: Path) -> None:
    """Excluded from the fit is not excluded from the manifold.

    A borderline cell must be projected, not dropped, or QC would be deleting cells through
    the back door while reporting that it flagged them.
    """
    adata = _cohort()
    _run_pca(adata, tmp_path)

    assert adata.obsm["X_pca"].shape[0] == adata.n_obs
    assert np.isfinite(adata.obsm["X_pca"]).all()


def test_projected_coordinates_match_what_a_core_only_fit_computed(tmp_path: Path) -> None:
    """For the core cells, projection must reproduce scanpy's own embedding.

    The claim being checked is that ``(X - mean) @ PCs`` *is* the PCA transform and not an
    approximation of it. If this drifts, every projected cell is subtly misplaced.
    """
    full = _cohort()
    _run_pca(full, tmp_path)

    core_only = full[full.obs[FIT_COLUMN].to_numpy(bool)].copy()
    del core_only.obs[FIT_COLUMN]
    _run_pca(core_only, tmp_path)

    core_rows = full.obs[FIT_COLUMN].to_numpy(bool)
    np.testing.assert_allclose(full.obsm["X_pca"][core_rows], core_only.obsm["X_pca"], atol=1e-4)


def test_outliers_land_far_from_the_core_in_the_embedding(tmp_path: Path) -> None:
    """Projection must still place them where they belong — visibly apart.

    That is the point of projecting rather than dropping: a figure can show where the
    questionable cells sit relative to the manifold they did not define.
    """
    adata = _cohort()
    _run_pca(adata, tmp_path)

    embedding = adata.obsm["X_pca"]
    core = adata.obs[FIT_COLUMN].to_numpy(bool)
    spread = np.linalg.norm(embedding[core] - embedding[core].mean(axis=0), axis=1).mean()
    distance = np.linalg.norm(embedding[~core] - embedding[core].mean(axis=0), axis=1).mean()

    assert distance > spread, "projected outliers were not distinguishable from the core"


# ═══ 3. The projection primitive itself ════════════════════════════════════════════


@pytest.mark.parametrize("sparse", [False, True])
def test_projection_handles_sparse_and_dense_identically(sparse: bool) -> None:
    """The sparse-safe rearrangement must not change the answer.

    ``X @ PCs - meanᵀ @ PCs`` avoids densifying, which on the validation cohort is the
    difference between a matmul and tens of gigabytes. It has to be numerically identical.
    """
    rng = np.random.default_rng(0)
    dense = rng.normal(0.0, 1.0, size=(50, 12))
    loadings = rng.normal(0.0, 1.0, size=(12, 4))
    means = dense.mean(axis=0)

    matrix = sp.csr_matrix(dense) if sparse else dense
    got = project_onto_fitted_basis(matrix, loadings, means)
    expected = (dense - means) @ loadings

    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_sparse_input_survives_the_whole_stage(tmp_path: Path) -> None:
    """Real data is sparse, so the stage must never rely on a dense layer."""
    adata = _cohort(sparse=True)
    _run_pca(adata, tmp_path)

    assert adata.obsm["X_pca"].shape[0] == adata.n_obs
    assert np.isfinite(adata.obsm["X_pca"]).all()


# ═══ 4. Degradation: absent, empty, and small fit populations ══════════════════════


def test_a_dataset_without_graded_qc_behaves_as_before(tmp_path: Path) -> None:
    """Absent QC columns must not become a hidden dependency."""
    adata = _cohort()
    del adata.obs[FIT_COLUMN]
    result = _run_pca(adata, tmp_path)

    assert adata.obsm["X_pca"].shape == (adata.n_obs, 10)
    assert not any("QC-permitted" in note for note in result.notes)


def test_an_empty_fit_population_falls_back_rather_than_fitting_on_nothing(
    tmp_path: Path,
) -> None:
    """An all-False mask is a misconfiguration, not an instruction to fit on zero cells."""
    adata = _cohort()
    adata.obs[FIT_COLUMN] = False
    _run_pca(adata, tmp_path)

    assert adata.obsm["X_pca"].shape[0] == adata.n_obs


def test_component_count_is_capped_by_the_fit_population_not_the_cohort(
    tmp_path: Path,
) -> None:
    """A small core must cap n_comps, or the SVD fails on a dimension nobody chose.

    Requesting 10 components from 6 fitting cells is not a user error worth crashing on; the
    cap belongs where the fit population is known.
    """
    adata = _cohort(n_core=6, n_outlier=60)
    _run_pca(adata, tmp_path, n_pcs="auto", max_pcs=10)

    assert adata.obsm["X_pca"].shape[0] == adata.n_obs
    assert adata.obsm["X_pca"].shape[1] <= 5  # n_fit_cells - 1


# ═══ 5. Provenance: the decision has to be legible after the run ═══════════════════


def test_the_run_records_how_many_cells_fitted_and_how_many_were_projected(
    tmp_path: Path,
) -> None:
    """A scope decision nobody can see after the fact is the failure being designed out."""
    adata = _cohort(n_core=200, n_outlier=40)
    result = _run_pca(adata, tmp_path)

    scope_notes = [note for note in result.notes if "QC-permitted" in note]
    assert len(scope_notes) == 1
    assert "200" in scope_notes[0]
    assert "40" in scope_notes[0]
