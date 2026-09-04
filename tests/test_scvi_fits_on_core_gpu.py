"""scVI and scANVI must train on core cells and encode everyone — verified on a real GPU.

Unlike Harmony, a trained VAE encoder is a function, so these methods can honour
``fit_scope=CORE`` exactly: train on the QC fit population, then push every cell through the
trained encoder. ``tests/test_integration_fit_population.py`` covers the *decision* of what to
train on, which is pure Python. This file covers the part that actually needs hardware — that
the split trains, encodes, and leaves the reference untouched.

Runs are kept tiny (few cells, few epochs) because the claim under test is structural, not
about embedding quality: does the excluded cell get a coordinate, and did it fail to move the
model that produced it.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.qc.eligibility import Analysis, EligibilityMasks, Permission

FIT_COLUMN = EligibilityMasks.column_name(Analysis.MANIFOLD, Permission.FIT)
BATCH = "patient_id"


def _gpu_or_skip() -> None:
    """Skip unless a CUDA device and scvi-tools are both actually present."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("scvi")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")


class _Registry:
    """Backend registry stub that reports GPU availability."""

    def available(self, name: str) -> bool:
        return name == "gpu"


class _Context:
    """Pipeline context stub carrying only what the VAE methods read."""

    backend_registry = _Registry()


def _cohort(
    n_core: int = 120,
    n_excluded: int = 40,
    n_genes: int = 50,
    *,
    seed: int = 0,
) -> ad.AnnData:
    """A two-batch cohort where both batches are represented in the core.

    Both batches must appear in the fit population or the coverage guard refuses the split,
    which is its job — but then this file would be testing the fallback instead of the
    encoder.
    """
    rng = np.random.default_rng(seed)
    counts = rng.negative_binomial(5, 0.3, size=(n_core + n_excluded, n_genes))

    half = (n_core + n_excluded) // 2
    obs = pd.DataFrame(
        {
            BATCH: ["p1"] * half + ["p2"] * (n_core + n_excluded - half),
            FIT_COLUMN: [True] * n_core + [False] * n_excluded,
            "_scanvi_labels": (["A"] * (half // 2) + ["B"] * (half - half // 2)) * 2,
        },
        index=[f"cell_{i}" for i in range(n_core + n_excluded)],
    )
    adata = ad.AnnData(
        X=counts.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


def _run_scvi(adata: ad.AnnData, **overrides: object):
    """Run the scVI integration method with a short training budget."""
    from cellquorum.stages.integration.scvi_methods import ScVIMethod

    config: dict[str, object] = {
        "batch_key": BATCH,
        "n_latent": 5,
        "output_rep": "X_scvi",
        "max_epochs": 2,
        "random_state": 0,
    }
    config.update(overrides)
    return ScVIMethod()._run(adata, config, context=_Context())


# ═══ scVI: the split trains and encodes on real hardware ═══════════════════════════


def test_scvi_trains_on_core_and_encodes_every_cell() -> None:
    """The excluded cells must come out with latent coordinates they did not shape."""
    _gpu_or_skip()
    adata = _cohort()

    result = _run_scvi(adata)

    latent = adata.obsm["X_scvi"]
    assert latent.shape == (adata.n_obs, 5)
    assert np.isfinite(latent).all()

    scope = [note for note in result.notes if "QC-permitted cells" in note]
    assert len(scope) == 1, result.notes
    assert "120 QC-permitted cells" in scope[0]
    assert "40 further cells encoded" in scope[0]


def test_scvi_leaves_the_core_cells_latent_unchanged_by_the_excluded_ones() -> None:
    """Reference immutability, measured through the encoder rather than argued for.

    Same seed and same training rows mean the same model, so the core cells' coordinates must
    be identical whether or not the excluded cells were in the file. Any drift here would mean
    the excluded cells reached the optimiser.
    """
    _gpu_or_skip()

    full = _cohort()
    _run_scvi(full)

    core_only = _cohort()
    core_only = core_only[core_only.obs[FIT_COLUMN].to_numpy(bool)].copy()
    del core_only.obs[FIT_COLUMN]
    _run_scvi(core_only)

    core_rows = full.obs[FIT_COLUMN].to_numpy(bool)
    np.testing.assert_allclose(
        full.obsm["X_scvi"][core_rows],
        core_only.obsm["X_scvi"],
        atol=1e-4,
        err_msg="the excluded cells changed the core cells' latent space",
    )


def test_scvi_without_graded_qc_still_trains_on_everything() -> None:
    """Absent QC columns must not become a hidden dependency of the GPU path."""
    _gpu_or_skip()
    adata = _cohort()
    del adata.obs[FIT_COLUMN]

    result = _run_scvi(adata)

    assert adata.obsm["X_scvi"].shape == (adata.n_obs, 5)
    assert not any("QC-permitted" in note for note in result.notes)


def test_scvi_falls_back_and_says_so_when_a_batch_is_missing_from_core() -> None:
    """The coverage guard must hold on the real path, not just in the pure unit test.

    scVI conditions the decoder on batch. A batch the model never saw cannot be encoded, so
    the guard trades the core-only split for a disclosed full-cohort fit.
    """
    _gpu_or_skip()
    adata = _cohort()
    # Make p2 entirely non-core, so the fit population covers only p1.
    adata.obs[FIT_COLUMN] = (adata.obs[BATCH] == "p1").to_numpy()

    result = _run_scvi(adata)

    assert adata.obsm["X_scvi"].shape == (adata.n_obs, 5)
    disclosures = [note for note in result.notes if "trained on all cells" in note]
    assert len(disclosures) == 1, result.notes
    assert "p2" in disclosures[0]


# ═══ scANVI: same contract, one more conditioning variable ═════════════════════════


def test_scanvi_trains_on_core_and_encodes_every_cell() -> None:
    """scANVI conditions on labels too, so the split has to survive both covariates."""
    _gpu_or_skip()
    from cellquorum.stages.integration.scanvi_methods import ScANVIMethod

    adata = _cohort()
    result = ScANVIMethod()._run(
        adata,
        {
            "batch_key": BATCH,
            "label_key": "_scanvi_labels",
            "n_latent": 5,
            "output_rep": "X_scanvi",
            "max_epochs": 2,
            "random_state": 0,
        },
        context=_Context(),
    )

    latent = adata.obsm["X_scanvi"]
    assert latent.shape == (adata.n_obs, 5)
    assert np.isfinite(latent).all()
    assert any("QC-permitted cells" in note for note in result.notes), result.notes
