"""The PFlog1pPF proportional-fitting target must be estimated from core cells only.

This is the cohort-derived quantity that is hardest to see, and it was nearly missed twice.
The first reading blamed ``target_sum`` in ``normalization.py``, which is wrong: every
pure-matrix recipe there is per-cell (``x_ij / sum_j(x_ij)``, or a constant ``target_sum``),
so none of them fits anything.

The actual quantity lives one layer down, in the scclr backend that implements the real
Booeshaghi/Pachter transform. ``scclr_target`` defaults to ``"auto"``, which estimates the
negative-binomial overdispersion alpha **across cells**; ``mean``/``median`` take a cohort
depth. Either way one damaged cell moves the target, and the target scales every cell's
normalized values — before HVG, before PCA, before anything downstream. It does not look like
a fitted model, and it is one.

The fix exploits a property of the transform rather than changing it: a *fixed numeric* K makes
PFlog1pPF purely per-cell. So K is fitted on the permitted cells, then handed to the full pass
as a literal number, and the second pass leaks nothing by construction.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.preprocessing.config import NormalizationConfig
from cellquorum.stages.preprocessing.normalization import run_scclr_pflog
from cellquorum.stages.qc.eligibility import Analysis, EligibilityMasks, Permission

FIT_COLUMN = EligibilityMasks.column_name(Analysis.MANIFOLD, Permission.FIT)


def _scclr_backend_or_skip():
    """Return an available scclr backend, or skip when its isolated env is absent."""
    from cellquorum.backends.scclr_backend import build_scclr_backend

    backend = build_scclr_backend()
    if not backend.status().available:
        pytest.skip("scclr environment unavailable (isolated micromamba env not built)")
    return backend


def _cohort(
    n_core: int = 200,
    n_deep: int = 60,
    n_genes: int = 120,
    *,
    seed: int = 0,
) -> ad.AnnData:
    """Core cells at one depth, non-core cells far deeper.

    Depth is the lever because the target is a depth/overdispersion statistic. Making the
    excluded cells an order of magnitude deeper means an unmasked fit must land on a
    noticeably different K, which is what gives the control test its teeth.
    """
    rng = np.random.default_rng(seed)
    # NB-distributed (overdispersed) so scclr's target="auto" alpha estimate is valid.
    core = rng.negative_binomial(5, 0.35, size=(n_core, n_genes))
    deep = rng.negative_binomial(60, 0.35, size=(n_deep, n_genes))

    matrix = np.vstack([core, deep]).astype(np.float32)
    obs = pd.DataFrame(
        {FIT_COLUMN: [True] * n_core + [False] * n_deep},
        index=[f"cell_{i}" for i in range(len(matrix))],
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    adata.layers["counts"] = matrix.copy()
    return adata


def _normalize(adata: ad.AnnData, backend, target: str = "auto", tmp_path=None):
    """Run the scclr PFlog1pPF path the way the preprocessing stage would."""
    config = NormalizationConfig(
        recipe="cellquorum_pf_log1p_pf_v1",
        input_layer="counts",
        scclr_target=target,
        overwrite=True,
    )
    return run_scclr_pflog(
        matrix=adata.layers["counts"],
        adata=adata,
        config=config,
        backend=backend,
        scratch_dir=tmp_path,
    )


# ═══ The target is fitted on the permitted cells ═══════════════════════════════════


def test_the_target_becomes_a_fixed_k_fitted_on_core_cells(tmp_path) -> None:
    """With masks present, the full pass runs on a literal K rather than re-estimating.

    A literal K is what makes the second pass per-cell, so this is the mechanism, not a
    cosmetic difference in how the target is spelled.
    """
    backend = _scclr_backend_or_skip()
    adata = _cohort()

    _, diagnostics, warnings = _normalize(adata, backend, tmp_path=tmp_path)

    assert diagnostics["scclr_target"] == "auto", "the requested target must stay on the record"
    effective = str(diagnostics["scclr_effective_target"])
    assert effective != "auto"
    float(effective)  # must parse as a literal K

    assert any("QC-permitted cells" in note for note in warnings)


def test_the_fitted_target_matches_one_fitted_without_the_deep_cells(tmp_path) -> None:
    """Reference immutability: the excluded cells must not move K at all."""
    backend = _scclr_backend_or_skip()

    full = _cohort()
    _, full_diagnostics, _ = _normalize(full, backend, tmp_path=tmp_path)

    core_only = _cohort()
    core_only = core_only[core_only.obs[FIT_COLUMN].to_numpy(bool)].copy()
    del core_only.obs[FIT_COLUMN]
    _, core_diagnostics, _ = _normalize(core_only, backend, tmp_path=tmp_path)

    assert float(full_diagnostics["scclr_effective_target"]) == pytest.approx(
        float(core_diagnostics["scclr_k"]), rel=1e-9
    )


def test_the_control_deep_cells_really_do_move_the_target(tmp_path) -> None:
    """Without the mask K shifts, so the test above is not vacuous."""
    backend = _scclr_backend_or_skip()

    masked = _cohort()
    _, masked_diagnostics, _ = _normalize(masked, backend, tmp_path=tmp_path)

    unmasked = _cohort()
    del unmasked.obs[FIT_COLUMN]
    _, unmasked_diagnostics, _ = _normalize(unmasked, backend, tmp_path=tmp_path)

    fitted = float(masked_diagnostics["scclr_effective_target"])
    leaked = float(unmasked_diagnostics["scclr_k"])
    assert fitted != pytest.approx(leaked, rel=1e-6), (
        f"the deep cells did not move the target (fitted K={fitted:.6g}, unmasked "
        f"K={leaked:.6g}), so the masked assertion above proves nothing — strengthen the "
        f"fixture"
    )


# ═══ Every cell is still normalized ════════════════════════════════════════════════


def test_all_cells_are_normalized_not_just_the_fitted_ones(tmp_path) -> None:
    """Fitting on a subset must not drop the rest — the target applies to everyone."""
    backend = _scclr_backend_or_skip()
    adata = _cohort()

    normalized, diagnostics, _ = _normalize(adata, backend, tmp_path=tmp_path)

    assert normalized.shape[0] == adata.n_obs
    assert len(diagnostics["scclr_row_center"]) == adata.n_obs


# ═══ Degradation ══════════════════════════════════════════════════════════════════


def test_a_fixed_numeric_target_skips_the_fit_pass(tmp_path) -> None:
    """A literal K is already cohort-independent, so there is nothing to fit.

    Skipping matters: the fit pass is a second trip through the isolated environment over the
    whole counts matrix.
    """
    backend = _scclr_backend_or_skip()
    adata = _cohort()

    _, diagnostics, warnings = _normalize(adata, backend, target="1000.0", tmp_path=tmp_path)

    assert str(diagnostics["scclr_effective_target"]) == "1000.0"
    assert not any("QC-permitted" in note for note in warnings)


def test_a_dataset_without_graded_qc_behaves_as_before(tmp_path) -> None:
    """Absent QC columns must not become a hidden dependency."""
    backend = _scclr_backend_or_skip()
    adata = _cohort()
    del adata.obs[FIT_COLUMN]

    _, diagnostics, warnings = _normalize(adata, backend, tmp_path=tmp_path)

    assert diagnostics["scclr_effective_target"] == "auto"
    assert not any("QC-permitted" in note for note in warnings)


def test_an_empty_fit_population_falls_back_rather_than_fitting_on_nothing(tmp_path) -> None:
    """An all-False mask is a misconfiguration, not an instruction to fit on zero cells."""
    backend = _scclr_backend_or_skip()
    adata = _cohort()
    adata.obs[FIT_COLUMN] = False

    normalized, diagnostics, _ = _normalize(adata, backend, tmp_path=tmp_path)

    assert diagnostics["scclr_effective_target"] == "auto"
    assert normalized.shape[0] == adata.n_obs
