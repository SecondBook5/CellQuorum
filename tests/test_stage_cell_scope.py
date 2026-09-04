"""Every stage that fits something across cells must declare whose cells it may fit on.

This test exists because of a specific, measured failure. QC used to write one boolean
verdict, ``cellquorum_qc_keep``, and across the whole codebase three places read it — two of
them figure code. Not preprocessing, not feature selection, not PCA, not integration, not
clustering, not annotation, not DE. So a careful QC verdict controlled nothing, and the
production default of ``flag_no_drop`` meant QC was reporting without control.

That was an **engine-contract failure**, not a QC-design failure: nothing stopped a
developer writing ``model.fit(adata)`` on every cell. Replacing one boolean with better
columns would recreate it exactly, so the columns are paired with a registration-level
declaration and this test.

The point is not to check today's five stages. It is that a *new* stage which fits a model
cannot be added without someone deciding, in writing, whose cells it may learn from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cellquorum.core.stage_catalog import CellScope, CellScopePolicy
from cellquorum.core.stages import all_stage_specs

#: Stages that estimate a quantity across cells, so a questionable cell could shape the
#: biological reference every later stage is measured against.
#:
#: "Fits a model" is broader than it sounds and that is the trap. A normalization target, a
#: gene prevalence filter, an HVG dispersion, a scaling mean, a PCA loading, a neighbour
#: graph and a cluster centroid are all cohort-derived quantities used to transform
#: biological data. PFlog1pPF is the clearest case: ``scclr_target="auto"`` estimates the NB
#: overdispersion alpha across cells, which does not look like a fitted model and is one.
#: (Not ``target_sum`` in ``normalization.py`` — those recipes are all per-cell.)
#:
#: Adding a stage here without declaring its scope is a failing test, by design. If a new
#: stage fits something, add it here *and* declare the scope.
STAGES_THAT_FIT_ACROSS_CELLS: frozenset[str] = frozenset(
    {
        "preprocessing",
        "feature_selection",
        "dimensionality",
        "integration",
        "clustering",
    }
)


def _spec(name: str):
    """The registered spec for one stage name."""
    specs = {spec.name: spec for spec in all_stage_specs()}
    assert name in specs, f"{name!r} is in STAGES_THAT_FIT_ACROSS_CELLS but not registered"
    return specs[name]


@pytest.mark.parametrize("stage_name", sorted(STAGES_THAT_FIT_ACROSS_CELLS))
def test_fitting_stage_declares_a_cell_scope(stage_name: str) -> None:
    """A stage that fits across cells must not leave its fit population implicit.

    Args:
        stage_name: Registered stage name.
    """
    spec = _spec(stage_name)
    assert spec.cell_scope is not None, (
        f"stage {stage_name!r} fits a quantity across cells but declares no cell_scope. "
        f"Pass cell_scope=CellScopePolicy(fit_scope=...) to @register_stage. Leaving it "
        f"implicit is how a QC verdict ends up controlling nothing."
    )


@pytest.mark.parametrize("stage_name", sorted(STAGES_THAT_FIT_ACROSS_CELLS))
def test_fitting_on_every_cell_requires_a_stated_reason(stage_name: str) -> None:
    """``fit_scope=ALL`` is allowed, but only with a justification on the record.

    Some transforms are genuinely per-cell and independent. The policy permits that and
    refuses to let it be reached silently.

    Args:
        stage_name: Registered stage name.
    """
    scope = _spec(stage_name).cell_scope
    assert scope is not None
    if scope.fit_scope is CellScope.ALL:
        assert scope.reason, (
            f"stage {stage_name!r} fits on every cell without a reason; questionable cells "
            f"would define the biological reference"
        )


def test_the_fitted_chain_is_protected_end_to_end() -> None:
    """The whole fitted chain must fit on core, not just the parts that look like models.

    Damaged cells carry stress genes, mitochondrial genes and immediate-early genes. If they
    take part in HVG selection they change the biological manifold *even if* they are later
    excluded from PCA. Protecting only the obvious steps leaves the leak upstream.
    """
    scopes = {
        spec.name: spec.cell_scope
        for spec in all_stage_specs()
        if spec.name in STAGES_THAT_FIT_ACROSS_CELLS
    }
    leaking = sorted(
        name
        for name, scope in scopes.items()
        if scope is None or scope.fit_scope is not CellScope.CORE
    )
    assert not leaking, f"these stages may fit on non-core cells: {leaking}"


def test_policy_rejects_fitting_on_all_cells_without_a_reason() -> None:
    """The guard lives in the type, so it cannot be bypassed by forgetting this test."""
    with pytest.raises(ValueError, match="requires a reason"):
        CellScopePolicy(fit_scope=CellScope.ALL)


def test_policy_allows_fitting_on_all_cells_with_a_reason() -> None:
    """A genuinely per-cell transform is legitimate when it says so."""
    policy = CellScopePolicy(
        fit_scope=CellScope.ALL, reason="per-cell independent transform, no cohort statistic"
    )
    assert policy.fit_scope is CellScope.ALL


def test_declared_scopes_survive_registration() -> None:
    """A declaration that the registry drops is worse than none — it reads as compliant.

    ``register_stage`` accepted ``cell_scope`` and did not store it on the spec, so five
    stages declared core-only fitting and the registry reported zero. Nothing failed; the
    contract was simply absent. This asserts the round trip.
    """
    declared = [spec for spec in all_stage_specs() if spec.cell_scope is not None]
    assert len(declared) >= len(STAGES_THAT_FIT_ACROSS_CELLS), (
        f"only {len(declared)} specs carry a cell_scope but "
        f"{len(STAGES_THAT_FIT_ACROSS_CELLS)} stages declare one — register_stage is "
        f"dropping the declaration"
    )


#: Where each fitting stage's methods live. Non-recursive on purpose: ``preprocessing`` has
#: ``feature_selection`` and ``dimensionality`` beneath it, and a recursive search would let
#: the parent pass on its children's work.
_STAGE_SOURCE_DIRS: dict[str, str] = {
    "preprocessing": "stages/preprocessing",
    "feature_selection": "stages/preprocessing/feature_selection",
    "dimensionality": "stages/preprocessing/dimensionality",
    "integration": "stages/integration",
    "clustering": "stages/clustering",
}

#: Reading the fit population, however it is spelled. ``fitting_cells`` is the shared reader;
#: ``resolve_training_set`` is integration's wrapper around it for the VAE methods.
_FIT_POPULATION_READERS = ("fitting_cells", "resolve_training_set")


@pytest.mark.parametrize("stage_name", sorted(STAGES_THAT_FIT_ACROSS_CELLS))
def test_a_core_declaring_stage_actually_reads_the_fit_population(stage_name: str) -> None:
    """A declaration nothing acts on is the failure this contract exists to prevent.

    ``test_declared_scopes_survive_registration`` proves the declaration reaches the registry.
    This proves some code in the stage *consults* it. The behavioural proofs live in
    ``test_qc_fit_mask_is_honoured``, ``test_pca_fits_on_core_and_projects``,
    ``test_clustering_fits_on_core_and_transfers``, ``test_normalization_fits_target_on_core``
    and ``test_integration_fit_population``; this is the cheap structural guard that fails when
    a *new* method or stage is added and simply never wired.

    Args:
        stage_name: Registered stage name.
    """
    scope = _spec(stage_name).cell_scope
    assert scope is not None
    if scope.fit_scope is not CellScope.CORE:
        pytest.skip(f"{stage_name} does not declare CORE")

    import cellquorum

    directory = Path(cellquorum.__file__).parent / _STAGE_SOURCE_DIRS[stage_name]
    sources = sorted(path for path in directory.glob("*.py") if path.name != "__init__.py")
    assert sources, f"no source files found for {stage_name} at {directory}"

    wired = [
        path.name
        for path in sources
        if any(reader in path.read_text() for reader in _FIT_POPULATION_READERS)
    ]
    assert wired, (
        f"stage {stage_name!r} declares fit_scope=CORE but nothing in {directory} reads the "
        f"fit population (looked for {' or '.join(_FIT_POPULATION_READERS)}). A declared scope "
        f"that no code consults reads as compliant while controlling nothing."
    )
