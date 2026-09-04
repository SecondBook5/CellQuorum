"""Central catalog of pipeline stages — the single source of truth for stage
identity, ordering, and config wiring.

Every implemented stage declares itself once via the :func:`register_stage`
decorator co-located with its class. The executor builds its runnable registry
from the catalog's implemented specs; the planner builds its ordered plan from
the full catalog sorted by ``order``. This replaces the former hand-maintained
import block + registry dict in ``core/executor.py`` and the ordered tuple list
in ``core/planner.py``.

Adding a stage: put ``@register_stage(...)`` on the stage class (in its own
module) and add the matching ``StageSelectionConfig`` flag and
``CellQuorumConfig`` sub-block field. ``tests/test_stage_catalog.py`` fails
loudly if either config field is missing or orphaned.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from cellquorum.core.stage import PipelineStage


class StageCatalogError(RuntimeError):
    """Raised on duplicate stage name or duplicate stage order registration."""


@dataclass(frozen=True)
class CellScope(StrEnum):
    """Whose cells a stage may use for a given permission.

    Exists because of a specific failure: QC wrote one boolean verdict and three places in
    the whole codebase read it, two of them figure code. Nothing in the engine prevented a
    stage from calling ``model.fit(adata)`` on every cell, so a careful verdict controlled
    nothing. Declaring scope at registration makes that choice visible and testable instead
    of implicit.
    """

    #: Only cells QC deemed eligible to define the biological reference.
    CORE = "core"

    #: Every cell. Legitimate for a per-cell independent transform, but it must be stated
    #: with a reason rather than reached by default.
    ALL = "all"

    #: The stage fits nothing across cells.
    NONE = "none"


@dataclass(frozen=True)
class CellScopePolicy:
    """A stage's declaration of whose cells it may fit, transform, and infer from.

    ## What a declaration obliges a stage to do

    The scope is declared per *stage*, but a stage dispatches to several *methods*, and their
    algorithms differ in what they can promise. So ``fit_scope=CORE`` means:

    1. Estimate the cohort quantity on :func:`cellquorum.stages.qc.eligibility.fitting_cells`.
    2. Apply it to every cell, so nobody is silently dropped.
    3. If the method has no out-of-sample transform and (2) is impossible, fit on everything
       and **say so in the stage's notes**.

    Point 3 is not a loophole; it is the only honest option for a method whose output cannot
    be applied to held-out cells. Harmony is the live example: it returns corrected
    coordinates rather than a reusable correction, so it discloses that it fitted on all
    cells, while scVI and scANVI train on core and encode everyone through the trained
    encoder. The same split appears inside the PCA stage, where the standard path fits on core
    and projects and the scclr implicit-centered path discloses that it cannot.

    An undisclosed inability is the failure this whole contract exists to prevent: a
    declaration that reads as compliant while nothing enforces it is exactly the QC verdict
    that three places read, one level up.

    Args:
        fit_scope: Cells allowed to determine parameters or cohort statistics. This covers
            more than models: normalization targets, gene prevalence filters, HVG
            dispersions, scaling means, PCA loadings, neighbour graphs, cluster centroids.
        transform_scope: Cells allowed to receive the stage's output.
        inference_scope: Cells allowed to contribute to a scientific conclusion.
        reason: Required when ``fit_scope`` is ``ALL`` — the justification for fitting on
            questionable cells, so the choice is auditable.

    Raises:
        ValueError: If ``fit_scope`` is ``ALL`` without a reason.
    """

    fit_scope: CellScope
    transform_scope: CellScope = CellScope.ALL
    inference_scope: CellScope = CellScope.CORE
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.fit_scope is CellScope.ALL and not self.reason:
            raise ValueError(
                "fit_scope=ALL requires a reason. Fitting on every cell lets questionable "
                "cells define the biological reference, so it must be justified explicitly "
                "rather than chosen by default."
            )


@dataclass(frozen=True)
class StageSpec:
    """Immutable description of one pipeline stage.

    Attributes:
        name: Stable stage identifier (planner/executor key and the stage
            instance's ``.name``).
        order: Canonical pipeline position; the planner sorts by this. Use
            multiples of 10 so stages can be inserted between neighbours.
        config_flag: ``StageSelectionConfig`` field that toggles the stage.
            Equals ``name`` for every stage except ``ccc_network`` (flag
            ``network_analysis``).
        config_field: ``CellQuorumConfig`` sub-block field name; ``None`` for
            planned-only stages with no implementation yet.
        category: Method-registry category set onto the class as
            ``stage_category``; ``None`` for stages that do not declare one.
        factory: Zero-argument callable returning a ``PipelineStage`` (the
            decorated class); ``None`` for planned-only stages.
        cell_scope: Declaration of whose cells the stage may fit, transform, and infer
            from. ``None`` means undeclared, which
            ``tests/test_stage_cell_scope.py`` rejects for any stage that fits a model
            across cells.
    """

    name: str
    order: int
    config_flag: str
    config_field: str | None
    category: str | None
    factory: Callable[[], PipelineStage] | None
    cell_scope: CellScopePolicy | None = None

    @property
    def is_implemented(self) -> bool:
        """True when the stage has a runnable implementation (a factory)."""
        return self.factory is not None


class StageCatalog:
    """Ordered collection of :class:`StageSpec`, keyed by unique name."""

    def __init__(self) -> None:
        self._specs: dict[str, StageSpec] = {}
        self._orders: dict[int, str] = {}

    def register(self, spec: StageSpec) -> None:
        """Add a spec. Raises on duplicate name or duplicate order."""
        if spec.name in self._specs:
            raise StageCatalogError(f"duplicate stage name: {spec.name!r}")
        if spec.order in self._orders:
            raise StageCatalogError(
                f"duplicate stage order {spec.order} for {spec.name!r}; "
                f"already used by {self._orders[spec.order]!r}"
            )
        self._specs[spec.name] = spec
        self._orders[spec.order] = spec.name

    def specs(self) -> tuple[StageSpec, ...]:
        """All specs sorted by ``order``."""
        return tuple(sorted(self._specs.values(), key=lambda s: s.order))

    def implemented(self) -> tuple[StageSpec, ...]:
        """Only specs with a factory, sorted by ``order``."""
        return tuple(s for s in self.specs() if s.is_implemented)

    def get(self, name: str) -> StageSpec:
        """Return the spec for ``name`` (raises ``KeyError`` if absent)."""
        return self._specs[name]

    def __len__(self) -> int:
        return len(self._specs)


# Process-wide default catalog, populated at import time by the
# @register_stage decorators as stage modules are imported (see core/stages.py).
_DEFAULT_CATALOG = StageCatalog()


def register_stage(
    *,
    name: str,
    order: int,
    config_flag: str,
    config_field: str,
    category: str | None = None,
    cell_scope: CellScopePolicy | None = None,
    catalog: StageCatalog | None = None,
) -> Callable[[type], type]:
    """Class decorator: register an implemented stage into the catalog.

    Sets ``cls.name`` (and ``cls.stage_category`` when ``category`` is given)
    so the decorator is the single source of truth for the stage's identity.
    """
    target = catalog if catalog is not None else _DEFAULT_CATALOG

    def _decorate(cls: type) -> type:
        cls.name = name
        if category is not None:
            cls.stage_category = category
        target.register(
            StageSpec(
                name=name,
                order=order,
                config_flag=config_flag,
                config_field=config_field,
                category=category,
                factory=cls,
                cell_scope=cell_scope,
            )
        )
        return cls

    return _decorate


def register_planned_stage(
    *,
    name: str,
    order: int,
    config_flag: str,
    catalog: StageCatalog | None = None,
) -> None:
    """Register a planned-but-unimplemented stage (no class, no factory).

    Planned stages appear in the planner's ordering and carry a
    ``StageSelectionConfig`` flag, but have no ``CellQuorumConfig`` sub-block
    and never run (the executor only instantiates implemented specs).
    """
    target = catalog if catalog is not None else _DEFAULT_CATALOG
    target.register(
        StageSpec(
            name=name,
            order=order,
            config_flag=config_flag,
            config_field=None,
            category=None,
            factory=None,
        )
    )


def iter_stage_specs(catalog: StageCatalog | None = None) -> tuple[StageSpec, ...]:
    """Return all specs from ``catalog`` (default: process-wide) sorted by order.

    Reads whatever has been registered so far. Callers that need the fully
    populated catalog import :mod:`cellquorum.core.stages` first (it imports
    every stage module, firing all decorators).
    """
    target = catalog if catalog is not None else _DEFAULT_CATALOG
    return target.specs()
