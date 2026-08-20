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

from cellquorum.core.stage import PipelineStage


class StageCatalogError(RuntimeError):
    """Raised on duplicate stage name or duplicate stage order registration."""


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
    """

    name: str
    order: int
    config_flag: str
    config_field: str | None
    category: str | None
    factory: Callable[[], PipelineStage] | None

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
