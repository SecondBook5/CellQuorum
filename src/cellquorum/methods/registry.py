"""Registry mapping (stage_category, method name) -> AnalysisMethod class.

The registry is what turns a config string into an executable strategy:
``stages.ambient_correction.method: decontx`` resolves to ``DecontXMethod``.
Lookups fail loud so a typo cannot silently change pipeline behavior.
"""

from __future__ import annotations

from cellquorum.contracts import CellQuorumContractError
from cellquorum.methods.base import AnalysisMethod


class MethodRegistry:
    """
    Store analysis-method classes keyed by (stage_category, name).
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""

        # Nested mapping: category -> {name -> method class}.
        self._methods: dict[str, dict[str, type[AnalysisMethod]]] = {}

    def register(self, method_cls: type[AnalysisMethod]) -> None:
        """
        Register an AnalysisMethod subclass.

        Args:
            method_cls: The method class to register (uses its class attributes).

        Raises:
            CellQuorumContractError: If the class lacks required attributes or a
                duplicate (category, name) is registered.
        """

        # Validate required class attributes are set.
        category = getattr(method_cls, "stage_category", None)
        name = getattr(method_cls, "name", None)
        if not category or not name:
            raise CellQuorumContractError(
                f"Method class {method_cls!r} must set 'stage_category' and 'name'."
            )

        # Reject duplicate registrations.
        bucket = self._methods.setdefault(category, {})
        if name in bucket:
            raise CellQuorumContractError(
                f"Method '{name}' already registered for stage '{category}'."
            )

        # Store the class.
        bucket[name] = method_cls

    def get(self, stage_category: str, name: str) -> type[AnalysisMethod]:
        """
        Return a registered method class, or raise if unknown.

        Args:
            stage_category: Stage category the method belongs to.
            name: Method name.

        Returns:
            The registered method class.

        Raises:
            CellQuorumContractError: If no such method is registered.
        """

        # Look up the category then the name, failing loud at each level.
        bucket = self._methods.get(stage_category, {})
        method_cls = bucket.get(name)
        if method_cls is None:
            raise CellQuorumContractError(
                f"No method '{name}' registered for stage '{stage_category}'. "
                f"Available: {sorted(bucket)}."
            )
        return method_cls

    def has(self, stage_category: str, name: str) -> bool:
        """
        Return whether a method is registered for a stage category.

        Args:
            stage_category: Stage category the method belongs to.
            name: Method name.

        Returns:
            True if the method is registered, False otherwise.
        """

        # Check the nested mapping without raising.
        return name in self._methods.get(stage_category, {})

    def names(self, stage_category: str) -> list[str]:
        """Return registered method names for a stage category, in registration order."""

        # Return the names for the category (empty list if none).
        return list(self._methods.get(stage_category, {}).keys())


# Module-level singleton used by stages and method modules at import time.
METHOD_REGISTRY = MethodRegistry()


__all__ = ["METHOD_REGISTRY", "MethodRegistry"]
