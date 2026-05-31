"""Stage registry utilities for CellQuorum."""

from __future__ import annotations

# Import regular expressions for validating stable stage names.
import re

# Import Iterator for typed registry iteration.
from collections.abc import Iterator

# Import dataclass for immutable registry snapshots.
from dataclasses import dataclass

# Import shared CellQuorum exception base.
from cellquorum.core.exceptions import CellQuorumError

# Import the stage protocol.
from cellquorum.core.stage import PipelineStage


class StageRegistryError(CellQuorumError):
    """
    Report stage registry failures.

    The stage registry is responsible for storing and retrieving executable
    pipeline stages by stable names. Registry errors should fail early and
    clearly because ambiguous stage lookup would make execution planning,
    provenance, and report generation unreliable.
    """


@dataclass(frozen=True)
class StageRegistrySnapshot:
    """
    Store a lightweight snapshot of registered stages.

    Snapshots are useful for provenance, tests, CLI diagnostics, and future
    report generation. The snapshot intentionally stores only stage names rather
    than stage objects because stage objects may contain non-serializable runtime
    state.

    Args:
        stage_names: Ordered registered stage names.
        n_stages: Number of registered stages.
    """

    # Store registered stage names in registry order.
    stage_names: tuple[str, ...]

    # Store the number of registered stages.
    n_stages: int

    def to_dict(self) -> dict[str, object]:
        """
        Convert the registry snapshot to a JSON-friendly dictionary.

        Returns:
            Dictionary containing stage names and stage count.
        """

        # Return a JSON-friendly snapshot payload.
        return {
            "stage_names": list(self.stage_names),
            "n_stages": self.n_stages,
        }


class StageRegistry:
    """
    Register and retrieve CellQuorum pipeline stages.

    This registry is intentionally separate from the backend registry. Backends
    describe execution environments such as Python, R, Rscript, GPU, and RAPIDS.
    This registry stores actual pipeline stages such as QC, preprocessing,
    annotation, differential expression, communication analysis, and reporting.

    Stages are registered by their stable `name` attribute. Names must be
    lowercase, snake-case compatible identifiers so they can safely appear in
    configuration files, artifact paths, provenance records, and CLI output.
    """

    # Store the allowed stage-name pattern.
    _VALID_STAGE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(self) -> None:
        """
        Initialize an empty stage registry.
        """

        # Store stage objects by stable stage name.
        self._stages: dict[str, PipelineStage] = {}

    def register(self, stage: PipelineStage, *, overwrite: bool = False) -> None:
        """
        Register a pipeline stage.

        Args:
            stage: Stage object with a stable `name` and callable `run` method.
            overwrite: Whether to replace an existing stage with the same name.

        Raises:
            StageRegistryError: If the stage is invalid or a duplicate is
                registered without overwrite enabled.
        """

        # Validate the stage object and extract its normalized name.
        stage_name = self._validate_stage(stage)

        # Reject duplicate registrations unless overwrite is explicitly enabled.
        if stage_name in self._stages and not overwrite:
            raise StageRegistryError(
                f"Stage '{stage_name}' is already registered. "
                "Pass overwrite=True to replace it intentionally."
            )

        # Store the stage object by name.
        self._stages[stage_name] = stage

    def unregister(self, stage_name: str) -> PipelineStage:
        """
        Remove and return a registered stage.

        Args:
            stage_name: Stable stage name to remove.

        Returns:
            Removed pipeline stage.

        Raises:
            StageRegistryError: If the stage name is invalid or absent.
        """

        # Normalize and validate the requested stage name.
        normalized_name = self._normalize_stage_name(stage_name)

        # Reject unknown stage names with a clear message.
        if normalized_name not in self._stages:
            raise StageRegistryError(
                f"Cannot unregister unknown stage '{normalized_name}'. "
                f"Available stages: {self.names()}."
            )

        # Remove and return the registered stage.
        return self._stages.pop(normalized_name)

    def get(self, stage_name: str) -> PipelineStage | None:
        """
        Return a registered stage by name, or None if absent.

        Args:
            stage_name: Stable stage name to retrieve.

        Returns:
            Matching stage object, or None when absent.

        Raises:
            StageRegistryError: If the stage name is invalid.
        """

        # Normalize and validate the requested stage name.
        normalized_name = self._normalize_stage_name(stage_name)

        # Return the stage if present.
        return self._stages.get(normalized_name)

    def require(self, stage_name: str) -> PipelineStage:
        """
        Return a registered stage by name and fail if absent.

        Args:
            stage_name: Stable stage name to retrieve.

        Returns:
            Matching stage object.

        Raises:
            StageRegistryError: If the stage name is invalid or absent.
        """

        # Normalize and validate the requested stage name.
        normalized_name = self._normalize_stage_name(stage_name)

        # Retrieve the stage.
        stage = self._stages.get(normalized_name)

        # Reject unknown stages with a clear message.
        if stage is None:
            raise StageRegistryError(
                f"Required stage '{normalized_name}' is not registered. "
                f"Available stages: {self.names()}."
            )

        # Return the registered stage.
        return stage

    def has(self, stage_name: str) -> bool:
        """
        Return whether a stage name is registered.

        Args:
            stage_name: Stable stage name to check.

        Returns:
            True if the stage is registered, otherwise False.

        Raises:
            StageRegistryError: If the stage name is invalid.
        """

        # Normalize and validate the requested stage name.
        normalized_name = self._normalize_stage_name(stage_name)

        # Return whether the normalized stage name is registered.
        return normalized_name in self._stages

    def names(self) -> list[str]:
        """
        Return registered stage names in insertion order.

        Returns:
            Ordered list of registered stage names.
        """

        # Return registry keys as an ordered list.
        return list(self._stages.keys())

    def stages(self) -> list[PipelineStage]:
        """
        Return registered stage objects in insertion order.

        Returns:
            Ordered list of registered stage objects.
        """

        # Return registry values as an ordered list.
        return list(self._stages.values())

    def items(self) -> list[tuple[str, PipelineStage]]:
        """
        Return registered stage name/object pairs in insertion order.

        Returns:
            Ordered list of stage name and stage object pairs.
        """

        # Return registry items as an ordered list.
        return list(self._stages.items())

    def clear(self) -> None:
        """
        Remove all registered stages.
        """

        # Clear the stage mapping.
        self._stages.clear()

    def snapshot(self) -> StageRegistrySnapshot:
        """
        Build a lightweight snapshot of the registry.

        Returns:
            StageRegistrySnapshot containing registered stage names and count.
        """

        # Build the immutable registry snapshot.
        return StageRegistrySnapshot(
            stage_names=tuple(self.names()),
            n_stages=len(self),
        )

    def to_dict(self) -> dict[str, object]:
        """
        Convert the registry to a JSON-friendly dictionary.

        Returns:
            Dictionary containing registered stage names and stage count.
        """

        # Return the snapshot dictionary representation.
        return self.snapshot().to_dict()

    def __contains__(self, stage_name: object) -> bool:
        """
        Return whether a stage name is registered.

        Args:
            stage_name: Candidate stage name.

        Returns:
            True when stage_name is a valid registered string, otherwise False.
        """

        # Return False for non-string candidates.
        if not isinstance(stage_name, str):
            return False

        # Return False when validation fails.
        try:
            # Check whether the normalized name is registered.
            return self.has(stage_name)

        # Treat invalid names as absent for containment checks.
        except StageRegistryError:
            return False

    def __len__(self) -> int:
        """
        Return the number of registered stages.

        Returns:
            Number of registered stages.
        """

        # Return the stage count.
        return len(self._stages)

    def __iter__(self) -> Iterator[PipelineStage]:
        """
        Iterate over registered stage objects.

        Returns:
            Iterator over registered stage objects in insertion order.
        """

        # Return an iterator over stage objects.
        return iter(self._stages.values())

    @classmethod
    def _normalize_stage_name(cls, stage_name: str) -> str:
        """
        Normalize and validate a stage name.

        Stage names are kept strict because they appear in configuration,
        provenance, artifact names, and CLI output. The allowed format is
        lowercase snake_case beginning with a letter.

        Args:
            stage_name: Candidate stage name.

        Returns:
            Normalized stage name.

        Raises:
            StageRegistryError: If the stage name is not a valid string.
        """

        # Reject non-string stage names.
        if not isinstance(stage_name, str):
            raise StageRegistryError(
                "Stage name must be a string. " f"Received: {type(stage_name).__name__}."
            )

        # Strip harmless surrounding whitespace.
        normalized_name = stage_name.strip()

        # Reject empty names.
        if not normalized_name:
            raise StageRegistryError("Stage name cannot be empty.")

        # Reject names that do not match the stable stage-name pattern.
        if cls._VALID_STAGE_NAME_PATTERN.fullmatch(normalized_name) is None:
            raise StageRegistryError(
                "Stage name must be lowercase snake_case, begin with a letter, "
                "and contain only lowercase letters, numbers, and underscores. "
                f"Received: '{stage_name}'."
            )

        # Return the normalized stage name.
        return normalized_name

    @classmethod
    def _validate_stage(cls, stage: PipelineStage) -> str:
        """
        Validate a stage object and return its normalized name.

        Args:
            stage: Candidate stage object.

        Returns:
            Normalized stage name.

        Raises:
            StageRegistryError: If the stage object does not expose a valid
                `name` attribute and callable `run` method.
        """

        # Extract the stage name from the object.
        stage_name = getattr(stage, "name", None)

        # Normalize and validate the stage name.
        normalized_name = cls._normalize_stage_name(stage_name)

        # Extract the run attribute from the object.
        run_method = getattr(stage, "run", None)

        # Reject objects without a callable run method.
        if not callable(run_method):
            raise StageRegistryError(
                f"Stage '{normalized_name}' must define a callable run(context) method."
            )

        # Return the normalized stage name.
        return normalized_name


def build_stage_registry(stages: list[PipelineStage] | None = None) -> StageRegistry:
    """
    Build a stage registry from optional stage objects.

    Args:
        stages: Optional list of stage objects to register.

    Returns:
        StageRegistry containing the supplied stages.
    """

    # Create an empty stage registry.
    registry = StageRegistry()

    # Return the empty registry when no stages are provided.
    if stages is None:
        return registry

    # Register each supplied stage.
    for stage in stages:
        registry.register(stage)

    # Return the populated registry.
    return registry


__all__ = [
    "StageRegistry",
    "StageRegistryError",
    "StageRegistrySnapshot",
    "build_stage_registry",
]
