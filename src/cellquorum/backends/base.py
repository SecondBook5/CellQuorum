"""Base backend contracts for CellQuorum execution backends."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

BackendKind = Literal["python", "r", "rscript", "gpu", "rapids", "external"]


@dataclass(frozen=True)
class BackendRequirement:
    """
    Describe one software requirement for a backend.

    CellQuorum backends may depend on Python packages, R packages, command-line
    tools, GPU libraries, or external executables. This model keeps those
    requirements explicit so the planner can report which methods are available
    before the user starts a run.

    Args:
        name: Human-readable requirement name.
        requirement_type: Type of requirement, such as python_package or executable.
        required: Whether this requirement is mandatory for the backend.
        install_hint: Optional user-facing instruction for installing the requirement.
    """

    # Store the requirement name.
    name: str

    # Store the requirement type.
    requirement_type: Literal["python_package", "r_package", "executable", "cuda", "other"]

    # Store whether this requirement is mandatory.
    required: bool = True

    # Store an optional install hint for user-facing error messages.
    install_hint: str | None = None


@dataclass(frozen=True)
class BackendStatus:
    """
    Store availability information for one backend.

    Backend status objects are used by the planner, reports, and stage gates.
    They make it possible to say exactly why a backend can or cannot run instead
    of failing later with unclear import, executable, or CUDA errors.

    Args:
        name: Stable backend name.
        kind: Backend family.
        available: Whether the backend is currently available.
        requirements: Requirements checked for this backend.
        missing: Missing mandatory requirements.
        warnings: Non-fatal backend warnings.
        details: Additional JSON-serializable backend metadata.
    """

    # Store the stable backend name.
    name: str

    # Store the backend family.
    kind: BackendKind

    # Store whether the backend is available.
    available: bool

    # Store all requirements considered for the backend.
    requirements: list[BackendRequirement] = field(default_factory=list)

    # Store missing mandatory requirements.
    missing: list[str] = field(default_factory=list)

    # Store non-fatal backend warnings.
    warnings: list[str] = field(default_factory=list)

    # Store additional backend metadata.
    details: dict[str, Any] = field(default_factory=dict)

    def raise_if_unavailable(self) -> None:
        """
        Raise a clear error if this backend is unavailable.

        Stages should call this helper before attempting backend-specific work.
        That keeps failures user-facing and actionable instead of exposing raw
        import errors or missing executable traces.

        Raises:
            BackendUnavailableError: If the backend is unavailable.
        """

        # Return immediately when the backend is available.
        if self.available:
            return

        # Build a readable missing-requirements message.
        missing_message = ", ".join(self.missing) if self.missing else "unknown requirement"

        # Raise a backend-specific availability error.
        raise BackendUnavailableError(
            backend_name=self.name,
            message=(
                f"Backend '{self.name}' is unavailable. "
                f"Missing required component(s): {missing_message}."
            ),
        )


class BackendUnavailableError(RuntimeError):
    """
    Report that a requested backend cannot be used.

    This exception is intentionally specific so the planner and CLI can catch it
    and explain what needs to be installed or enabled.
    """

    def __init__(self, backend_name: str, message: str) -> None:
        """
        Initialize a backend availability error.

        Args:
            backend_name: Name of the unavailable backend.
            message: User-facing error message.
        """

        # Store the unavailable backend name for programmatic handling.
        self.backend_name = backend_name

        # Initialize the RuntimeError base class with the user-facing message.
        super().__init__(message)


class Backend(Protocol):
    """
    Define the interface every CellQuorum backend must implement.

    Backends represent execution families such as Python, R, Rscript, GPU,
    RAPIDS, or external tools. They should expose availability checks and
    structured status metadata without forcing stages to know low-level import
    or executable details.
    """

    # Store the stable backend name.
    name: str

    # Store the backend family.
    kind: BackendKind

    def requirements(self) -> list[BackendRequirement]:
        """
        Return requirements needed by this backend.

        Returns:
            List of backend requirements.
        """
        ...

    def status(self) -> BackendStatus:
        """
        Return availability status for this backend.

        Returns:
            BackendStatus describing whether the backend can run.
        """
        ...


@dataclass
class BaseBackend:
    """
    Provide common backend availability behavior.

    Concrete backend classes can inherit from BaseBackend and override
    `requirements()` when they need Python packages, executables, R packages, or
    CUDA-specific checks. The default status implementation handles Python
    packages and command-line executables directly. Specialized checks for R,
    CUDA, and RAPIDS can be implemented in subclasses.

    Args:
        name: Stable backend name.
        kind: Backend family.
        requirement_list: Requirements checked by this backend.
    """

    # Store the stable backend name.
    name: str

    # Store the backend family.
    kind: BackendKind

    # Store the requirements checked by this backend.
    requirement_list: list[BackendRequirement] = field(default_factory=list)

    def requirements(self) -> list[BackendRequirement]:
        """
        Return requirements needed by this backend.

        Returns:
            List of backend requirements.
        """

        # Return a shallow copy so callers cannot mutate the backend state.
        return list(self.requirement_list)

    def status(self) -> BackendStatus:
        """
        Check whether this backend is currently available.

        The base implementation supports Python package and executable checks.
        Requirement types that need specialized handling are reported as warnings
        unless they are implemented by a subclass.

        Returns:
            BackendStatus describing availability, missing requirements, and warnings.
        """

        # Initialize the missing requirement list.
        missing: list[str] = []

        # Initialize the warning list.
        warnings: list[str] = []

        # Iterate through each declared backend requirement.
        for requirement in self.requirements():
            # Check Python package availability through importlib.
            if requirement.requirement_type == "python_package":
                # Add missing mandatory Python packages to the missing list.
                if not self._python_package_available(requirement.name) and requirement.required:
                    missing.append(requirement.name)

            # Check executable availability through PATH lookup.
            elif requirement.requirement_type == "executable":
                # Add missing mandatory executables to the missing list.
                if not self._executable_available(requirement.name) and requirement.required:
                    missing.append(requirement.name)

            # Defer R package checks to R-specific backends.
            elif requirement.requirement_type == "r_package":
                # Record that the base backend does not perform R package checks.
                warnings.append(
                    f"Requirement '{requirement.name}' is an R package and must be checked "
                    "by an R-specific backend."
                )

            # Defer CUDA checks to GPU-specific backends.
            elif requirement.requirement_type == "cuda":
                # Record that the base backend does not perform CUDA checks.
                warnings.append(
                    f"Requirement '{requirement.name}' is CUDA-related and must be checked "
                    "by a GPU-specific backend."
                )

            # Record unsupported requirement types as warnings.
            else:
                # Add a warning for unrecognized requirement categories.
                warnings.append(
                    f"Requirement '{requirement.name}' has unsupported type "
                    f"'{requirement.requirement_type}' in BaseBackend."
                )

        # Determine availability from missing mandatory requirements.
        available = len(missing) == 0

        # Return structured backend status.
        return BackendStatus(
            name=self.name,
            kind=self.kind,
            available=available,
            requirements=self.requirements(),
            missing=missing,
            warnings=warnings,
            details={},
        )

    @staticmethod
    def _python_package_available(package_name: str) -> bool:
        """
        Check whether a Python package can be imported.

        Args:
            package_name: Python package or module name.

        Returns:
            True if the package can be found, otherwise False.
        """

        # Return whether importlib can find the requested package.
        return importlib.util.find_spec(package_name) is not None

    @staticmethod
    def _executable_available(executable_name: str) -> bool:
        """
        Check whether an executable is available on PATH.

        Args:
            executable_name: Command-line executable name.

        Returns:
            True if the executable can be found, otherwise False.
        """

        # Return whether shutil can locate the executable on PATH.
        return shutil.which(executable_name) is not None
