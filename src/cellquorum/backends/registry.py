"""Backend registry for CellQuorum execution backends."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellquorum.backends.base import Backend, BackendStatus, BackendUnavailableError
from cellquorum.backends.gpu import build_gpu_backend
from cellquorum.backends.hdwgcna_backend import build_hdwgcna_backend
from cellquorum.backends.pyscenic_backend import build_pyscenic_backend
from cellquorum.backends.python import build_default_python_backends
from cellquorum.backends.r import build_r_backend
from cellquorum.backends.rapids import build_rapids_backend
from cellquorum.backends.rscript import build_rscript_backend
from cellquorum.backends.scclr_backend import build_scclr_backend
from cellquorum.backends.sccoda_backend import build_sccoda_backend


@dataclass
class BackendRegistry:
    """
    Store and manage CellQuorum execution backends.

    The backend registry is the central lookup table for Python, R, Rscript, GPU,
    RAPIDS, and future external tool backends. Stages should ask the registry for
    backend availability instead of importing optional dependencies directly.
    This keeps optional R/GPU/generative dependencies isolated and makes the
    planner able to explain what can and cannot run before execution starts.

    Args:
        backends: Mapping from backend name to backend object.
    """

    # Store backend objects by stable backend name.
    backends: dict[str, Backend] = field(default_factory=dict)

    def register(self, backend: Backend, *, overwrite: bool = False) -> None:
        """
        Register a backend.

        Backend names must be unique unless overwrite is explicitly enabled.
        This prevents accidental replacement of a backend implementation during
        package setup or test construction.

        Args:
            backend: Backend object to register.
            overwrite: Whether to replace an existing backend with the same name.

        Raises:
            ValueError: If a backend with the same name already exists and
                overwrite is False.
        """

        # Check whether the backend name is already registered.
        if backend.name in self.backends and not overwrite:
            # Raise a clear error instead of silently replacing the backend.
            raise ValueError(
                f"Backend '{backend.name}' is already registered. "
                "Pass overwrite=True to replace it."
            )

        # Register the backend by stable name.
        self.backends[backend.name] = backend

    def has(self, name: str) -> bool:
        """
        Return whether a backend is registered.

        Args:
            name: Backend name to check.

        Returns:
            True when the backend is registered, otherwise False.
        """

        # Return whether the requested backend name exists in the registry.
        return name in self.backends

    def get(self, name: str) -> Backend:
        """
        Retrieve a registered backend.

        Args:
            name: Backend name to retrieve.

        Returns:
            Registered backend object.

        Raises:
            KeyError: If the backend is not registered.
        """

        # Return the backend when it exists.
        if name in self.backends:
            return self.backends[name]

        # Raise a clear error listing the available backend names.
        raise KeyError(
            f"Backend '{name}' is not registered. "
            f"Available backends: {', '.join(self.names()) or 'none'}."
        )

    def names(self) -> list[str]:
        """
        Return registered backend names in sorted order.

        Returns:
            Sorted list of registered backend names.
        """

        # Return backend names in deterministic sorted order.
        return sorted(self.backends)

    def statuses(self) -> dict[str, BackendStatus]:
        """
        Return availability status for every registered backend.

        Returns:
            Mapping from backend name to BackendStatus.
        """

        # Build a status mapping by asking every backend to check availability.
        return {name: backend.status() for name, backend in self.backends.items()}

    def status(self, name: str) -> BackendStatus:
        """
        Return availability status for one registered backend.

        Args:
            name: Backend name to check.

        Returns:
            BackendStatus for the requested backend.
        """

        # Retrieve the backend and return its availability status.
        return self.get(name).status()

    def available(self, name: str) -> bool:
        """
        Return whether a registered backend is currently available.

        Args:
            name: Backend name to check.

        Returns:
            True if the backend status is available, otherwise False.
        """

        # Return the availability flag from the backend status.
        return self.status(name).available

    def require(self, name: str) -> Backend:
        """
        Retrieve a backend and require that it is available.

        Stages should use this method when they cannot proceed without a specific
        backend. The resulting errors are explicit and can be caught by the CLI,
        planner, or report generator.

        Args:
            name: Backend name to retrieve and validate.

        Returns:
            Available backend object.

        Raises:
            BackendUnavailableError: If the backend is registered but unavailable.
            KeyError: If the backend is not registered.
        """

        # Retrieve the requested backend.
        backend = self.get(name)

        # Get the backend availability status.
        status = backend.status()

        # Raise a backend-specific error if unavailable.
        status.raise_if_unavailable()

        # Return the available backend.
        return backend

    def require_any(self, names: list[str]) -> Backend:
        """
        Return the first available backend from a candidate list.

        This supports fallback logic such as trying RAPIDS before CPU or rpy2
        before Rscript. The method preserves the caller-provided priority order.

        Args:
            names: Candidate backend names in priority order.

        Returns:
            First available backend from the candidate list.

        Raises:
            ValueError: If the candidate list is empty.
            BackendUnavailableError: If no candidate backend is available.
            KeyError: If any requested backend is not registered.
        """

        # Reject empty fallback lists because they cannot produce a backend.
        if not names:
            raise ValueError("require_any expected at least one backend name.")

        # Store unavailable backend messages for a final aggregated error.
        unavailable_messages: list[str] = []

        # Check each backend in caller-provided priority order.
        for name in names:
            # Retrieve the backend, raising KeyError if the name is invalid.
            backend = self.get(name)

            # Check the backend status.
            status = backend.status()

            # Return the first backend that is available.
            if status.available:
                return backend

            # Store a readable message for the final error if this backend is unavailable.
            missing = ", ".join(status.missing) if status.missing else "unknown requirement"
            unavailable_messages.append(f"{name}: {missing}")

        # Raise a clear error after all candidates fail.
        raise BackendUnavailableError(
            backend_name=",".join(names),
            message=(
                "None of the requested backends are available. "
                f"Checked: {'; '.join(unavailable_messages)}."
            ),
        )

    def to_status_table(self) -> list[dict[str, object]]:
        """
        Return backend statuses as a JSON-serializable table.

        This format is useful for planner output, backend status reports, and
        provenance artifacts.

        Returns:
            List of dictionaries describing registered backend status.
        """

        # Initialize status rows.
        rows: list[dict[str, object]] = []

        # Iterate over backend statuses in deterministic backend-name order.
        for name in self.names():
            # Get the backend status.
            status = self.status(name)

            # Add one JSON-serializable status row.
            rows.append(
                {
                    "name": status.name,
                    "kind": status.kind,
                    "available": status.available,
                    "missing": list(status.missing),
                    "warnings": list(status.warnings),
                    "requirements": [
                        {
                            "name": requirement.name,
                            "requirement_type": requirement.requirement_type,
                            "required": requirement.required,
                            "install_hint": requirement.install_hint,
                        }
                        for requirement in status.requirements
                    ],
                    "details": dict(status.details),
                }
            )

        # Return the status rows.
        return rows


def build_default_backend_registry() -> BackendRegistry:
    """
    Build the default CellQuorum backend registry.

    This helper registers all baseline backend families without importing heavy
    optional dependencies at module import time. Individual backend status checks
    remain lazy, so users can install and import CellQuorum in CPU-only
    environments while still seeing R, GPU, and RAPIDS availability in planner
    output.

    Returns:
        BackendRegistry containing Python, R, Rscript, GPU, and RAPIDS backends.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Register core and optional Python backends.
    for backend in build_default_python_backends():
        registry.register(backend)

    # Register the in-process rpy2 backend.
    registry.register(build_r_backend())

    # Register the batch/HPC-friendly Rscript backend.
    registry.register(build_rscript_backend())

    # Register the general GPU backend.
    registry.register(build_gpu_backend())

    # Register the RAPIDS-singlecell backend.
    registry.register(build_rapids_backend())

    # Register the isolated-env scclr backend (PFlog1pPF + sparse PCA).
    registry.register(build_scclr_backend())

    # Register the isolated-env scCODA backend (compositional DA).
    registry.register(build_sccoda_backend())

    # Register the isolated-env hdWGCNA backend (co-expression modules).
    registry.register(build_hdwgcna_backend())

    # Register the isolated-env pySCENIC backend (gene regulatory networks).
    registry.register(build_pyscenic_backend())

    # Return the populated backend registry.
    return registry
