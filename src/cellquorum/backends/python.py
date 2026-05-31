"""Python backend definitions for CellQuorum."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellquorum.backends.base import BackendRequirement, BaseBackend


@dataclass
class PythonBackend(BaseBackend):
    """
    Backend for core Python execution.

    This backend represents the standard Python/scverse runtime used by the
    default CellQuorum pipeline. It checks for core Python packages that should
    be available in the main development or production environment.

    Args:
        name: Stable backend name.
        requirement_list: Python package requirements for the backend.
    """

    # Store the stable backend name.
    name: str = "python"

    # Store the backend family.
    kind: str = "python"

    # Store Python package requirements for the backend.
    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            BackendRequirement(
                name="anndata",
                requirement_type="python_package",
                required=True,
                install_hint="Install CellQuorum core dependencies.",
            ),
            BackendRequirement(
                name="scanpy",
                requirement_type="python_package",
                required=True,
                install_hint="Install CellQuorum core dependencies.",
            ),
            BackendRequirement(
                name="pandas",
                requirement_type="python_package",
                required=True,
                install_hint="Install CellQuorum core dependencies.",
            ),
            BackendRequirement(
                name="numpy",
                requirement_type="python_package",
                required=True,
                install_hint="Install CellQuorum core dependencies.",
            ),
            BackendRequirement(
                name="scipy",
                requirement_type="python_package",
                required=True,
                install_hint="Install CellQuorum core dependencies.",
            ),
        ]
    )


@dataclass
class PythonOptionalBackend(BaseBackend):
    """
    Backend for optional pure-Python analysis capabilities.

    This backend tracks optional Python packages that are useful for advanced
    analyses but should not be required for the core package to import. The
    requirements are marked optional so this backend remains available while
    still exposing optional package status in planner and provenance outputs.

    Args:
        name: Stable backend name.
        requirement_list: Optional Python package requirements.
    """

    # Store the stable backend name.
    name: str = "python_optional"

    # Store the backend family.
    kind: str = "python"

    # Store optional Python package requirements.
    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            BackendRequirement(
                name="networkx",
                requirement_type="python_package",
                required=False,
                install_hint="Install network analysis extras when network modules are enabled.",
            ),
            BackendRequirement(
                name="igraph",
                requirement_type="python_package",
                required=False,
                install_hint="Install graph clustering extras when graph modules are enabled.",
            ),
            BackendRequirement(
                name="leidenalg",
                requirement_type="python_package",
                required=False,
                install_hint="Install Leiden clustering extras when clustering modules are enabled.",
            ),
        ]
    )


def build_default_python_backends() -> list[BaseBackend]:
    """
    Build the default Python backend set.

    The backend registry uses this helper to register core and optional Python
    capability checks in one place. Keeping backend construction centralized
    makes future CLI, planner, and tests easier to maintain.

    Returns:
        List containing the core Python backend and optional Python backend.
    """

    # Return the default Python backend instances.
    return [
        PythonBackend(),
        PythonOptionalBackend(),
    ]