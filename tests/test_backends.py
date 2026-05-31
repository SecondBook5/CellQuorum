"""Tests for CellQuorum backend availability contracts."""

from __future__ import annotations

import pytest

from cellquorum.backends.base import (
    BackendRequirement,
    BackendStatus,
    BackendUnavailableError,
    BaseBackend,
)


def test_backend_requirement_stores_requirement_metadata() -> None:
    """
    Verify that BackendRequirement stores dependency metadata.

    Backend requirements are used by the planner and backend registry to explain
    which Python packages, executables, R packages, CUDA components, or external
    tools are needed before a stage can run.
    """

    # Create a representative Python package requirement.
    requirement = BackendRequirement(
        name="anndata",
        requirement_type="python_package",
        required=True,
        install_hint="Install with pip install anndata.",
    )

    # Confirm the requirement name was stored.
    assert requirement.name == "anndata"

    # Confirm the requirement type was stored.
    assert requirement.requirement_type == "python_package"

    # Confirm the requirement is marked as mandatory.
    assert requirement.required is True

    # Confirm the install hint was stored.
    assert requirement.install_hint == "Install with pip install anndata."


def test_base_backend_reports_available_python_package() -> None:
    """
    Verify that BaseBackend detects installed Python packages.

    The base backend should be able to check importable Python dependencies so
    Python-only backends can report availability without custom logic.
    """

    # Create a backend requiring a package that should exist in the test environment.
    backend = BaseBackend(
        name="python_core",
        kind="python",
        requirement_list=[
            BackendRequirement(
                name="anndata",
                requirement_type="python_package",
                required=True,
            )
        ],
    )

    # Check the backend availability status.
    status = backend.status()

    # Confirm the backend is available.
    assert status.available is True

    # Confirm no required dependencies are missing.
    assert status.missing == []

    # Confirm the backend name is preserved.
    assert status.name == "python_core"

    # Confirm the backend kind is preserved.
    assert status.kind == "python"


def test_base_backend_reports_missing_python_package() -> None:
    """
    Verify that BaseBackend detects missing Python packages.

    Missing mandatory Python packages should make a backend unavailable and
    should appear in the missing requirement list.
    """

    # Create a deliberately impossible package name.
    missing_package = "cellquorum_package_that_should_not_exist_12345"

    # Create a backend requiring the impossible package.
    backend = BaseBackend(
        name="missing_python_backend",
        kind="python",
        requirement_list=[
            BackendRequirement(
                name=missing_package,
                requirement_type="python_package",
                required=True,
            )
        ],
    )

    # Check the backend availability status.
    status = backend.status()

    # Confirm the backend is unavailable.
    assert status.available is False

    # Confirm the missing package appears in the missing list.
    assert status.missing == [missing_package]


def test_base_backend_ignores_missing_optional_python_package() -> None:
    """
    Verify that optional missing requirements do not block backend availability.

    Optional dependencies should be visible in the requirements list, but they
    should not make a backend unavailable when absent.
    """

    # Create a deliberately impossible optional package name.
    optional_package = "cellquorum_optional_package_that_should_not_exist_12345"

    # Create a backend with the impossible package marked optional.
    backend = BaseBackend(
        name="optional_python_backend",
        kind="python",
        requirement_list=[
            BackendRequirement(
                name=optional_package,
                requirement_type="python_package",
                required=False,
            )
        ],
    )

    # Check the backend availability status.
    status = backend.status()

    # Confirm the backend remains available.
    assert status.available is True

    # Confirm no mandatory requirements are missing.
    assert status.missing == []


def test_base_backend_reports_available_executable() -> None:
    """
    Verify that BaseBackend detects available command-line executables.

    Executable checks are needed for Rscript, external tools, and later
    production wrappers that call command-line methods.
    """

    # Create a backend requiring the Python executable.
    backend = BaseBackend(
        name="python_executable_backend",
        kind="external",
        requirement_list=[
            BackendRequirement(
                name="python",
                requirement_type="executable",
                required=True,
            )
        ],
    )

    # Check the backend availability status.
    status = backend.status()

    # Confirm the backend is available.
    assert status.available is True

    # Confirm no required executables are missing.
    assert status.missing == []


def test_base_backend_reports_missing_executable() -> None:
    """
    Verify that BaseBackend detects missing command-line executables.

    Missing mandatory executables should make a backend unavailable and appear in
    the missing requirement list.
    """

    # Create a deliberately impossible executable name.
    missing_executable = "cellquorum_executable_that_should_not_exist_12345"

    # Create a backend requiring the impossible executable.
    backend = BaseBackend(
        name="missing_executable_backend",
        kind="external",
        requirement_list=[
            BackendRequirement(
                name=missing_executable,
                requirement_type="executable",
                required=True,
            )
        ],
    )

    # Check the backend availability status.
    status = backend.status()

    # Confirm the backend is unavailable.
    assert status.available is False

    # Confirm the missing executable appears in the missing list.
    assert status.missing == [missing_executable]


def test_base_backend_warns_for_r_package_requirements() -> None:
    """
    Verify that BaseBackend warns when asked to check R package requirements.

    R packages require an R-specific backend because availability depends on the
    selected R executable, library paths, and execution mode.
    """

    # Create a backend with an R package requirement.
    backend = BaseBackend(
        name="r_package_backend",
        kind="r",
        requirement_list=[
            BackendRequirement(
                name="scDblFinder",
                requirement_type="r_package",
                required=True,
            )
        ],
    )

    # Check the backend availability status.
    status = backend.status()

    # Confirm BaseBackend does not mark the R package as missing directly.
    assert status.missing == []

    # Confirm a warning explains that R packages need R-specific checks.
    assert any("R package" in warning for warning in status.warnings)


def test_base_backend_warns_for_cuda_requirements() -> None:
    """
    Verify that BaseBackend warns when asked to check CUDA requirements.

    CUDA availability requires GPU-specific checks, so the base backend should
    not pretend it can validate CUDA correctly.
    """

    # Create a backend with a CUDA requirement.
    backend = BaseBackend(
        name="cuda_backend",
        kind="gpu",
        requirement_list=[
            BackendRequirement(
                name="cuda",
                requirement_type="cuda",
                required=True,
            )
        ],
    )

    # Check the backend availability status.
    status = backend.status()

    # Confirm BaseBackend does not mark CUDA as missing directly.
    assert status.missing == []

    # Confirm a warning explains that CUDA needs GPU-specific checks.
    assert any("CUDA" in warning for warning in status.warnings)


def test_backend_status_raise_if_unavailable_does_nothing_when_available() -> None:
    """
    Verify that BackendStatus.raise_if_unavailable is silent when available.

    Stages should be able to call this method safely before backend-specific
    execution.
    """

    # Create an available backend status.
    status = BackendStatus(
        name="available_backend",
        kind="python",
        available=True,
    )

    # Confirm no exception is raised.
    status.raise_if_unavailable()


def test_backend_status_raise_if_unavailable_raises_clear_error() -> None:
    """
    Verify that BackendStatus.raise_if_unavailable raises a backend-specific error.

    This keeps backend failures actionable and easy for the CLI or planner to
    catch and report.
    """

    # Create an unavailable backend status.
    status = BackendStatus(
        name="missing_backend",
        kind="python",
        available=False,
        missing=["missing_dependency"],
    )

    # Confirm the unavailable backend raises the expected error type.
    with pytest.raises(BackendUnavailableError, match="missing_dependency"):
        status.raise_if_unavailable()


def test_backend_unavailable_error_stores_backend_name() -> None:
    """
    Verify that BackendUnavailableError stores the backend name.

    The CLI and planner can use this attribute to produce structured error
    reports rather than parsing exception strings.
    """

    # Create a backend availability error.
    error = BackendUnavailableError(
        backend_name="example_backend",
        message="Example backend is unavailable.",
    )

    # Confirm the backend name is stored.
    assert error.backend_name == "example_backend"

    # Confirm the error message is preserved.
    assert str(error) == "Example backend is unavailable."
