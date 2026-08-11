"""Tests for CellQuorum backend registry behavior."""

from __future__ import annotations

import pytest

from cellquorum.backends.base import BackendRequirement, BackendUnavailableError, BaseBackend
from cellquorum.backends.registry import BackendRegistry, build_default_backend_registry


def test_backend_registry_registers_and_reports_backend_names() -> None:
    """
    Verify that BackendRegistry can register backends and report their names.

    The registry is the central lookup table for optional execution backends.
    This test confirms that registered backend names are tracked in deterministic
    sorted order so planner outputs, reports, and provenance files are stable.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Create a Python backend with no requirements.
    python_backend = BaseBackend(name="python", kind="python")

    # Create an external backend with no requirements.
    external_backend = BaseBackend(name="external", kind="external")

    # Register the external backend first to test sorted output.
    registry.register(external_backend)

    # Register the Python backend second to test sorted output.
    registry.register(python_backend)

    # Confirm both backend names are reported in sorted order.
    assert registry.names() == ["external", "python"]

    # Confirm the registry reports existing backends.
    assert registry.has("python") is True

    # Confirm the registry reports missing backends as absent.
    assert registry.has("missing") is False


def test_backend_registry_rejects_duplicate_registration_without_overwrite() -> None:
    """
    Verify that duplicate backend registration fails unless overwrite is enabled.

    Silent backend replacement would make execution behavior difficult to audit.
    The registry should therefore reject duplicate names by default.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Create the first backend.
    first_backend = BaseBackend(name="python", kind="python")

    # Create a second backend with the same name.
    second_backend = BaseBackend(name="python", kind="python")

    # Register the first backend.
    registry.register(first_backend)

    # Confirm registering the second backend without overwrite raises a clear error.
    with pytest.raises(ValueError, match="already registered"):
        registry.register(second_backend)


def test_backend_registry_allows_duplicate_registration_with_overwrite() -> None:
    """
    Verify that explicit overwrite replaces an existing backend.

    Overwrite support is useful for tests and advanced users who intentionally
    want to replace a backend implementation.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Create the original backend.
    original_backend = BaseBackend(name="python", kind="python")

    # Create a replacement backend with the same name and a visible requirement.
    replacement_backend = BaseBackend(
        name="python",
        kind="python",
        requirement_list=[
            BackendRequirement(
                name="anndata",
                requirement_type="python_package",
                required=True,
            )
        ],
    )

    # Register the original backend.
    registry.register(original_backend)

    # Register the replacement backend with explicit overwrite.
    registry.register(replacement_backend, overwrite=True)

    # Confirm the registry now returns the replacement backend.
    assert registry.get("python") is replacement_backend


def test_backend_registry_get_returns_registered_backend() -> None:
    """
    Verify that BackendRegistry.get returns a registered backend.

    Stage code and planner code rely on this method to retrieve backend objects
    by stable backend name.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Create a backend.
    backend = BaseBackend(name="python", kind="python")

    # Register the backend.
    registry.register(backend)

    # Confirm the same backend object is returned.
    assert registry.get("python") is backend


def test_backend_registry_get_raises_clear_error_for_missing_backend() -> None:
    """
    Verify that BackendRegistry.get raises a clear error for unknown backends.

    Missing backend names should fail with available backend names listed so
    users can quickly identify spelling or configuration mistakes.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Register one available backend.
    registry.register(BaseBackend(name="python", kind="python"))

    # Confirm missing backend lookup raises a helpful KeyError.
    with pytest.raises(KeyError, match="Available backends: python"):
        registry.get("missing")


def test_backend_registry_status_returns_single_backend_status() -> None:
    """
    Verify that BackendRegistry.status returns one backend status object.

    This method is used by planners and stages that need to check one specific
    backend before execution.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Register an available Python backend.
    registry.register(
        BaseBackend(
            name="python",
            kind="python",
            requirement_list=[
                BackendRequirement(
                    name="anndata",
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Get the backend status.
    status = registry.status("python")

    # Confirm the returned status belongs to the requested backend.
    assert status.name == "python"

    # Confirm the backend is available in the test environment.
    assert status.available is True


def test_backend_registry_statuses_returns_all_backend_statuses() -> None:
    """
    Verify that BackendRegistry.statuses returns status for all backends.

    Backend status summaries feed planner output, provenance, and report files.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Register a Python backend.
    registry.register(BaseBackend(name="python", kind="python"))

    # Register an external backend.
    registry.register(BaseBackend(name="external", kind="external"))

    # Get all statuses.
    statuses = registry.statuses()

    # Confirm both backends are present in the status mapping.
    assert set(statuses) == {"python", "external"}

    # Confirm the Python backend status has the expected name.
    assert statuses["python"].name == "python"


def test_backend_registry_available_reports_true_for_available_backend() -> None:
    """
    Verify that BackendRegistry.available returns True for available backends.

    This convenience method supports simple planner checks.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Register an available backend.
    registry.register(BaseBackend(name="python", kind="python"))

    # Confirm the backend is reported as available.
    assert registry.available("python") is True


def test_backend_registry_available_reports_false_for_unavailable_backend() -> None:
    """
    Verify that BackendRegistry.available returns False for unavailable backends.

    Unavailable backends should be detected before stages try to execute optional
    methods.
    """

    # Create a deliberately impossible package name.
    missing_package = "cellquorum_missing_registry_package_12345"

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Register a backend that requires the impossible package.
    registry.register(
        BaseBackend(
            name="missing_backend",
            kind="python",
            requirement_list=[
                BackendRequirement(
                    name=missing_package,
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Confirm the backend is reported as unavailable.
    assert registry.available("missing_backend") is False


def test_backend_registry_require_returns_available_backend() -> None:
    """
    Verify that BackendRegistry.require returns an available backend.

    Stages should use this method when a requested backend is mandatory.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Create an available backend.
    backend = BaseBackend(name="python", kind="python")

    # Register the available backend.
    registry.register(backend)

    # Confirm require returns the available backend.
    assert registry.require("python") is backend


def test_backend_registry_require_raises_for_unavailable_backend() -> None:
    """
    Verify that BackendRegistry.require raises for unavailable backends.

    This prevents stages from reaching backend-specific execution paths when
    required dependencies are missing.
    """

    # Create a deliberately impossible package name.
    missing_package = "cellquorum_missing_required_package_12345"

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Register an unavailable backend.
    registry.register(
        BaseBackend(
            name="missing_backend",
            kind="python",
            requirement_list=[
                BackendRequirement(
                    name=missing_package,
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Confirm requiring the unavailable backend raises a backend-specific error.
    with pytest.raises(BackendUnavailableError, match=missing_package):
        registry.require("missing_backend")


def test_backend_registry_require_any_returns_first_available_backend() -> None:
    """
    Verify that require_any returns the first available backend by priority.

    This supports ordered fallback behavior such as RAPIDS first, then CPU.
    """

    # Create a deliberately impossible package name.
    missing_package = "cellquorum_missing_priority_package_12345"

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Register an unavailable high-priority backend.
    registry.register(
        BaseBackend(
            name="rapids",
            kind="rapids",
            requirement_list=[
                BackendRequirement(
                    name=missing_package,
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Register an available fallback backend.
    cpu_backend = BaseBackend(name="python", kind="python")

    # Add the fallback backend to the registry.
    registry.register(cpu_backend)

    # Confirm the first available backend is returned.
    assert registry.require_any(["rapids", "python"]) is cpu_backend


def test_backend_registry_require_any_rejects_empty_candidate_list() -> None:
    """
    Verify that require_any rejects empty candidate lists.

    Empty fallback lists indicate a configuration or caller error and should fail
    clearly.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Confirm an empty candidate list raises a clear error.
    with pytest.raises(ValueError, match="at least one backend"):
        registry.require_any([])


def test_backend_registry_require_any_raises_when_no_backends_available() -> None:
    """
    Verify that require_any raises when all candidate backends are unavailable.

    The final error should summarize all checked backend names and missing
    requirements.
    """

    # Create two deliberately impossible package names.
    first_missing_package = "cellquorum_missing_first_package_12345"
    second_missing_package = "cellquorum_missing_second_package_12345"

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Register the first unavailable backend.
    registry.register(
        BaseBackend(
            name="first",
            kind="python",
            requirement_list=[
                BackendRequirement(
                    name=first_missing_package,
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Register the second unavailable backend.
    registry.register(
        BaseBackend(
            name="second",
            kind="python",
            requirement_list=[
                BackendRequirement(
                    name=second_missing_package,
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Confirm the aggregated error includes both missing backends.
    with pytest.raises(BackendUnavailableError, match="None of the requested backends"):
        registry.require_any(["first", "second"])


def test_backend_registry_to_status_table_returns_json_serializable_rows() -> None:
    """
    Verify that BackendRegistry can export backend statuses as table rows.

    These rows are intended for planner output, backend status reports, and
    provenance artifacts.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Register a backend with one requirement.
    registry.register(
        BaseBackend(
            name="python",
            kind="python",
            requirement_list=[
                BackendRequirement(
                    name="anndata",
                    requirement_type="python_package",
                    required=True,
                    install_hint="Install with pip install anndata.",
                )
            ],
        )
    )

    # Convert backend statuses to table rows.
    rows = registry.to_status_table()

    # Confirm one row is returned.
    assert len(rows) == 1

    # Confirm the backend name is included.
    assert rows[0]["name"] == "python"

    # Confirm availability is included.
    assert rows[0]["available"] is True

    # Confirm requirements are represented as dictionaries.
    assert rows[0]["requirements"] == [
        {
            "name": "anndata",
            "requirement_type": "python_package",
            "required": True,
            "install_hint": "Install with pip install anndata.",
        }
    ]


def test_build_default_backend_registry_registers_expected_backends() -> None:
    """
    Verify that the default registry includes all baseline backend families.

    CellQuorum should know about Python, optional Python, R, Rscript, GPU, and
    RAPIDS from the start. These backends may not all be available in a given
    environment, but they should be registered so the planner can report their
    availability clearly.
    """

    # Build the default CellQuorum backend registry.
    registry = build_default_backend_registry()

    # Confirm the expected default backend names are registered.
    assert registry.names() == [
        "gpu",
        "hdwgcna_r",
        "python",
        "python_optional",
        "r",
        "rapids",
        "rscript",
        "scclr",
        "sccoda",
    ]


def test_build_default_backend_registry_status_table_contains_expected_backends() -> None:
    """
    Verify that the default registry can produce planner-friendly status rows.

    This test does not require R, RAPIDS, or GPU availability. It only checks
    that every default backend is represented in the status table so the planner
    can explain which advanced capabilities are available or missing.
    """

    # Build the default CellQuorum backend registry.
    registry = build_default_backend_registry()

    # Convert backend statuses to table rows.
    rows = registry.to_status_table()

    # Extract backend names from the status rows.
    row_names = {row["name"] for row in rows}

    # Confirm all expected backend names are represented.
    assert row_names == {
        "gpu",
        "hdwgcna_r",
        "python",
        "python_optional",
        "r",
        "rapids",
        "rscript",
        "scclr",
        "sccoda",
    }

    # Confirm every row includes the standard status fields.
    for row in rows:
        # Confirm the backend name field exists.
        assert "name" in row

        # Confirm the backend kind field exists.
        assert "kind" in row

        # Confirm the availability field exists.
        assert "available" in row

        # Confirm the missing dependency list exists.
        assert "missing" in row

        # Confirm the warning list exists.
        assert "warnings" in row

        # Confirm the requirements list exists.
        assert "requirements" in row

        # Confirm the details dictionary exists.
        assert "details" in row
