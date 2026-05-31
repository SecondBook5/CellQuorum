"""RAPIDS backend definitions for CellQuorum."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

from cellquorum.backends.base import BackendRequirement, BackendStatus, BaseBackend


@dataclass
class RapidsBackend(BaseBackend):
    """
    Backend for RAPIDS-singlecell GPU acceleration.

    RAPIDS-singlecell is the main GPU acceleration backend planned for
    CellQuorum's scRNA-seq preprocessing, dimensionality reduction, neighbor
    graph construction, UMAP, clustering, and related scverse-compatible stages.

    This backend checks availability without importing RAPIDS packages at module
    import time. That matters because CPU-only users should still be able to
    import CellQuorum even when CUDA, CuPy, RAPIDS, or rapids-singlecell are not
    installed.

    Args:
        name: Stable backend name.
        requirement_list: RAPIDS-related Python package requirements.
    """

    # Store the stable backend name.
    name: str = "rapids"

    # Store the backend family.
    kind: str = "rapids"

    # Store RAPIDS backend requirements.
    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            BackendRequirement(
                name="rapids_singlecell",
                requirement_type="python_package",
                required=True,
                install_hint=(
                    "Install RAPIDS-singlecell in the CellQuorum GPU environment. "
                    "Prefer the staged GPU environment instead of installing RAPIDS "
                    "into the core environment."
                ),
            ),
            BackendRequirement(
                name="cupy",
                requirement_type="python_package",
                required=True,
                install_hint=(
                    "Install CuPy with CUDA support through the CellQuorum GPU environment."
                ),
            ),
            BackendRequirement(
                name="cuml",
                requirement_type="python_package",
                required=False,
                install_hint=(
                    "Install RAPIDS cuML when GPU PCA, UMAP, neighbors, or clustering "
                    "support is enabled."
                ),
            ),
            BackendRequirement(
                name="cugraph",
                requirement_type="python_package",
                required=False,
                install_hint=(
                    "Install RAPIDS cuGraph when GPU graph operations are enabled."
                ),
            ),
        ]
    )

    def status(self) -> BackendStatus:
        """
        Check whether RAPIDS-singlecell appears available.

        The backend is considered available only when mandatory RAPIDS packages
        can be imported and CuPy can see at least one CUDA device. Optional RAPIDS
        packages are reported in details and warnings but do not block basic
        availability.

        Returns:
            BackendStatus describing RAPIDS availability, missing components,
            warnings, and backend details.
        """

        # Initialize the missing requirement list.
        missing: list[str] = []

        # Initialize the warning list.
        warnings: list[str] = []

        # Initialize additional backend details.
        details: dict[str, Any] = {
            "rapids_singlecell_available": self._python_package_available(
                "rapids_singlecell"
            ),
            "cupy_available": self._python_package_available("cupy"),
            "cuml_available": self._python_package_available("cuml"),
            "cugraph_available": self._python_package_available("cugraph"),
            "cuda_device_count": 0,
            "cuda_visible": False,
        }

        # Iterate over declared requirements.
        for requirement in self.requirements():
            # Check Python package requirements.
            if requirement.requirement_type == "python_package":
                # Determine whether the Python package can be found.
                package_available = self._python_package_available(requirement.name)

                # Record missing mandatory Python packages.
                if not package_available and requirement.required:
                    missing.append(requirement.name)

                # Record warnings for missing optional RAPIDS packages.
                if not package_available and not requirement.required:
                    warnings.append(
                        f"Optional RAPIDS package '{requirement.name}' is not available. "
                        "Some GPU-accelerated operations may fall back or be disabled."
                    )

            # Warn on non-Python requirements attached to this backend.
            else:
                # Store a warning for unexpected requirement categories.
                warnings.append(
                    f"Requirement '{requirement.name}' has unsupported type "
                    f"'{requirement.requirement_type}' for RapidsBackend."
                )

        # Check CUDA device visibility only when CuPy is importable.
        if details["cupy_available"]:
            # Count visible CUDA devices through CuPy.
            device_count = self._cupy_cuda_device_count()

            # Store the visible CUDA device count.
            details["cuda_device_count"] = device_count

            # Store whether CUDA appears visible through CuPy.
            details["cuda_visible"] = device_count > 0

            # Record a missing CUDA device when mandatory packages exist but no device is visible.
            if device_count <= 0:
                missing.append("cuda_device")
                warnings.append(
                    "CuPy is installed, but no CUDA device is visible. "
                    "RAPIDS stages cannot run without an accessible GPU."
                )

        # Warn when CuPy is missing because CUDA cannot be checked.
        else:
            # Record a warning explaining why CUDA visibility was not checked.
            warnings.append(
                "CuPy is not available, so RAPIDS CUDA device visibility could not be checked."
            )

        # Determine backend availability from missing mandatory requirements.
        available = len(missing) == 0

        # Return structured RAPIDS backend status.
        return BackendStatus(
            name=self.name,
            kind=self.kind,
            available=available,
            requirements=self.requirements(),
            missing=missing,
            warnings=warnings,
            details=details,
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
    def _cupy_cuda_device_count() -> int:
        """
        Count visible CUDA devices through CuPy.

        Returns:
            Number of visible CUDA devices. Returns zero if CuPy cannot be
            imported or CUDA device discovery fails.
        """

        # Import CuPy lazily so CPU-only users can import CellQuorum.
        try:
            # Import CuPy only during the RAPIDS availability check.
            import cupy as cp

        # Return zero when CuPy is unavailable.
        except ImportError:
            return 0

        # Try to query CUDA device count through CuPy.
        try:
            # Return the number of visible CUDA devices.
            return int(cp.cuda.runtime.getDeviceCount())

        # Return zero when CUDA runtime discovery fails.
        except Exception:
            return 0


def build_rapids_backend() -> RapidsBackend:
    """
    Build the RAPIDS-singlecell backend.

    Returns:
        Configured RapidsBackend instance.
    """

    # Return the configured RAPIDS backend.
    return RapidsBackend()