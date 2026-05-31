"""GPU backend definitions for CellQuorum."""

from __future__ import annotations

import importlib.util
import subprocess
from dataclasses import dataclass, field
from typing import Any

from cellquorum.backends.base import BackendRequirement, BackendStatus, BaseBackend


@dataclass
class GPUBackend(BaseBackend):
    """
    Backend for general GPU availability checks.

    CellQuorum treats GPU acceleration as a first-class execution option, but it
    must remain optional. This backend checks whether GPU-related Python packages
    and CUDA visibility are available without importing heavy CUDA libraries at
    module import time.

    This backend is intentionally general. RAPIDS-specific checks belong in the
    RAPIDS backend, while PyTorch/scVI checks can use this backend or a future
    torch-specific backend.

    Args:
        name: Stable backend name.
        cuda_check_command: Command used to check visible NVIDIA GPUs.
        timeout_seconds: Maximum time allowed for GPU availability checks.
        requirement_list: Requirements checked by this backend.
    """

    # Store the stable backend name.
    name: str = "gpu"

    # Store the backend family.
    kind: str = "gpu"

    # Store the command used to check visible NVIDIA GPUs.
    cuda_check_command: tuple[str, ...] = ("nvidia-smi",)

    # Store the timeout used for command-line GPU checks.
    timeout_seconds: int = 10

    # Store GPU backend requirements.
    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            BackendRequirement(
                name="nvidia-smi",
                requirement_type="executable",
                required=False,
                install_hint=(
                    "Install NVIDIA drivers and ensure nvidia-smi is available on PATH "
                    "if CUDA GPU execution is desired."
                ),
            ),
            BackendRequirement(
                name="torch",
                requirement_type="python_package",
                required=False,
                install_hint="Install PyTorch with CUDA support for GPU model backends.",
            ),
        ]
    )

    def status(self) -> BackendStatus:
        """
        Check whether general GPU execution appears available.

        This method does not require GPU availability because CPU-only CellQuorum
        runs must remain valid. Instead, it reports GPU visibility, optional
        Python package availability, and warnings that can be used by the planner.

        Returns:
            BackendStatus describing GPU-related availability and warnings.
        """

        # Initialize the missing requirement list.
        missing: list[str] = []

        # Initialize the warning list.
        warnings: list[str] = []

        # Initialize additional backend details.
        details: dict[str, Any] = {
            "cuda_check_command": list(self.cuda_check_command),
            "timeout_seconds": self.timeout_seconds,
            "nvidia_smi_available": self._executable_available(self.cuda_check_command[0]),
            "nvidia_gpu_visible": False,
            "torch_available": self._python_package_available("torch"),
            "torch_cuda_available": False,
        }

        # Check NVIDIA GPU visibility through nvidia-smi when available.
        if details["nvidia_smi_available"]:
            # Run nvidia-smi to check whether a GPU is visible.
            gpu_visible = self._nvidia_gpu_visible()

            # Store the GPU visibility result.
            details["nvidia_gpu_visible"] = gpu_visible

            # Warn when nvidia-smi exists but no GPU is visible.
            if not gpu_visible:
                warnings.append(
                    "nvidia-smi is available, but no visible NVIDIA GPU was detected."
                )

        # Warn when nvidia-smi is not available.
        else:
            # Store a non-fatal warning because CPU-only runs are allowed.
            warnings.append(
                "nvidia-smi was not found. GPU stages will be unavailable unless another "
                "GPU backend can verify acceleration."
            )

        # Check PyTorch CUDA availability only if torch can be imported.
        if details["torch_available"]:
            # Store whether torch reports CUDA availability.
            details["torch_cuda_available"] = self._torch_cuda_available()

            # Warn when torch is present but CUDA is not available.
            if not details["torch_cuda_available"]:
                warnings.append("PyTorch is installed, but torch.cuda.is_available() is False.")

        # Warn when torch is unavailable because GPU model backends may need it.
        else:
            # Store a non-fatal warning because RAPIDS-only GPU runs may not need torch.
            warnings.append("PyTorch is not installed. GPU model backends may be unavailable.")

        # Treat the generic GPU backend as available only when either NVIDIA or torch CUDA is visible.
        available = bool(details["nvidia_gpu_visible"] or details["torch_cuda_available"])

        # Return structured backend status.
        return BackendStatus(
            name=self.name,
            kind=self.kind,
            available=available,
            requirements=self.requirements(),
            missing=missing,
            warnings=warnings,
            details=details,
        )

    def _nvidia_gpu_visible(self) -> bool:
        """
        Check whether nvidia-smi reports at least one visible GPU.

        Returns:
            True if nvidia-smi succeeds and reports a GPU, otherwise False.
        """

        # Execute nvidia-smi with a compact GPU query.
        try:
            # Run a short nvidia-smi query and capture output.
            result = subprocess.run(
                [
                    self.cuda_check_command[0],
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )

        # Treat subprocess failures as no visible GPU.
        except (OSError, subprocess.TimeoutExpired):
            # Return False when the command cannot complete safely.
            return False

        # Return False when nvidia-smi exits with an error.
        if result.returncode != 0:
            return False

        # Return whether at least one non-empty GPU name was reported.
        return bool(result.stdout.strip())

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
    def _torch_cuda_available() -> bool:
        """
        Check whether PyTorch reports CUDA availability.

        Returns:
            True if torch is installed and CUDA is available, otherwise False.
        """

        # Import torch lazily so CPU-only users can import CellQuorum.
        try:
            # Import PyTorch only during the availability check.
            import torch

        # Return False when PyTorch is not installed or cannot be imported.
        except ImportError:
            # Report CUDA as unavailable without raising.
            return False

        # Return whether PyTorch can access CUDA.
        return bool(torch.cuda.is_available())


def build_gpu_backend(
    *,
    timeout_seconds: int = 10,
) -> GPUBackend:
    """
    Build the general GPU backend.

    Args:
        timeout_seconds: Maximum time allowed for command-line GPU checks.

    Returns:
        Configured GPUBackend instance.
    """

    # Return the configured general GPU backend.
    return GPUBackend(timeout_seconds=timeout_seconds)