"""rpy2 backend definitions for CellQuorum."""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from typing import Any

from cellquorum.backends.base import BackendRequirement, BackendStatus, BaseBackend


@dataclass
class RBackend(BaseBackend):
    """
    Backend for in-process R/Bioconductor execution through rpy2.

    CellQuorum treats R/Bioconductor as a first-class backend because many strong
    single-cell methods are R-native, including scDblFinder, scran, edgeR,
    limma, dreamlet, SingleR, UCell, AUCell, STRINGdb, viper, and RTN. This
    backend checks whether rpy2 is importable and can optionally check whether
    required R packages are installed in the active R environment.

    The backend does not import rpy2 at module import time. That is deliberate:
    users should be able to import CellQuorum in a Python-only environment
    without R or rpy2 installed. rpy2 is imported only during availability checks
    or execution.

    Args:
        name: Stable backend name.
        timeout_seconds: Reserved timeout value for future long-running checks.
        requirement_list: Requirements checked by this backend.
    """

    # Store the stable backend name.
    name: str = "r"

    # Store the backend family.
    kind: str = "r"

    # Store the timeout used by future R availability or execution checks.
    timeout_seconds: int = 30

    # Store R backend requirements.
    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            BackendRequirement(
                name="rpy2",
                requirement_type="python_package",
                required=True,
                install_hint="Install CellQuorum R extras with `pip install cellquorum[r]`.",
            )
        ]
    )

    def status(self) -> BackendStatus:
        """
        Check whether the rpy2 backend is available.

        This method checks whether rpy2 is importable and whether any declared R
        package requirements can be loaded through rpy2. It returns structured
        status metadata instead of raising immediately so the planner can explain
        backend availability before a run begins.

        Returns:
            BackendStatus describing rpy2 and R package availability.
        """

        # Initialize the missing requirement list.
        missing: list[str] = []

        # Initialize the warning list.
        warnings: list[str] = []

        # Initialize additional backend details.
        details: dict[str, Any] = {
            "timeout_seconds": self.timeout_seconds,
            "execution_mode": "rpy2",
        }

        # Check whether rpy2 is importable before attempting any R calls.
        if not self._python_package_available("rpy2"):
            # Record rpy2 as a missing mandatory requirement.
            missing.append("rpy2")

            # Return immediately because R package checks require rpy2.
            return BackendStatus(
                name=self.name,
                kind=self.kind,
                available=False,
                requirements=self.requirements(),
                missing=missing,
                warnings=warnings,
                details=details,
            )

        # Iterate over all declared requirements.
        for requirement in self.requirements():
            # Skip the rpy2 Python package because it was already checked above.
            if requirement.requirement_type == "python_package" and requirement.name == "rpy2":
                continue

            # Check R package requirements through rpy2.
            if requirement.requirement_type == "r_package":
                # Check whether the requested R package is available.
                package_available = self._r_package_available(requirement.name)

                # Record missing mandatory R packages.
                if not package_available and requirement.required:
                    missing.append(requirement.name)

            # Warn if executables are accidentally attached to this backend.
            elif requirement.requirement_type == "executable":
                # Store a warning because executable checks belong to Rscript or external backends.
                warnings.append(
                    f"Requirement '{requirement.name}' is an executable and should be checked "
                    "by an Rscript or external backend."
                )

            # Warn if CUDA is accidentally attached to this backend.
            elif requirement.requirement_type == "cuda":
                # Store a warning because CUDA checks belong to GPU backends.
                warnings.append(
                    f"Requirement '{requirement.name}' is CUDA-related and should be checked "
                    "by a GPU backend."
                )

            # Warn on unsupported requirement types.
            elif requirement.requirement_type not in {"python_package", "r_package"}:
                # Store a warning for unsupported requirement categories.
                warnings.append(
                    f"Requirement '{requirement.name}' has unsupported type "
                    f"'{requirement.requirement_type}' for RBackend."
                )

        # Determine backend availability from missing mandatory requirements.
        available = len(missing) == 0

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

    def run_expression(self, expression: str) -> Any:
        """
        Execute a short R expression through rpy2.

        This helper is intended for lightweight checks and future small adapter
        calls. Long-running R methods should usually be wrapped in explicit
        input/output adapters that preserve R logs, package versions, and
        intermediate files.

        Args:
            expression: R expression to evaluate through rpy2.

        Returns:
            Result returned by rpy2's R evaluator.

        Raises:
            TypeError: If expression is not a string.
            ValueError: If expression is empty.
            ImportError: If rpy2 is unavailable.
        """

        # Validate that the expression is a string.
        if not isinstance(expression, str):
            # Raise a specific type error for malformed calls.
            raise TypeError(
                "run_expression expected a string R expression. "
                f"Received: {type(expression).__name__}"
            )

        # Validate that the expression is not empty.
        if not expression.strip():
            # Raise a specific value error for empty expressions.
            raise ValueError("run_expression expected a non-empty R expression.")

        # Import rpy2 lazily so Python-only users can import CellQuorum.
        try:
            # Import the rpy2 R object for expression execution.
            from rpy2.robjects import r
        except ImportError as error:
            # Raise a clearer backend-oriented import error.
            raise ImportError(
                "The rpy2 backend was requested, but rpy2 is not installed. "
                "Install CellQuorum R extras or use the Rscript backend."
            ) from error

        # Execute and return the R expression result.
        return r(expression)

    def _r_package_available(self, package_name: str) -> bool:
        """
        Check whether an R package is available through rpy2.

        The package name is validated before being inserted into the R expression
        so malformed names cannot create confusing R calls.

        Args:
            package_name: R package name to check.

        Returns:
            True if the package is available, otherwise False.

        Raises:
            ValueError: If the package name contains unsupported characters.
        """

        # Validate the R package name before building an R expression.
        if not self._valid_r_package_name(package_name):
            # Raise a specific value error for malformed R package names.
            raise ValueError(
                "Invalid R package name. Expected letters, numbers, dots, or underscores "
                f"starting with a letter. Received: {package_name}"
            )

        # Build the R expression for checking package availability.
        expression = f"requireNamespace('{package_name}', quietly = TRUE)"

        # Evaluate the package availability expression.
        result = self.run_expression(expression)

        # Convert the rpy2 vector-like result into a Python boolean.
        try:
            # Return the first element as a boolean.
            return bool(result[0])
        except (TypeError, IndexError, ValueError):
            # Return False when the result cannot be interpreted safely.
            return False

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
    def _valid_r_package_name(package_name: str) -> bool:
        """
        Validate an R package name for safe use in a simple R expression.

        Args:
            package_name: Candidate R package name.

        Returns:
            True if the package name is valid, otherwise False.
        """

        # Return False for non-string package names.
        if not isinstance(package_name, str):
            return False

        # Return whether the package name matches a conservative R package pattern.
        return re.fullmatch(r"[A-Za-z][A-Za-z0-9._]*", package_name) is not None


def build_r_backend(
    *,
    timeout_seconds: int = 30,
    r_packages: list[str] | None = None,
) -> RBackend:
    """
    Build an rpy2 backend with optional R package checks.

    Args:
        timeout_seconds: Reserved timeout value for future R checks or execution.
        r_packages: Optional list of R packages that should be checked.

    Returns:
        Configured RBackend instance.
    """

    # Initialize the mandatory rpy2 requirement.
    requirements = [
        BackendRequirement(
            name="rpy2",
            requirement_type="python_package",
            required=True,
            install_hint="Install CellQuorum R extras with `pip install cellquorum[r]`.",
        )
    ]

    # Add optional R package requirements when provided.
    for package_name in r_packages or []:
        # Add each requested R package as a mandatory requirement.
        requirements.append(
            BackendRequirement(
                name=package_name,
                requirement_type="r_package",
                required=True,
                install_hint=f"Install the R package '{package_name}' in the active R library.",
            )
        )

    # Return the configured rpy2 backend.
    return RBackend(
        timeout_seconds=timeout_seconds,
        requirement_list=requirements,
    )