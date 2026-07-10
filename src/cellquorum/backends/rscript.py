"""Rscript backend definitions for CellQuorum."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends.base import BackendRequirement, BackendStatus, BaseBackend


@dataclass
class RscriptBackend(BaseBackend):
    """
    Backend for executing R/Bioconductor methods through Rscript.

    CellQuorum should support R in two ways: in-process execution through rpy2
    and batch/HPC-safe execution through Rscript. This backend handles the
    Rscript fallback path. It checks whether the Rscript executable is available
    and can optionally verify required R packages by calling `requireNamespace`
    inside a short Rscript process.

    Args:
        name: Stable backend name.
        rscript_path: Executable name or path used to call Rscript.
        timeout_seconds: Maximum time allowed for backend availability checks.
        requirement_list: Requirements checked by this backend.
    """

    # Store the stable backend name.
    name: str = "rscript"

    # Store the backend family.
    kind: str = "rscript"

    # Store the Rscript executable name or path.
    rscript_path: str = "Rscript"

    # Store the timeout used for availability checks.
    timeout_seconds: int = 30

    # Store the timeout used for script execution (separate from availability checks).
    script_timeout_seconds: int = 600

    # Store Rscript backend requirements.
    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            BackendRequirement(
                name="Rscript",
                requirement_type="executable",
                required=True,
                install_hint="Install R and ensure Rscript is available on PATH.",
            )
        ]
    )

    def status(self) -> BackendStatus:
        """
        Check whether the Rscript backend is available.

        This method checks the configured Rscript executable and any declared R
        package requirements. It returns structured status metadata instead of
        raising immediately so the planner can explain what is available before
        a pipeline run begins.

        Returns:
            BackendStatus describing Rscript availability.
        """

        # Initialize the missing requirement list.
        missing: list[str] = []

        # Initialize the warning list.
        warnings: list[str] = []

        # Initialize additional backend details.
        details: dict[str, str | int | bool] = {
            "rscript_path": self.rscript_path,
            "timeout_seconds": self.timeout_seconds,
        }

        # Check whether the configured Rscript executable exists.
        if not self._rscript_available():
            # Record the missing executable as a mandatory missing requirement.
            missing.append(self.rscript_path)

            # Return immediately because R package checks require Rscript.
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
            # Skip the executable requirement because it was already checked above.
            if requirement.requirement_type == "executable":
                continue

            # Check declared R package requirements.
            if requirement.requirement_type == "r_package":
                # Check whether the requested R package can be loaded.
                package_available = self._r_package_available(requirement.name)

                # Record missing mandatory R packages.
                if not package_available and requirement.required:
                    missing.append(requirement.name)

            # Warn if a Python package is accidentally attached to this backend.
            elif requirement.requirement_type == "python_package":
                # Store a warning because Python packages belong to Python backends.
                warnings.append(
                    f"Requirement '{requirement.name}' is a Python package and should be checked "
                    "by a Python backend."
                )

            # Warn if CUDA is accidentally attached to this backend.
            elif requirement.requirement_type == "cuda":
                # Store a warning because CUDA checks belong to GPU backends.
                warnings.append(
                    f"Requirement '{requirement.name}' is CUDA-related and should be checked "
                    "by a GPU backend."
                )

            # Warn on unsupported requirement types.
            else:
                # Store a warning for unsupported requirement categories.
                warnings.append(
                    f"Requirement '{requirement.name}' has unsupported type "
                    f"'{requirement.requirement_type}' for RscriptBackend."
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

    def run_expression(self, expression: str) -> subprocess.CompletedProcess[str]:
        """
        Execute a short R expression through Rscript.

        This helper is intended for lightweight backend checks and later small
        adapter calls. Long-running R methods should use dedicated wrapper
        functions that write input/output files explicitly and preserve logs.

        Args:
            expression: R expression passed to `Rscript --vanilla -e`.

        Returns:
            Completed subprocess result with captured stdout and stderr.

        Raises:
            TypeError: If expression is not a string.
            ValueError: If expression is empty.
            FileNotFoundError: If Rscript is unavailable.
            subprocess.TimeoutExpired: If Rscript exceeds the timeout.
        """

        # Validate that the expression is a string.
        if not isinstance(expression, str):
            raise TypeError(
                "run_expression expected a string R expression. "
                f"Received: {type(expression).__name__}"
            )

        # Validate that the expression is not empty.
        if not expression.strip():
            raise ValueError("run_expression expected a non-empty R expression.")

        # Raise a clear error if Rscript cannot be found.
        if not self._rscript_available():
            raise FileNotFoundError(
                f"Rscript executable '{self.rscript_path}' was not found on PATH."
            )

        # Execute Rscript with captured text output.
        return subprocess.run(
            [self.rscript_path, "--vanilla", "-e", expression],
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )

    def run_script(
        self,
        script_path: str | Path,
        args: list[str] | None = None,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """
        Execute an R script file through Rscript with arguments.

        This is the generic primitive for R-backed methods (scDblFinder, SoupX)
        that need to run a real script and exchange files with it. The caller is
        responsible for writing any input files and reading output files; this
        method only runs ``Rscript --vanilla <script> [args...]`` and captures
        output. Non-zero exit codes are returned (not raised) so the caller can
        inspect stderr and produce a domain-specific error.

        Args:
            script_path: Path to the R script to execute.
            args: Positional arguments passed to the script (via commandArgs).
            timeout: Optional timeout override; defaults to script_timeout_seconds.

        Returns:
            Completed subprocess result with captured stdout and stderr.

        Raises:
            FileNotFoundError: If Rscript or the script file is unavailable.
        """

        # Normalize the script path and confirm it exists.
        script = Path(script_path)
        if not script.is_file():
            raise FileNotFoundError(f"R script not found: {script}")

        # Raise a clear error if Rscript cannot be found.
        if not self._rscript_available():
            raise FileNotFoundError(
                f"Rscript executable '{self.rscript_path}' was not found on PATH."
            )

        # Build the argument list and run with a long (script) timeout.
        cmd = [self.rscript_path, "--vanilla", str(script), *(args or [])]
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout if timeout is not None else self.script_timeout_seconds,
        )

    def _rscript_available(self) -> bool:
        """
        Check whether the configured Rscript executable is available.

        Returns:
            True if Rscript is available, otherwise False.
        """

        # Return whether the configured executable can be found on PATH.
        return shutil.which(self.rscript_path) is not None

    def _r_package_available(self, package_name: str) -> bool:
        """
        Check whether an R package is available through Rscript.

        The package name is validated before being inserted into the R expression
        so malformed names cannot produce confusing Rscript calls.

        Args:
            package_name: R package name to check.

        Returns:
            True if the R package is available, otherwise False.

        Raises:
            ValueError: If the R package name contains unsupported characters.
        """

        # Validate the R package name before building an R expression.
        if not self._valid_r_package_name(package_name):
            raise ValueError(
                "Invalid R package name. Expected letters, numbers, dots, or underscores "
                f"starting with a letter. Received: {package_name}"
            )

        # Build a small R expression that exits 0 when the package is available.
        expression = (
            "quit(status = ifelse(" f"requireNamespace('{package_name}', quietly = TRUE), 0, 1" "))"
        )

        # Run the R package availability expression.
        result = self.run_expression(expression)

        # Return whether Rscript reported success.
        return result.returncode == 0

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


def build_rscript_backend(
    *,
    rscript_path: str = "Rscript",
    timeout_seconds: int = 30,
    r_packages: list[str] | None = None,
) -> RscriptBackend:
    """
    Build an Rscript backend with optional R package checks.

    Args:
        rscript_path: Executable name or path used to call Rscript.
        timeout_seconds: Maximum time allowed for Rscript checks.
        r_packages: Optional list of R packages that should be checked.

    Returns:
        Configured RscriptBackend instance.
    """

    # Initialize the mandatory Rscript executable requirement.
    requirements = [
        BackendRequirement(
            name=rscript_path,
            requirement_type="executable",
            required=True,
            install_hint="Install R and ensure Rscript is available on PATH.",
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

    # Return the configured Rscript backend.
    return RscriptBackend(
        rscript_path=rscript_path,
        timeout_seconds=timeout_seconds,
        requirement_list=requirements,
    )
