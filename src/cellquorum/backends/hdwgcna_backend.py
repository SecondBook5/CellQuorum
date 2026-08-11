"""hdWGCNA subprocess backend for CellQuorum.

hdWGCNA (hierarchical Weighted Gene Co-expression Network Analysis) is an R
package for co-expression analysis that cannot be a direct dependency of
CellQuorum. The resolution is env isolation — hdWGCNA lives in its own
micromamba environment (hdwgcna_env), and CellQuorum invokes it as a
subprocess through Rscript. This backend runs
``micromamba run -n <env> Rscript <script> [args...]`` and exchanges data
through temporary files; it never imports hdWGCNA into the CellQuorum process.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends.base import BackendRequirement, BackendStatus, BaseBackend

# Directory holding the in-env helper scripts run INSIDE the hdwgcna environment.
_HDWGCNA_SCRIPTS_DIR = Path(__file__).parent / "r_scripts"


@dataclass
class HdwgcnaBackend(BaseBackend):
    """
    Backend for executing hdWGCNA in an isolated micromamba environment.

    Availability requires (a) a micromamba/conda launcher on PATH and (b) that
    the hdWGCNA R package is available inside the configured environment. Both
    are checked lazily so the planner can report hdWGCNA availability before a run.

    Args:
        name: Stable backend name.
        kind: Backend family.
        env_name: Name of the micromamba environment that has hdWGCNA installed.
        launcher: Environment launcher executable (``micromamba``/``conda``/``mamba``).
        timeout_seconds: Timeout for the availability check.
        script_timeout_seconds: Timeout for helper-script execution.
        requirement_list: Requirements checked by this backend.
    """

    # Store the stable backend name.
    name: str = "hdwgcna_r"

    # Store the backend family.
    kind: str = "external"

    # Store the isolated environment name that provides hdWGCNA.
    env_name: str = "hdwgcna_env"

    # Store the environment launcher used to enter the hdwgcna env.
    launcher: str = "micromamba"

    # Store the timeout used for the availability check.
    timeout_seconds: int = 60

    # Store the timeout used for helper-script execution.
    script_timeout_seconds: int = 3600

    # Store backend requirements.
    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            BackendRequirement(
                name="micromamba",
                requirement_type="executable",
                required=True,
                install_hint="Install micromamba (or set launcher to conda/mamba).",
            ),
            BackendRequirement(
                name="hdWGCNA",
                requirement_type="other",
                required=True,
                install_hint=(
                    "Create an isolated env with hdWGCNA: "
                    "`micromamba create -n hdwgcna_env -c conda-forge -c bioconda "
                    "r-seurat r-hdwgcna r-wgcna bioconductor-zellkonverter`."
                ),
            ),
        ]
    )

    def status(self) -> BackendStatus:
        """
        Check whether hdWGCNA can run in its isolated environment.

        Returns:
            BackendStatus describing hdWGCNA availability.
        """

        missing: list[str] = []
        warnings: list[str] = []
        details: dict[str, str | int | bool] = {
            "env_name": self.env_name,
            "launcher": self.launcher,
        }

        # The launcher must exist before we can enter the env at all.
        if not self._launcher_available():
            missing.append(self.launcher)
            return BackendStatus(
                name=self.name,
                kind=self.kind,
                available=False,
                requirements=self.requirements(),
                missing=missing,
                warnings=warnings,
                details=details,
            )

        # hdWGCNA itself must be available inside the configured env.
        if not self._r_package_available("hdWGCNA"):
            missing.append("hdWGCNA")

        available = len(missing) == 0
        return BackendStatus(
            name=self.name,
            kind=self.kind,
            available=available,
            requirements=self.requirements(),
            missing=missing,
            warnings=warnings,
            details=details,
        )

    def run_script(
        self,
        script_path: str | Path,
        args: list[str] | None = None,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """
        Run an in-env helper script inside the hdwgcna environment.

        Mirrors ``RscriptBackend.run_script``: the caller writes any input files
        and reads output files; this method only invokes
        ``<launcher> run -n <env> Rscript <script> [args...]`` and captures output.
        Non-zero exit codes are returned (not raised) so the caller can inspect
        stderr and raise a domain-specific error.

        Args:
            script_path: Path to the helper script (runs inside the hdwgcna env).
            args: Positional arguments passed to the helper.
            timeout: Optional timeout override; defaults to script_timeout_seconds.

        Returns:
            Completed subprocess result with captured stdout and stderr.

        Raises:
            FileNotFoundError: If the launcher or the helper script is missing.
        """

        script = Path(script_path)
        if not script.is_file():
            raise FileNotFoundError(f"hdwgcna helper script not found: {script}")

        if not self._launcher_available():
            raise FileNotFoundError(
                f"Environment launcher '{self.launcher}' was not found on PATH."
            )

        cmd = [
            self.launcher,
            "run",
            "-n",
            self.env_name,
            "Rscript",
            str(script),
            *(args or []),
        ]
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout if timeout is not None else self.script_timeout_seconds,
        )

    def _launcher_available(self) -> bool:
        """Return whether the environment launcher is on PATH."""

        return shutil.which(self.launcher) is not None

    def _r_package_available(self, package_name: str) -> bool:
        """
        Return whether an R package is available inside the configured env.

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

        try:
            result = subprocess.run(
                [
                    self.launcher,
                    "run",
                    "-n",
                    self.env_name,
                    "Rscript",
                    "-e",
                    f"quit(status=ifelse(requireNamespace('{package_name}',quietly=TRUE),0,1))",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
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


def build_hdwgcna_backend(
    *,
    env_name: str = "hdwgcna_env",
    launcher: str = "micromamba",
    timeout_seconds: int = 60,
) -> HdwgcnaBackend:
    """
    Build an hdWGCNA subprocess backend.

    Args:
        env_name: Name of the isolated environment providing hdWGCNA.
        launcher: Environment launcher executable.
        timeout_seconds: Availability-check timeout.

    Returns:
        Configured HdwgcnaBackend instance.
    """

    return HdwgcnaBackend(env_name=env_name, launcher=launcher, timeout_seconds=timeout_seconds)


# Path to the bundled hdWGCNA helper script, run inside hdwgcna env.
HDWGCNA_R = _HDWGCNA_SCRIPTS_DIR / "hdwgcna.R"


__all__ = ["HDWGCNA_R", "HdwgcnaBackend", "build_hdwgcna_backend"]
