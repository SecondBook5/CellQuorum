"""scclr subprocess backend for CellQuorum.

scclr (sparse PFlog1pPF normalization + sparse PCA, a Python shim over the Rust
crates runorm/rupca) cannot be a direct dependency of CellQuorum: it pins
``anndata<0.10.9`` while CellQuorum needs ``anndata>=0.11``, and its PyO3 build
caps at Python 3.13. The resolution is env isolation — scclr lives in its own
micromamba environment, and CellQuorum invokes it as a subprocess, exactly the
way the Rscript backend calls R tools. This backend runs
``micromamba run -n <env> python <helper> <args>`` and exchanges data through
temp files; it never imports scclr into the CellQuorum process.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends.base import BackendRequirement, BackendStatus, BaseBackend

# Directory holding the in-env helper scripts run INSIDE the scclr environment.
_SCCLR_SCRIPTS_DIR = Path(__file__).parent / "scclr_scripts"


@dataclass
class ScclrBackend(BaseBackend):
    """
    Backend for executing scclr in an isolated micromamba environment.

    Availability requires (a) a micromamba/conda launcher on PATH and (b) that
    ``python -c "import scclr"`` succeeds inside the configured environment. Both
    are checked lazily so the planner can report scclr availability before a run.

    Args:
        name: Stable backend name.
        env_name: Name of the micromamba environment that has scclr installed.
        launcher: Environment launcher executable (``micromamba``/``conda``/``mamba``).
        timeout_seconds: Timeout for the availability check.
        script_timeout_seconds: Timeout for helper-script execution.
    """

    # Store the stable backend name.
    name: str = "scclr"

    # Store the backend family.
    kind: str = "external"

    # Store the isolated environment name that provides scclr.
    env_name: str = "scclr"

    # Store the environment launcher used to enter the scclr env.
    launcher: str = "micromamba"

    # Store the timeout used for the availability check.
    timeout_seconds: int = 60

    # Store the timeout used for helper-script execution.
    script_timeout_seconds: int = 1800

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
                name="scclr",
                requirement_type="other",
                required=True,
                install_hint=(
                    "Create an isolated env with scclr: "
                    "`micromamba create -n scclr python=3.13 rust maturin pip` then "
                    "`micromamba run -n scclr pip install -e /path/to/scclr`."
                ),
            ),
        ]
    )

    def status(self) -> BackendStatus:
        """
        Check whether scclr can run in its isolated environment.

        Returns:
            BackendStatus describing scclr availability.
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

        # scclr itself must import inside the configured env.
        if not self._scclr_importable():
            missing.append("scclr")

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

    def run_helper(
        self,
        script_path: str | Path,
        args: list[str] | None = None,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """
        Run an in-env helper script inside the scclr environment.

        Mirrors ``RscriptBackend.run_script``: the caller writes any input files
        and reads output files; this method only invokes
        ``<launcher> run -n <env> python <script> [args...]`` and captures output.
        Non-zero exit codes are returned (not raised) so the caller can inspect
        stderr and raise a domain-specific error.

        Args:
            script_path: Path to the helper script (runs inside the scclr env).
            args: Positional arguments passed to the helper.
            timeout: Optional timeout override; defaults to script_timeout_seconds.

        Returns:
            Completed subprocess result with captured stdout and stderr.

        Raises:
            FileNotFoundError: If the launcher or the helper script is missing.
        """

        script = Path(script_path)
        if not script.is_file():
            raise FileNotFoundError(f"scclr helper script not found: {script}")

        if not self._launcher_available():
            raise FileNotFoundError(
                f"Environment launcher '{self.launcher}' was not found on PATH."
            )

        cmd = [
            self.launcher,
            "run",
            "-n",
            self.env_name,
            "python",
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

    def _scclr_importable(self) -> bool:
        """Return whether ``import scclr`` succeeds inside the configured env."""

        try:
            result = subprocess.run(
                [self.launcher, "run", "-n", self.env_name, "python", "-c", "import scclr"],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0


def build_scclr_backend(
    *,
    env_name: str = "scclr",
    launcher: str = "micromamba",
    timeout_seconds: int = 60,
) -> ScclrBackend:
    """
    Build a scclr subprocess backend.

    Args:
        env_name: Name of the isolated environment providing scclr.
        launcher: Environment launcher executable.
        timeout_seconds: Availability-check timeout.

    Returns:
        Configured ScclrBackend instance.
    """

    return ScclrBackend(env_name=env_name, launcher=launcher, timeout_seconds=timeout_seconds)


# Path to the bundled PFlog helper (normalize + pca modes), run inside scclr env.
PFLOG_HELPER = _SCCLR_SCRIPTS_DIR / "pflog.py"


__all__ = ["PFLOG_HELPER", "ScclrBackend", "build_scclr_backend"]
