"""CellOracle subprocess backend for CellQuorum.

CellOracle (in-silico knockout for perturbation analysis) has specific dependency
requirements that make env isolation desirable. CellOracle lives in its own
micromamba environment (celloracle_env), and CellQuorum invokes it as a subprocess
through the in-env python. This backend runs
``micromamba run -n <env> python <script> [args...]`` and exchanges data through
files; it never imports CellOracle into the CellQuorum process.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Import the process-level probe cache (see _probe.py for why this exists: importing
# celloracle in a subprocess costs ~7.7s and the probe used to run uncached).
from cellquorum.backends._probe import env_python_module_available
from cellquorum.backends.base import BackendRequirement, BackendStatus, BaseBackend

# Directory holding the in-env helper scripts run INSIDE the celloracle environment.
_CELLORACLE_SCRIPTS_DIR = Path(__file__).parent / "celloracle_scripts"


@dataclass
class CellOracleBackend(BaseBackend):
    """Backend for executing CellOracle in an isolated micromamba environment.

    Availability requires (a) a micromamba/conda launcher on PATH and (b) that the
    ``celloracle`` Python module is importable inside the configured environment.
    Both are checked lazily so the planner can report availability before a run.

    Args:
        name: Stable backend name.
        kind: Backend family.
        env_name: Name of the micromamba environment that has CellOracle installed.
        launcher: Environment launcher executable (micromamba/conda/mamba).
        timeout_seconds: Timeout for the availability check.
        script_timeout_seconds: Timeout for helper-script execution.
        requirement_list: Requirements checked by this backend.
    """

    name: str = "celloracle"
    kind: str = "external"
    env_name: str = "celloracle_env"
    launcher: str = "micromamba"
    timeout_seconds: int = 60
    script_timeout_seconds: int = 10800

    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            BackendRequirement(
                name="micromamba",
                requirement_type="executable",
                required=True,
                install_hint="Install micromamba (or set launcher to conda/mamba).",
            ),
            BackendRequirement(
                name="celloracle",
                requirement_type="other",
                required=True,
                install_hint=(
                    "Create a frozen isolated env with CellOracle: "
                    "`micromamba create -n celloracle_env -c conda-forge celloracle`. "
                    "The promoter base GRN (hg38/mm10) ships with CellOracle."
                ),
            ),
        ]
    )

    def status(self) -> BackendStatus:
        """Check whether CellOracle can run in its isolated environment."""

        missing: list[str] = []
        warnings: list[str] = []
        details: dict[str, str | int | bool] = {
            "env_name": self.env_name,
            "launcher": self.launcher,
        }

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

        if not self._py_module_available("celloracle"):
            missing.append("celloracle")

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
        """Run an in-env helper script inside the celloracle environment.

        Runs ``<launcher> run -n <env> python <script> [args...]`` and captures
        output. Non-zero exit codes are returned (not raised) so the caller can
        inspect stderr and raise a domain-specific error.

        Args:
            script_path: Path to the helper script (runs inside the celloracle env).
            args: Positional arguments passed to the helper.
            timeout: Optional timeout override; defaults to script_timeout_seconds.

        Returns:
            Completed subprocess result with captured stdout and stderr.

        Raises:
            FileNotFoundError: If the launcher or the helper script is missing.
        """

        script = Path(script_path)
        if not script.is_file():
            raise FileNotFoundError(f"celloracle helper script not found: {script}")

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

    def _py_module_available(self, module_name: str) -> bool:
        """Return whether a Python module is importable inside the configured env.

        Args:
            module_name: Python module name to probe.

        Returns:
            True if the module imports, otherwise False.

        Raises:
            ValueError: If the module name contains unsupported characters.
        """

        if not self._valid_module_name(module_name):
            raise ValueError(
                "Invalid Python module name. Expected letters, numbers, dots, or "
                f"underscores starting with a letter. Received: {module_name}"
            )

        # Delegate to the process-level cache. Importing celloracle in a subprocess
        # costs ~7.7s, and this probe runs once per backend status table — which the
        # planner, the CLI, and most planner tests each build at least once. Validation
        # stays above so an invalid name still raises before anything is cached.
        return env_python_module_available(
            self.launcher,
            self.env_name,
            module_name,
            self.timeout_seconds,
        )

    @staticmethod
    def _valid_module_name(module_name: str) -> bool:
        """Validate a Python module name for safe use in a simple import expression."""

        if not isinstance(module_name, str):
            return False
        return re.fullmatch(r"[A-Za-z][A-Za-z0-9._]*", module_name) is not None


def build_celloracle_backend(
    *,
    env_name: str = "celloracle_env",
    launcher: str = "micromamba",
    timeout_seconds: int = 60,
) -> CellOracleBackend:
    """Build a CellOracle subprocess backend."""

    return CellOracleBackend(env_name=env_name, launcher=launcher, timeout_seconds=timeout_seconds)


# Paths to the bundled in-env helper scripts, run inside celloracle_env.
CELLORACLE_KO_PY = _CELLORACLE_SCRIPTS_DIR / "celloracle_ko.py"


__all__ = [
    "CELLORACLE_KO_PY",
    "CellOracleBackend",
    "build_celloracle_backend",
]
