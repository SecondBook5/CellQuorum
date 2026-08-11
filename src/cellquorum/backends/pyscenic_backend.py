"""pySCENIC subprocess backend for CellQuorum.

pySCENIC (classic; GRNBoost2 -> cisTarget -> AUCell) has a version-brittle
dependency stack (numpy 1.23.5, pandas 1.5.3, setuptools<81) that cannot be a
direct dependency of CellQuorum. The resolution is env isolation: pySCENIC lives
in its own micromamba environment (pyscenic_env), and CellQuorum invokes it as a
subprocess through the in-env python. This backend runs
``micromamba run -n <env> python <script> [args...]`` and exchanges data through
files; it never imports pySCENIC into the CellQuorum process.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends.base import BackendRequirement, BackendStatus, BaseBackend

# Directory holding the in-env helper scripts run INSIDE the pyscenic environment.
_PYSCENIC_SCRIPTS_DIR = Path(__file__).parent / "pyscenic_scripts"


@dataclass
class PyscenicBackend(BaseBackend):
    """Backend for executing pySCENIC in an isolated micromamba environment.

    Availability requires (a) a micromamba/conda launcher on PATH and (b) that the
    ``pyscenic`` Python module is importable inside the configured environment.
    Both are checked lazily so the planner can report availability before a run.

    Args:
        name: Stable backend name.
        kind: Backend family.
        env_name: Name of the micromamba environment that has pySCENIC installed.
        launcher: Environment launcher executable (micromamba/conda/mamba).
        timeout_seconds: Timeout for the availability check.
        script_timeout_seconds: Timeout for helper-script execution.
        requirement_list: Requirements checked by this backend.
    """

    name: str = "pyscenic"
    kind: str = "external"
    env_name: str = "pyscenic_env"
    launcher: str = "micromamba"
    timeout_seconds: int = 60
    script_timeout_seconds: int = 7200

    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            BackendRequirement(
                name="micromamba",
                requirement_type="executable",
                required=True,
                install_hint="Install micromamba (or set launcher to conda/mamba).",
            ),
            BackendRequirement(
                name="pyscenic",
                requirement_type="other",
                required=True,
                install_hint=(
                    "Create a frozen isolated env with classic pySCENIC: "
                    "`micromamba create -n pyscenic_env -c conda-forge -c bioconda "
                    "'python=3.10' 'numpy=1.23.5' 'pandas=1.5.3' 'setuptools<81' "
                    "pyscenic loompy`. Download cisTarget DBs (TFs, motifs, rankings) "
                    "separately and set grn.tfs_path / motifs_path / rankings_glob."
                ),
            ),
        ]
    )

    def status(self) -> BackendStatus:
        """Check whether pySCENIC can run in its isolated environment."""

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

        if not self._py_module_available("pyscenic"):
            missing.append("pyscenic")

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
        """Run an in-env helper script inside the pyscenic environment.

        Runs ``<launcher> run -n <env> python <script> [args...]`` and captures
        output. Non-zero exit codes are returned (not raised) so the caller can
        inspect stderr and raise a domain-specific error.

        Args:
            script_path: Path to the helper script (runs inside the pyscenic env).
            args: Positional arguments passed to the helper.
            timeout: Optional timeout override; defaults to script_timeout_seconds.

        Returns:
            Completed subprocess result with captured stdout and stderr.

        Raises:
            FileNotFoundError: If the launcher or the helper script is missing.
        """

        script = Path(script_path)
        if not script.is_file():
            raise FileNotFoundError(f"pyscenic helper script not found: {script}")

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

        try:
            result = subprocess.run(
                [
                    self.launcher,
                    "run",
                    "-n",
                    self.env_name,
                    "python",
                    "-c",
                    f"import {module_name}",
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
    def _valid_module_name(module_name: str) -> bool:
        """Validate a Python module name for safe use in a simple import expression."""

        if not isinstance(module_name, str):
            return False
        return re.fullmatch(r"[A-Za-z][A-Za-z0-9._]*", module_name) is not None


def build_pyscenic_backend(
    *,
    env_name: str = "pyscenic_env",
    launcher: str = "micromamba",
    timeout_seconds: int = 60,
) -> PyscenicBackend:
    """Build a pySCENIC subprocess backend."""

    return PyscenicBackend(env_name=env_name, launcher=launcher, timeout_seconds=timeout_seconds)


# Paths to the bundled in-env helper scripts, run inside pyscenic_env.
PYSCENIC_GRN_PY = _PYSCENIC_SCRIPTS_DIR / "pyscenic_grn.py"
PYSCENIC_AUCELL_PY = _PYSCENIC_SCRIPTS_DIR / "pyscenic_aucell.py"


__all__ = [
    "PYSCENIC_AUCELL_PY",
    "PYSCENIC_GRN_PY",
    "PyscenicBackend",
    "build_pyscenic_backend",
]
