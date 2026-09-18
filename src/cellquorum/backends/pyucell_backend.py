"""PyUCell subprocess backend for CellQuorum.

pyucell (per-cell gene-signature scoring, designed for small/sparse gene sets
where AUCell's rank-recovery-curve AUC goes noisy) hard-pins ``numpy<2`` on
PyPI as of its latest release (0.7.3) -- installing it directly into
CellQuorum's own environment would force a numpy downgrade that breaks
cellrank/pertpy/pydeseq2/shap and others that require numpy>=2 (confirmed
live, then reverted).

Rather than create yet another dedicated isolated environment, this
deliberately reuses ``celloracle_env``: it already ships numpy 1.26.4
(satisfies pyucell's ``numpy<2,>=1.23``) and scanpy 1.11.0 (satisfies
``scanpy>=1.9``), so `pip install pyucell` there added nothing else -- a
verified dry-run showed every other dependency already present.
CellQuorum invokes it as a subprocess through the in-env python, exactly
like the CellOracle backend: ``micromamba run -n celloracle_env python
<script> [args...]``, exchanging data through files, never importing
pyucell into the CellQuorum process itself.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends._probe import env_python_module_available, resolve_launcher
from cellquorum.backends.base import BackendRequirement, BackendStatus, BaseBackend

# Directory holding the in-env helper script run INSIDE celloracle_env.
_PYUCELL_SCRIPTS_DIR = Path(__file__).parent / "pyucell_scripts"


@dataclass
class PyUCellBackend(BaseBackend):
    """Backend for executing pyucell in an isolated micromamba environment.

    Availability requires (a) a micromamba/conda launcher on PATH and (b) that
    the ``pyucell`` Python module is importable inside the configured
    environment. Both are checked lazily so the planner can report
    availability before a run.

    Args:
        name: Stable backend name.
        kind: Backend family.
        env_name: Name of the micromamba environment that has pyucell
            installed. Defaults to the existing ``celloracle_env`` rather
            than a dedicated env -- see the module docstring for why.
        launcher: Environment launcher executable (micromamba/conda/mamba).
        timeout_seconds: Timeout for the availability check.
        script_timeout_seconds: Timeout for helper-script execution.
        requirement_list: Requirements checked by this backend.
    """

    name: str = "pyucell"
    kind: str = "external"
    env_name: str = "celloracle_env"
    launcher: str = "micromamba"
    timeout_seconds: int = 60
    script_timeout_seconds: int = 3600

    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            BackendRequirement(
                name="micromamba",
                requirement_type="executable",
                required=True,
                install_hint="Install micromamba (or set launcher to conda/mamba).",
            ),
            BackendRequirement(
                name="pyucell",
                requirement_type="other",
                required=True,
                install_hint=(
                    "Install pyucell into the existing celloracle_env "
                    "(numpy<2 already satisfied there): "
                    "`micromamba run -n celloracle_env pip install pyucell`."
                ),
            ),
        ]
    )

    def status(self) -> BackendStatus:
        """Check whether pyucell can run in its isolated environment."""

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

        if not self._py_module_available("pyucell"):
            missing.append("pyucell")

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
        """Run an in-env helper script inside the pyucell (celloracle_env) environment.

        Runs ``<launcher> run -n <env> python <script> [args...]`` and captures
        output. Non-zero exit codes are returned (not raised) so the caller can
        inspect stderr and raise a domain-specific error.

        Args:
            script_path: Path to the helper script (runs inside the env).
            args: Positional arguments passed to the helper.
            timeout: Optional timeout override; defaults to script_timeout_seconds.

        Returns:
            Completed subprocess result with captured stdout and stderr.

        Raises:
            FileNotFoundError: If the launcher or the helper script is missing.
        """

        script = Path(script_path)
        if not script.is_file():
            raise FileNotFoundError(f"pyucell helper script not found: {script}")

        if not self._launcher_available():
            raise FileNotFoundError(
                f"Environment launcher '{self.launcher}' was not found on PATH."
            )

        cmd = [
            resolve_launcher(self.launcher) or self.launcher,
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

        return resolve_launcher(self.launcher) is not None

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

        return env_python_module_available(
            resolve_launcher(self.launcher) or self.launcher,
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


def build_pyucell_backend(
    *,
    env_name: str = "celloracle_env",
    launcher: str = "micromamba",
    timeout_seconds: int = 60,
) -> PyUCellBackend:
    """Build a pyucell subprocess backend."""

    return PyUCellBackend(env_name=env_name, launcher=launcher, timeout_seconds=timeout_seconds)


# Path to the bundled in-env helper script, run inside celloracle_env.
PYUCELL_SCORE_PY = _PYUCELL_SCRIPTS_DIR / "ucell_score.py"


__all__ = [
    "PYUCELL_SCORE_PY",
    "PyUCellBackend",
    "build_pyucell_backend",
]
