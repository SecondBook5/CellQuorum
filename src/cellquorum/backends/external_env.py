"""Shared base for backends that run a tool inside its own conda environment.

Some tools cannot live in CellQuorum's environment: they pin an older numpy or anndata, or ship
a different Python. Each of those gets an isolated environment, and CellQuorum runs it as a
subprocess, ``<launcher> run -n <env> python <script> [args...]``, exchanging data through files
and never importing the tool into its own process.

Eight backends did this and each carried its own copy of the same skeleton: find the launcher,
probe whether the tool imports inside the environment, build the command line, run it under a
timeout. This base holds that skeleton once, so a fix to launcher lookup or timeout handling is
made in one place. A backend now supplies only what is specific to it: its name, its default
environment, the module that proves the tool is installed, and its install hints.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from cellquorum.backends._probe import env_python_module_available, resolve_launcher
from cellquorum.backends.base import BackendKind, BackendRequirement, BackendStatus, BaseBackend

_MODULE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9._]*")


def launcher_requirement() -> BackendRequirement:
    """Return the requirement every external-environment backend shares: a launcher on the path."""

    return BackendRequirement(
        name="micromamba",
        requirement_type="executable",
        required=True,
        install_hint="Install micromamba (or set launcher to conda/mamba).",
    )


@dataclass
class ExternalEnvBackend(BaseBackend):
    """Run a helper script inside an isolated micromamba environment.

    Availability needs a launcher on the path and the tool's module importable inside the
    environment. Both checks are lazy, so the planner can report availability before a run.

    Subclasses set ``name``, ``env_name`` and ``requirement_list`` as dataclass defaults and the
    class variable ``module_name``.

    Args:
        name: Stable backend name.
        kind: Backend family.
        env_name: Name of the micromamba environment the tool is installed in.
        launcher: Environment launcher executable (micromamba, conda or mamba).
        timeout_seconds: Timeout for the availability probe.
        script_timeout_seconds: Default timeout for a helper script.
    """

    name: str = ""
    kind: BackendKind = "external"
    env_name: str = ""
    launcher: str = "micromamba"
    timeout_seconds: int = 60
    script_timeout_seconds: int = 3600

    # The module whose import proves the tool is usable inside the environment.
    module_name: ClassVar[str] = ""
    # The program that runs a helper script inside the environment.
    interpreter: ClassVar[str] = "python"

    def status(self) -> BackendStatus:
        """Check whether the tool can run in its isolated environment."""

        details: dict[str, str | int | bool] = {
            "env_name": self.env_name,
            "launcher": self.launcher,
        }
        if not self._launcher_available():
            return self._status(missing=[self.launcher], details=details)
        return self._status(missing=self._missing_in_env(), details=details)

    def run_script(
        self,
        script_path: str | Path,
        args: list[str] | None = None,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a helper script inside the environment and capture its output.

        Runs ``<launcher> run -n <env> <interpreter> <script> [args...]``. A non-zero exit code
        is returned, not raised, so the caller can read stderr and raise an error of its own.

        Args:
            script_path: Path to the helper script, which runs inside the environment.
            args: Positional arguments passed to the helper.
            timeout: Optional timeout override; defaults to ``script_timeout_seconds``.

        Returns:
            The completed process, with captured stdout and stderr.

        Raises:
            FileNotFoundError: If the launcher or the helper script is missing.
        """

        script = Path(script_path)
        if not script.is_file():
            raise FileNotFoundError(f"{self.name} helper script not found: {script}")
        if not self._launcher_available():
            raise FileNotFoundError(
                f"Environment launcher '{self.launcher}' was not found on PATH."
            )

        command = [
            resolve_launcher(self.launcher) or self.launcher,
            "run",
            "-n",
            self.env_name,
            self.interpreter,
            str(script),
            *(args or []),
        ]
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout if timeout is not None else self.script_timeout_seconds,
            env=self._subprocess_env(),
        )

    def _subprocess_env(self) -> Mapping[str, str] | None:
        """Return the environment variables for the helper process, or None to inherit ours."""

        return None

    def _missing_in_env(self) -> list[str]:
        """Return what is missing inside the environment, once the launcher is known to exist."""

        return [] if self._py_module_available(self.module_name) else [self.module_name]

    def _status(self, *, missing: list[str], details: dict[str, str | int | bool]) -> BackendStatus:
        return BackendStatus(
            name=self.name,
            kind=self.kind,
            available=not missing,
            requirements=self.requirements(),
            missing=missing,
            warnings=[],
            details=details,
        )

    def _launcher_available(self) -> bool:
        """Return whether the environment launcher is on the path."""

        return resolve_launcher(self.launcher) is not None

    def _py_module_available(self, module_name: str) -> bool:
        """Return whether a Python module imports inside the configured environment.

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
        """Validate a module name for safe use in a simple import expression."""

        return isinstance(module_name, str) and _MODULE_NAME.fullmatch(module_name) is not None


__all__ = ["ExternalEnvBackend", "launcher_requirement"]
