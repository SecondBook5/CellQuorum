"""partipy subprocess backend for CellQuorum: archetypal analysis in an isolated env.

partipy (Pareto Task Inference, saezlab) fits a polytope to the data and returns its vertices
as *archetypes* — extreme phenotypes rather than dense clusters. That distinction is why it is
here at all: Leiden needs density, so a fifty-cell population is exactly what it merges away,
while a vertex does not care how few cells support it. Finding rare populations is the one job
clustering is structurally bad at.

## Why a subprocess and not a dependency

**Licensing.** partipy is GPL-3 and CellQuorum is BSD-3. Importing it into this process would
make the combined distribution copyleft, which would block the PyPI and conda-forge
distribution this package is aimed at. Running it as a separate process with data crossing as
files is not a workaround for a technicality — it is the difference between a derived work and
two programs exchanging files, and it keeps CellQuorum redistributable under BSD-3.

It also happens to be the right engineering call for the usual reason, the one scclr is here
for: partipy brings its own dependency stack, and env isolation stops that stack from
constraining CellQuorum's.

## What archetypes can and cannot decide

They cannot decide damage. On the severity axes a rare cell type and a dying cell are
geometrically identical — both are extreme — so a vertex is evidence of *coherent
extremeness*, not of health. What archetypes are good for is the question QC cannot answer
about itself: **is there a coherent population here that QC is quietly removing?** That is an
audit, and it is optional. Without this backend the audit is unavailable and nothing else
changes.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends.base import BackendRequirement, BackendStatus, BaseBackend

# Directory holding the in-env helper scripts run INSIDE the partipy environment.
_PARTIPY_SCRIPTS_DIR = Path(__file__).parent / "partipy_scripts"


@dataclass
class PartipyBackend(BaseBackend):
    """Backend for executing partipy in an isolated micromamba environment.

    Availability requires a micromamba/conda launcher on PATH and that
    ``python -c "import partipy"`` succeeds inside the configured environment. Both are
    checked lazily so the planner can report availability before a run starts.

    Args:
        name: Stable backend name.
        env_name: Name of the micromamba environment that has partipy installed.
        launcher: Environment launcher executable (``micromamba``/``conda``/``mamba``).
        timeout_seconds: Timeout for the availability check.
        script_timeout_seconds: Timeout for helper-script execution.
    """

    name: str = "partipy"
    kind: str = "external"
    env_name: str = "partipy"
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
                name="partipy",
                requirement_type="other",
                required=True,
                install_hint=(
                    "Create an isolated env with partipy: "
                    "`micromamba create -n partipy python=3.12 pip` then "
                    "`micromamba run -n partipy pip install partipy`. Kept isolated because "
                    "partipy is GPL-3 and CellQuorum is BSD-3."
                ),
            ),
        ]
    )

    def status(self) -> BackendStatus:
        """Check whether partipy can run in its isolated environment."""
        missing: list[str] = []
        details: dict[str, str | int | bool] = {
            "env_name": self.env_name,
            "launcher": self.launcher,
            "license": "GPL-3 (isolated; not linked into CellQuorum)",
        }

        if not self._launcher_available():
            missing.append(self.launcher)
            return BackendStatus(
                name=self.name,
                kind=self.kind,
                available=False,
                requirements=self.requirements(),
                missing=missing,
                warnings=[],
                details=details,
            )

        if not self._partipy_importable():
            missing.append("partipy")

        return BackendStatus(
            name=self.name,
            kind=self.kind,
            available=len(missing) == 0,
            requirements=self.requirements(),
            missing=missing,
            warnings=[],
            details=details,
        )

    def run_helper(
        self,
        script_path: str | Path,
        args: list[str] | None = None,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run an in-env helper script inside the partipy environment.

        Mirrors :meth:`cellquorum.backends.scclr_backend.ScclrBackend.run_helper`: the caller
        writes input files and reads output files, and a non-zero exit code is returned rather
        than raised so the caller can surface a domain-specific error.

        Args:
            script_path: Path to the helper script (runs inside the partipy env).
            args: Positional arguments passed to the helper.
            timeout: Optional timeout override; defaults to ``script_timeout_seconds``.

        Returns:
            Completed subprocess result with captured stdout and stderr.

        Raises:
            FileNotFoundError: If the launcher or the helper script is missing.
        """
        script = Path(script_path)
        if not script.is_file():
            raise FileNotFoundError(f"partipy helper script not found: {script}")
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

    def _partipy_importable(self) -> bool:
        """Return whether ``import partipy`` succeeds inside the configured env."""
        try:
            result = subprocess.run(
                [self.launcher, "run", "-n", self.env_name, "python", "-c", "import partipy"],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0


def build_partipy_backend(
    *,
    env_name: str = "partipy",
    launcher: str = "micromamba",
    timeout_seconds: int = 60,
) -> PartipyBackend:
    """Build a partipy subprocess backend.

    Args:
        env_name: Name of the isolated environment providing partipy.
        launcher: Environment launcher executable.
        timeout_seconds: Availability-check timeout.

    Returns:
        Configured PartipyBackend instance.
    """
    return PartipyBackend(env_name=env_name, launcher=launcher, timeout_seconds=timeout_seconds)


#: Path to the bundled archetype helper, run inside the partipy env.
ARCHETYPE_HELPER = _PARTIPY_SCRIPTS_DIR / "archetypes.py"


__all__ = ["ARCHETYPE_HELPER", "PartipyBackend", "build_partipy_backend"]
