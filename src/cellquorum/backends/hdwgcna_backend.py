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
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends._probe import resolve_env, resolve_launcher
from cellquorum.backends.base import BackendRequirement
from cellquorum.backends.external_env import ExternalEnvBackend, launcher_requirement

# Directory holding the in-env helper scripts run INSIDE the hdwgcna environment.
_HDWGCNA_SCRIPTS_DIR = Path(__file__).parent / "r_scripts"


@dataclass
class HdwgcnaBackend(ExternalEnvBackend):
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

    name: str = "hdwgcna_r"

    env_name: str = "hdwgcna_env"

    script_timeout_seconds: int = 3600
    module_name = "hdWGCNA"
    interpreter = "Rscript"

    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            launcher_requirement(),
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

    def _missing_in_env(self) -> list[str]:
        """hdWGCNA is an R package, so check it with R rather than a Python import."""

        return [] if self._r_package_available(self.module_name) else [self.module_name]

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
                    resolve_launcher(self.launcher) or self.launcher,
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

        return re.fullmatch(r"[A-Za-z][A-Za-z0-9._]*", package_name) is not None


def build_hdwgcna_backend(
    *,
    env_name: str | Sequence[str] = "hdwgcna_env",
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

    # Accept several acceptable environment names and pick the one that exists here.
    # Names are a fact about a machine: the container creates the canonical name, while a
    # given workstation may already have the same software under another. Resolving keeps one
    # analysis config correct in both places instead of pinning it to one filesystem.
    candidates = [env_name] if isinstance(env_name, str) else list(env_name)
    resolved = resolve_env(resolve_launcher(launcher) or launcher, candidates)
    return HdwgcnaBackend(
        env_name=resolved or candidates[0],
        launcher=launcher,
        timeout_seconds=timeout_seconds,
    )


# Path to the bundled hdWGCNA helper script, run inside hdwgcna env.
HDWGCNA_R = _HDWGCNA_SCRIPTS_DIR / "hdwgcna.R"


__all__ = ["HDWGCNA_R", "HdwgcnaBackend", "build_hdwgcna_backend"]
