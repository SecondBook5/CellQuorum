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

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends._probe import resolve_env, resolve_launcher
from cellquorum.backends.base import BackendRequirement
from cellquorum.backends.external_env import ExternalEnvBackend, launcher_requirement

# Directory holding the in-env helper scripts run INSIDE the pyscenic environment.
_PYSCENIC_SCRIPTS_DIR = Path(__file__).parent / "pyscenic_scripts"


@dataclass
class PyscenicBackend(ExternalEnvBackend):
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
    env_name: str = "pyscenic_env"
    script_timeout_seconds: int = 7200
    module_name = "pyscenic"

    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            launcher_requirement(),
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


def build_pyscenic_backend(
    *,
    env_name: str | Sequence[str] = "pyscenic_env",
    launcher: str = "micromamba",
    timeout_seconds: int = 60,
) -> PyscenicBackend:
    """Build a pySCENIC subprocess backend."""

    # Accept several acceptable environment names and pick the one that exists here.
    # Names are a fact about a machine: the container creates the canonical name, while a
    # given workstation may already have the same software under another. Resolving keeps one
    # analysis config correct in both places instead of pinning it to one filesystem.
    candidates = [env_name] if isinstance(env_name, str) else list(env_name)
    resolved = resolve_env(resolve_launcher(launcher) or launcher, candidates)
    return PyscenicBackend(
        env_name=resolved or candidates[0],
        launcher=launcher,
        timeout_seconds=timeout_seconds,
    )


# Paths to the bundled in-env helper scripts, run inside pyscenic_env.
PYSCENIC_GRN_PY = _PYSCENIC_SCRIPTS_DIR / "pyscenic_grn.py"
PYSCENIC_AUCELL_PY = _PYSCENIC_SCRIPTS_DIR / "pyscenic_aucell.py"


__all__ = [
    "PYSCENIC_AUCELL_PY",
    "PYSCENIC_GRN_PY",
    "PyscenicBackend",
    "build_pyscenic_backend",
]
