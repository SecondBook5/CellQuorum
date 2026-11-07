"""MetaCell2 subprocess backend for CellQuorum.

MetaCell2 (the ``metacells`` PyPI package, Baran & Tanay) has its own C-extension
build and pinning story that cannot be a direct dependency of CellQuorum. The
resolution is env isolation, identical in shape to pySCENIC: ``metacells`` lives in
its own micromamba environment (``metacells_env``), and CellQuorum invokes it as a
subprocess through the in-env python. This backend runs
``micromamba run -n <env> python <script> [args...]`` and exchanges data through
files; it never imports ``metacells`` into the CellQuorum process.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends._probe import resolve_env, resolve_launcher
from cellquorum.backends.base import BackendRequirement
from cellquorum.backends.external_env import ExternalEnvBackend, launcher_requirement

# Directory holding the in-env helper script run INSIDE the metacells environment.
_METACELLS_SCRIPTS_DIR = Path(__file__).parent / "metacells_scripts"


@dataclass
class MetacellsBackend(ExternalEnvBackend):
    """Backend for executing MetaCell2 in an isolated micromamba environment.

    Availability requires (a) a micromamba/conda launcher on PATH and (b) that the
    ``metacells`` Python module is importable inside the configured environment.
    Both are checked lazily so the planner can report availability before a run.

    Args:
        name: Stable backend name.
        kind: Backend family.
        env_name: Name of the micromamba environment that has ``metacells`` installed.
        launcher: Environment launcher executable (micromamba/conda/mamba).
        timeout_seconds: Timeout for the availability check.
        script_timeout_seconds: Timeout for helper-script execution.
        requirement_list: Requirements checked by this backend.
    """

    name: str = "metacells"
    env_name: str = "metacells_env"
    script_timeout_seconds: int = 3600
    module_name = "metacells"

    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            launcher_requirement(),
            BackendRequirement(
                name="metacells",
                requirement_type="other",
                required=True,
                install_hint=(
                    "Create an isolated env with MetaCell2: "
                    "`micromamba create -n metacells_env -c conda-forge -c bioconda "
                    "'python=3.11' metacells`."
                ),
            ),
        ]
    )


def build_metacells_backend(
    *,
    env_name: str | Sequence[str] = "metacells_env",
    launcher: str = "micromamba",
    timeout_seconds: int = 60,
) -> MetacellsBackend:
    """Build a MetaCell2 subprocess backend."""

    candidates = [env_name] if isinstance(env_name, str) else list(env_name)
    resolved = resolve_env(resolve_launcher(launcher) or launcher, candidates)
    return MetacellsBackend(
        env_name=resolved or candidates[0],
        launcher=launcher,
        timeout_seconds=timeout_seconds,
    )


# Path to the bundled in-env helper script, run inside metacells_env.
RUN_METACELL2_PY = _METACELLS_SCRIPTS_DIR / "run_metacell2.py"


__all__ = [
    "RUN_METACELL2_PY",
    "MetacellsBackend",
    "build_metacells_backend",
]
