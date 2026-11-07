"""scCODA subprocess backend for CellQuorum.

scCODA (Bayesian compositional differential abundance) cannot be a direct
dependency of CellQuorum: the main env's scipy removed scipy.signal.gaussian
which scCODA pins, and pertpy is also broken here. The resolution is env
isolation — scCODA lives in its own micromamba environment (sccoda_env), and
CellQuorum invokes it as a subprocess, exactly the way the Rscript backend
calls R tools. This backend runs ``micromamba run -n <env> python <helper> <args>``
and exchanges data through temp files; it never imports sccoda into the
CellQuorum process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends.base import BackendRequirement
from cellquorum.backends.external_env import ExternalEnvBackend, launcher_requirement

# Directory holding the in-env helper scripts run INSIDE the sccoda environment.
_SCCODA_SCRIPTS_DIR = Path(__file__).parent / "sccoda_scripts"


@dataclass
class SccodaBackend(ExternalEnvBackend):
    """
    Backend for executing scCODA in an isolated micromamba environment.

    Availability requires (a) a micromamba/conda launcher on PATH and (b) that
    ``python -c "import sccoda"`` succeeds inside the configured environment. Both
    are checked lazily so the planner can report scCODA availability before a run.

    Args:
        name: Stable backend name.
        env_name: Name of the micromamba environment that has scCODA installed.
        launcher: Environment launcher executable (``micromamba``/``conda``/``mamba``).
        timeout_seconds: Timeout for the availability check.
        script_timeout_seconds: Timeout for helper-script execution.
    """

    name: str = "sccoda"

    env_name: str = "sccoda_env"

    script_timeout_seconds: int = 3600
    module_name = "sccoda"

    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            launcher_requirement(),
            BackendRequirement(
                name="sccoda",
                requirement_type="other",
                required=True,
                install_hint=(
                    "Create an isolated env with scCODA: "
                    "`micromamba create -n sccoda_env python=3.10 pip` then "
                    "`micromamba run -n sccoda_env pip install sccoda tensorflow`."
                ),
            ),
        ]
    )


def build_sccoda_backend(
    *,
    env_name: str = "sccoda_env",
    launcher: str = "micromamba",
    timeout_seconds: int = 60,
) -> SccodaBackend:
    """
    Build a scCODA subprocess backend.

    Args:
        env_name: Name of the isolated environment providing scCODA.
        launcher: Environment launcher executable.
        timeout_seconds: Availability-check timeout.

    Returns:
        Configured SccodaBackend instance.
    """

    return SccodaBackend(env_name=env_name, launcher=launcher, timeout_seconds=timeout_seconds)


# Path to the bundled scCODA fit helper, run inside scCODA env.
SCCODA_HELPER = _SCCODA_SCRIPTS_DIR / "sccoda_fit.py"


__all__ = ["SCCODA_HELPER", "SccodaBackend", "build_sccoda_backend"]
