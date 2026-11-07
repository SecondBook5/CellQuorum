"""CellOracle subprocess backend for CellQuorum.

CellOracle (in-silico knockout for perturbation analysis) has specific dependency
requirements that make env isolation desirable. CellOracle lives in its own
micromamba environment (celloracle_env), and CellQuorum invokes it as a subprocess
through the in-env python. This backend runs
``micromamba run -n <env> python <script> [args...]`` and exchanges data through
files; it never imports CellOracle into the CellQuorum process.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

# Import the process-level probe cache (see _probe.py for why this exists: importing
# celloracle in a subprocess costs ~7.7s and the probe used to run uncached).
from cellquorum.backends.base import BackendRequirement
from cellquorum.backends.external_env import ExternalEnvBackend, launcher_requirement

# Directory holding the in-env helper scripts run INSIDE the celloracle environment.
_CELLORACLE_SCRIPTS_DIR = Path(__file__).parent / "celloracle_scripts"

# get_links() runs joblib.Parallel(n_jobs=-1) internally with no thread cap of its
# own; each worker process then makes its own threaded BLAS calls, and unconstrained
# threads-per-process times processes-per-core oversubscribes the machine. Confirmed
# live: get_links() on a real 61,608-cell object segfaulted (exit 139) with loky's
# resource_tracker reporting thousands of leaked semaphore/folder objects -- the
# signature of worker processes dying mid-fit under memory/thread pressure.
_THREAD_CAP_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass
class CellOracleBackend(ExternalEnvBackend):
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
    env_name: str = "celloracle_env"
    script_timeout_seconds: int = 10800
    module_name = "celloracle"

    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            launcher_requirement(),
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

    def _subprocess_env(self) -> Mapping[str, str] | None:
        """Cap the helper's math-library threads; CellOracle's helper otherwise oversubscribes."""

        return {**os.environ, **_THREAD_CAP_ENV}


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
