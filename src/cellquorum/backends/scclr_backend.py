"""scclr subprocess backend for CellQuorum.

scclr (sparse PFlog1pPF normalization + sparse PCA, a Python shim over the Rust
crates runorm/rupca) cannot be a direct dependency of CellQuorum: it pins
``anndata<0.10.9`` while CellQuorum needs ``anndata>=0.11``, and its PyO3 build
caps at Python 3.13. The resolution is env isolation — scclr lives in its own
micromamba environment, and CellQuorum invokes it as a subprocess, exactly the
way the Rscript backend calls R tools. This backend runs
``micromamba run -n <env> python <helper> <args>`` and exchanges data through
temp files; it never imports scclr into the CellQuorum process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends.base import BackendRequirement
from cellquorum.backends.external_env import ExternalEnvBackend, launcher_requirement

# Directory holding the in-env helper scripts run INSIDE the scclr environment.
_SCCLR_SCRIPTS_DIR = Path(__file__).parent / "scclr_scripts"


@dataclass
class ScclrBackend(ExternalEnvBackend):
    """
    Backend for executing scclr in an isolated micromamba environment.

    Availability requires (a) a micromamba/conda launcher on PATH and (b) that
    ``python -c "import scclr"`` succeeds inside the configured environment. Both
    are checked lazily so the planner can report scclr availability before a run.

    Args:
        name: Stable backend name.
        env_name: Name of the micromamba environment that has scclr installed.
        launcher: Environment launcher executable (``micromamba``/``conda``/``mamba``).
        timeout_seconds: Timeout for the availability check.
        script_timeout_seconds: Timeout for helper-script execution.
    """

    name: str = "scclr"

    env_name: str = "scclr"

    script_timeout_seconds: int = 1800
    module_name = "scclr"

    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            launcher_requirement(),
            BackendRequirement(
                name="scclr",
                requirement_type="other",
                required=True,
                install_hint=(
                    "Create an isolated env with scclr: "
                    "`micromamba create -n scclr python=3.13 rust maturin pip` then "
                    "`micromamba run -n scclr pip install -e /path/to/scclr`."
                ),
            ),
        ]
    )


def build_scclr_backend(
    *,
    env_name: str = "scclr",
    launcher: str = "micromamba",
    timeout_seconds: int = 60,
) -> ScclrBackend:
    """
    Build a scclr subprocess backend.

    Args:
        env_name: Name of the isolated environment providing scclr.
        launcher: Environment launcher executable.
        timeout_seconds: Availability-check timeout.

    Returns:
        Configured ScclrBackend instance.
    """

    return ScclrBackend(env_name=env_name, launcher=launcher, timeout_seconds=timeout_seconds)


# Path to the bundled PFlog helper (normalize + pca modes), run inside scclr env.
PFLOG_HELPER = _SCCLR_SCRIPTS_DIR / "pflog.py"


__all__ = ["PFLOG_HELPER", "ScclrBackend", "build_scclr_backend"]
