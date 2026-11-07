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

from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends.base import BackendRequirement
from cellquorum.backends.external_env import ExternalEnvBackend, launcher_requirement

# Directory holding the in-env helper script run INSIDE celloracle_env.
_PYUCELL_SCRIPTS_DIR = Path(__file__).parent / "pyucell_scripts"


@dataclass
class PyUCellBackend(ExternalEnvBackend):
    """Backend for executing pyucell in an isolated micromamba environment.

    Defaults to the existing ``celloracle_env`` rather than a dedicated environment; see the
    module docstring for why. The launcher check, the import probe and the script runner are
    inherited from :class:`ExternalEnvBackend`.
    """

    name: str = "pyucell"
    env_name: str = "celloracle_env"
    module_name = "pyucell"

    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            launcher_requirement(),
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
