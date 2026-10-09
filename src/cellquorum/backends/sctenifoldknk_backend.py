"""Isolated Python execution for the optional scTenifoldpy knockout method."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from cellquorum.backends.base import BackendKind, BackendRequirement, BackendStatus, BaseBackend

SCTENIFOLDKNK_SCRIPT = Path(__file__).parent / "sctenifoldknk_scripts" / "virtual_knockout.py"


@dataclass
class ScTenifoldKnkBackend(BaseBackend):
    """Run scTenifoldpy without importing its dependencies into CellQuorum."""

    name: str = "sctenifoldknk"
    kind: BackendKind = "external"
    python_executable: str = field(default_factory=lambda: sys.executable)
    requirement_list: list[BackendRequirement] = field(
        default_factory=lambda: [
            BackendRequirement(
                name="scTenifoldpy",
                requirement_type="other",
                install_hint="Install scTenifoldpy==0.5.1 in an isolated Python environment; "
                "set perturbation.sctenifoldknk.python_executable to its Python.",
            )
        ]
    )

    def status(self) -> BackendStatus:
        """Check the configured interpreter and actual package import."""
        available = False
        detail = ""
        try:
            proc = subprocess.run(
                [self.python_executable, "-c", "from scTenifold import virtual_knockout"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            available = proc.returncode == 0
            detail = proc.stderr[-1000:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            detail = str(exc)
        return BackendStatus(
            name=self.name,
            kind="external",
            available=available,
            requirements=self.requirements(),
            missing=[] if available else ["scTenifoldpy"],
            details={"python_executable": self.python_executable, "error": detail},
        )

    def run_script(self, args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        """Invoke the native-package helper with bounded BLAS threads."""
        caps = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        return subprocess.run(
            [self.python_executable, str(SCTENIFOLDKNK_SCRIPT), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env={**os.environ, **caps},
        )


def build_sctenifoldknk_backend(
    *,
    python_executable: str | None = None,
) -> ScTenifoldKnkBackend:
    """Build the optional backend for planner discovery."""
    return ScTenifoldKnkBackend(python_executable=python_executable or sys.executable)
