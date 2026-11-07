#!/usr/bin/env python3
"""Import every package the stages use, inside an environment, and report any that fail.

A lock file proves the packages were installed, not that they work together. Two real breaks
passed every test until this existed. ``mudata`` imports ``Settings`` from ``scverse_misc``, which
needs an optional extra that conda does not install, so ``import mudata``, ``import scvi`` and
``import liana`` all failed in the containers while the demo and the lock checks passed. And
``phate``, ``celltypist``, ``pycirclize`` and ``plotly`` were declared nowhere: they worked on a
workstation because they were installed by hand, and were missing from every fresh install.

Run it with the environment's Python, for example in an image:

    docker run --rm -v "$PWD/scripts":/s:ro --entrypoint micromamba IMAGE \\
        run -n cellquorum-core python /s/check_environment_imports.py core

Profiles are ``core`` (CPU) and ``gpu`` (everything in core plus the RAPIDS stack). Exit status is 1
when any required import fails. Optional external-environment backends (CellOracle, pySCENIC, scclr
and the like) live in their own environments and are not checked here.
"""

from __future__ import annotations

import importlib
import sys

CORE = [
    "anndata",
    "scanpy",
    "h5py",
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "statsmodels",
    "networkx",
    "matplotlib",
    "seaborn",
    "plotly",
    "pycirclize",
    "adjustText",
    "pydantic",
    "pydantic_settings",
    "omegaconf",
    "rich",
    "typer",
    "harmonypy",
    "decoupler",
    "mudata",
    "scvi",
    "liana",
    "cellrank",
    "scvelo",
    "palantir",
    "phate",
    "tensorly",
    "torch",
    "lightning",
    "celltypist",
    "scrublet",
    "cellquorum",
]
GPU = ["cupy", "cuml", "cugraph", "cuvs", "rapids_singlecell"]
PROFILES = {"core": CORE, "gpu": CORE + GPU}


def main() -> int:
    profile = sys.argv[1] if len(sys.argv) > 1 else "core"
    if profile not in PROFILES:
        print(f"unknown profile {profile!r}; use one of {sorted(PROFILES)}")
        return 2
    failed = []
    for name in PROFILES[profile]:
        try:
            module = importlib.import_module(name)
        except Exception as error:  # noqa: BLE001 - any failure to import is the finding
            failed.append(name)
            print(f"FAIL  {name}: {type(error).__name__}: {str(error)[:100]}")
        else:
            print(f"ok    {name} {getattr(module, '__version__', '')}")
    if profile == "gpu":
        import cupy
        import torch

        devices = cupy.cuda.runtime.getDeviceCount()
        available = torch.cuda.is_available()
        print(f"GPU devices visible to cupy: {devices}; torch.cuda.is_available(): {available}")
        if devices < 1 or not available:
            failed.append("gpu-device")
    print(f"\n{len(PROFILES[profile]) - len(failed)} of {len(PROFILES[profile])} imports ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
