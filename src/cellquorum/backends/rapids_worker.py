"""Execute RAPIDS kernels in-process or with an isolated pandas/cuDF install.

The external interpreter runs this file directly and only needs the worker requirements.
The controller exchanges indexed H5AD files and only imports the requested results.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anndata import AnnData


def worker_python() -> str | None:
    return os.environ.get("CELLQUORUM_RAPIDS_PYTHON") or None


def _invoke(python: str, *args: str) -> str:
    process = subprocess.run(
        [python, "-I", str(Path(__file__).resolve()), *args], capture_output=True, text=True
    )
    if process.returncode:
        reason = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"RAPIDS worker failed ({process.returncode}): {reason[-2000:]}")
    return process.stdout


@lru_cache(maxsize=4)
def worker_status(python: str) -> dict[str, Any]:
    return json.loads(_invoke(python, "probe").strip().splitlines()[-1])


def _probe() -> dict[str, Any]:
    from importlib.metadata import version

    import cupy
    import rapids_singlecell  # noqa: F401

    if (cupy.arange(8) + 1).sum().item() != 36:
        raise RuntimeError("CUDA reduction returned an incorrect result")
    device = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
    return {
        "cuda_visible": True,
        "cuda_device_count": cupy.cuda.runtime.getDeviceCount(),
        "device": device["name"].decode(),
        "versions": {
            p: version(p) for p in ("rapids-singlecell-cu12", "cudf-cu12", "pandas", "anndata")
        },
    }


def _apply(adata: AnnData, operation: str, options: dict[str, Any]) -> None:
    import rapids_singlecell as rsc

    if operation == "pca":
        layer = options["layer"]
        rsc.get.anndata_to_GPU(adata, layer=layer)
        try:
            rsc.pp.pca(adata, **options)
        finally:
            rsc.get.anndata_to_CPU(adata, layer=layer)
    elif operation == "cluster":
        rep = options["use_rep"]
        if rep == "X":
            rsc.get.anndata_to_GPU(adata)
        try:
            rsc.pp.neighbors(
                adata, **{k: options[k] for k in ("n_neighbors", "use_rep", "random_state")}
            )
            rsc.tl.leiden(
                adata, **{k: options[k] for k in ("resolution", "random_state", "key_added")}
            )
        finally:
            if rep == "X":
                rsc.get.anndata_to_CPU(adata)
    elif operation == "harmony":
        rsc.pp.harmony_integrate(adata, **options)
    elif operation == "umap":
        rsc.tl.umap(adata, **options)
    else:
        raise ValueError(f"Unknown RAPIDS operation {operation!r}")


def _indexed(adata: AnnData) -> AnnData:
    import anndata as ad
    import pandas as pd

    return ad.AnnData(
        obs=pd.DataFrame(index=adata.obs_names), var=pd.DataFrame(index=adata.var_names)
    )


def _inputs(adata: AnnData, operation: str, options: dict[str, Any]) -> AnnData:
    data = _indexed(adata)
    if operation == "pca":
        layer = options["layer"]
        if layer is None:
            data.X = adata.X
        else:
            data.layers[layer] = adata.layers[layer]
        if mask := options["mask_var"]:
            data.var[mask] = adata.var[mask]
    elif operation == "cluster":
        rep = options["use_rep"]
        if rep == "X":
            data.X = adata.X
        else:
            data.obsm[rep] = adata.obsm[rep]
    elif operation == "harmony":
        rep, key = options["basis"], options["key"]
        data.obsm[rep] = adata.obsm[rep]
        data.obs[key] = adata.obs[key]
    elif operation == "umap":
        neighbors = adata.uns["neighbors"]
        data.uns["neighbors"] = neighbors
        for slot in ("distances", "connectivities"):
            key = neighbors.get(slot + "_key", slot)
            if key in adata.obsp:
                data.obsp[key] = adata.obsp[key]
        rep = neighbors.get("params", {}).get("use_rep")
        if rep and rep != "X":
            data.obsm[rep] = adata.obsm[rep]
        elif rep is None and "X_pca" in adata.obsm:
            data.obsm["X_pca"] = adata.obsm["X_pca"]
        else:
            data.X = adata.X
    return data


def _copy_results(source: AnnData, dest: AnnData, operation: str, options: dict[str, Any]) -> None:
    if not source.obs_names.equals(dest.obs_names) or not source.var_names.equals(dest.var_names):
        raise ValueError("RAPIDS result cell or gene identifiers differ from input")
    if operation == "pca":
        dest.obsm["X_pca"] = source.obsm["X_pca"]
        dest.varm["PCs"] = source.varm["PCs"]
        dest.uns["pca"] = source.uns["pca"]
    elif operation == "cluster":
        key = options["key_added"]
        dest.obs[key] = source.obs[key]
        dest.uns[key] = source.uns[key]
        dest.uns["neighbors"] = source.uns["neighbors"]
        for slot in ("connectivities", "distances"):
            graph = source.uns["neighbors"].get(slot + "_key", slot)
            dest.obsp[graph] = source.obsp[graph]
    elif operation == "harmony":
        key = options["adjusted_basis"]
        dest.obsm[key] = source.obsm[key]
    elif operation == "umap":
        dest.obsm["X_umap"] = source.obsm["X_umap"]
        dest.uns["umap"] = source.uns["umap"]


def run_rapids(adata: AnnData, operation: str, **options: Any) -> None:
    python = worker_python()
    if not python:
        _apply(adata, operation, options)
        return
    import anndata as ad

    with TemporaryDirectory(prefix="cellquorum-rapids-") as scratch:
        folder = Path(scratch)
        _inputs(adata, operation, options).write_h5ad(folder / "input.h5ad")
        _invoke(python, operation, scratch, json.dumps(options))
        result = ad.read_h5ad(folder / "output.h5ad")
        _copy_results(result, adata, operation, options)


if __name__ == "__main__":
    if sys.argv[1] == "probe":
        print(json.dumps(_probe()))
    else:
        import anndata as ad

        operation, directory, raw_options = sys.argv[1:]
        folder = Path(directory)
        options = json.loads(raw_options)
        data = ad.read_h5ad(folder / "input.h5ad")
        _apply(data, operation, options)
        result = _indexed(data)
        _copy_results(data, result, operation, options)
        result.write_h5ad(folder / "output.h5ad")
