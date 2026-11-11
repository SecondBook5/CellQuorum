"""CytoTRACE 2 developmental-potency compute, behind an import guard.

CytoTRACE 2 ships a file-based entrypoint (a genes x cells table in, a
per-cell results table out) and downloads pretrained model weights on first
use. The heavy call is isolated in :func:`run_cytotrace2` so the method layer
can mock it and so every failure is retyped into a recoverable
``CytoTraceComputeError`` subclass (skip-not-crash).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Number of gene rows written to the CytoTRACE 2 input TSV per chunk. The table
# is genes x cells; at whole-object scale (e.g. 72k cells x 37k genes) a single
# dense float64 materialization is ~21 GB and the transposed DataFrame copy adds
# another ~21 GB — enough to OOM-kill the process. Streaming in gene-row slabs
# keeps peak extra memory to ~(chunk x n_cells x 8 bytes) while producing a TSV
# that is byte-for-byte identical to the one-shot write.
_GENE_CHUNK = 1000


class CytoTraceComputeError(Exception):
    """Base for recoverable CytoTRACE 2 failures (→ MethodSkip)."""


class CytoTraceUnavailable(CytoTraceComputeError):
    """cytotrace2-py is not importable."""


class NoCounts(CytoTraceComputeError):
    """No usable counts matrix (empty genes/cells) to score."""


class CytoTraceFailed(CytoTraceComputeError):
    """The CytoTRACE 2 run raised."""


def resolve_counts(adata: ad.AnnData, counts_layer: str | None) -> np.ndarray | sp.spmatrix:
    """Return the cells x genes counts matrix (configured layer or ``.X``).

    The matrix is returned AS-IS — sparse stays sparse — so the caller can stream
    it to disk without ever materializing the full dense array (see
    :func:`run_cytotrace2`). Raises ``NoCounts`` when the chosen source is empty
    or non-numeric.
    """
    if counts_layer:
        if counts_layer not in adata.layers:
            raise NoCounts(f"counts layer '{counts_layer}' absent")
        mat = adata.layers[counts_layer]
    else:
        mat = adata.X
    if not sp.issparse(mat):
        try:
            mat = np.asarray(mat)
        except (ValueError, TypeError) as exc:
            raise NoCounts(f"counts matrix is not array-like: {exc}") from exc
    try:
        if not np.issubdtype(np.dtype(mat.dtype), np.number):
            raise NoCounts(f"counts matrix is not numeric (dtype {mat.dtype})")
    except TypeError as exc:  # dtype not interpretable as a numpy dtype
        raise NoCounts(f"counts matrix is not numeric: {exc}") from exc
    shape = getattr(mat, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] == 0 or shape[1] == 0:
        raise NoCounts("counts matrix is empty")
    return mat


def _write_counts_tsv(
    counts: np.ndarray | sp.spmatrix,
    var_names: list[str],
    obs_names: list[str],
    input_path: Path,
) -> None:
    """Stream the genes x cells counts table to ``input_path`` in gene-row chunks.

    CytoTRACE 2 expects genes as rows and cells as columns. Writing the whole
    transposed matrix at once would densify the full object twice (once for the
    dense array, once for the DataFrame copy). Instead we transpose to a
    row-sliceable form and write ``_GENE_CHUNK`` gene rows at a time: the first
    chunk writes the header, the rest append. Each slab is cast to float64 so the
    emitted text is identical, value-for-value, to the one-shot
    ``DataFrame(counts.T).to_csv(...)`` this replaces.
    """
    if isinstance(counts, sp.spmatrix | sp.sparray):
        # .T on CSR/CSC is cheap (swaps orientation); tocsr makes gene-row
        # slicing efficient. Held as sparse, so no full dense copy.
        genes_by_cells = counts.T.tocsr()

        def slab(start: int, stop: int) -> np.ndarray:
            return genes_by_cells[start:stop].toarray().astype("float64", copy=False)
    else:
        genes_by_cells = np.asarray(counts).T

        def slab(start: int, stop: int) -> np.ndarray:
            return np.asarray(genes_by_cells[start:stop], dtype="float64")

    n_genes = genes_by_cells.shape[0]
    for start in range(0, n_genes, _GENE_CHUNK):
        stop = min(start + _GENE_CHUNK, n_genes)
        chunk = pd.DataFrame(
            slab(start, stop),
            index=list(var_names[start:stop]),
            columns=list(obs_names),
        )
        chunk.to_csv(
            input_path,
            sep="\t",
            header=(start == 0),
            mode="w" if start == 0 else "a",
        )
        del chunk


def _feature_gene_mask(var_names: list[str], species: str) -> np.ndarray | None:
    """Boolean mask selecting input genes that map into the CytoTRACE 2 model
    feature panel, using the library's OWN mapping tables.

    CytoTRACE 2's ``preprocess`` reindexes each batch to exactly the model
    ``features`` panel (dropping every gene outside it) BEFORE computing the CPM
    ``log2_data`` and the per-cell ``rank_data``. Genes that don't map into the
    panel therefore contribute nothing to any score. Dropping them up front is
    numerically identical to passing them (verified byte-for-byte on the model
    inputs) and shrinks the genes x cells table the library must hold in memory
    (~37k genes -> ~14k), which is what makes the whole-object run fit in RAM.

    Returns ``None`` (keep every gene) when the mapping can't be built or the
    species path isn't the exactly-replicated human one — a safe degrade that
    never alters the science, only the memory footprint.
    """
    try:
        from cytotrace2_py.common.gen_utils import build_mapping_dict
    except Exception:  # noqa: BLE001 — no library → don't subset, run full
        return None
    if str(species).lower() != "human":
        # The mouse alias path is order/precedence-sensitive; rather than risk a
        # divergent reimplementation we keep all genes (correct, just larger).
        return None
    try:
        mt_dict, alias_dict, _mouse_alias, features = build_mapping_dict()
        feature_set = set(features.values)
        names = pd.Index([str(v) for v in var_names])
        # Replicates cytotrace2 preprocess() lines: map via ortholog dict, then
        # fill NaNs from the alias/previous-symbol dict.
        mapped = names.map(mt_dict).astype(object)
        na = np.asarray(pd.isna(mapped))
        mapped_values = np.asarray(mapped, dtype=object).copy()
        if bool(na.any()):
            mapped_values[na] = np.asarray(names[na].map(alias_dict), dtype=object)
        mask = np.array([m in feature_set for m in mapped_values], dtype=bool)
    except Exception:  # noqa: BLE001 — any mapping hiccup → keep all genes
        return None
    if not mask.any():
        return None
    return mask


def worker_python() -> str:
    """Return the interpreter selected for both preflight and scoring."""
    return os.environ.get("CELLQUORUM_CYTOTRACE2_PYTHON") or sys.executable


def worker_issue() -> str | None:
    """Probe the selected interpreter before upstream lineage computations start."""
    probe = (
        "from importlib.metadata import version; import sys; "
        "from cytotrace2_py.cytotrace2_py import cytotrace2; "
        "sys.exit('CytoTRACE 2 1.1.0.4 needs pandas 2 for gene mapping') "
        "if version('cytotrace2-py') == '1.1.0.4' and int(version('pandas').split('.')[0]) >= 3 "
        "else None"
    )
    try:
        result = subprocess.run(
            [worker_python(), "-I", "-c", probe], capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return None
        reason = (result.stderr.strip().splitlines() or [f"exit {result.returncode}"])[-1]
    except (OSError, subprocess.TimeoutExpired) as exc:
        reason = str(exc)
    return f"Set CELLQUORUM_CYTOTRACE2_PYTHON to the installed CytoTRACE 2 worker ({reason})."


def run_cytotrace2(
    counts: np.ndarray | sp.spmatrix,
    var_names: list[str],
    obs_names: list[str],
    *,
    species: str,
    workdir: Path | str,
    seed: int,
    disable_parallelization: bool,
    batch_size: int,
    smooth_batch_size: int,
) -> pd.DataFrame:
    """Run CytoTRACE 2 over a counts matrix; return its per-cell results frame.

    Isolated so the method layer can monkeypatch it in tests (the real call
    downloads model weights and runs a torch model). Writes a temporary genes x
    cells table under ``workdir`` and reads the results table back, indexed by
    cell name. Any failure (import or run) is retyped to a
    ``CytoTraceComputeError`` subclass.
    """
    settings = dict(
        species=species,
        seed=seed,
        disable_parallelization=disable_parallelization,
        batch_size=batch_size,
        smooth_batch_size=smooth_batch_size,
    )
    work = Path(workdir).resolve()
    worker = worker_python()
    log = work / "cytotrace2-worker.log"
    try:
        work.mkdir(parents=True, exist_ok=True)
        # The model, mapping tables and human feature mask use the same interpreter.
        with TemporaryDirectory(dir=work) as temp:
            exchange = Path(temp)
            if sp.issparse(counts):
                sp.save_npz(exchange / "counts.npz", counts, compressed=False)
            else:
                np.save(exchange / "counts.npy", counts, allow_pickle=False)
            (exchange / "input.json").write_text(json.dumps([var_names, obs_names, settings]))
            with log.open("w") as handle:
                subprocess.run(
                    [worker, "-I", __file__, str(work)],
                    cwd=exchange,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            result = Path("cytotrace2_results/cytotrace2_results.txt")
            frame = pd.read_csv(exchange / result, sep="\t", dtype={0: str}, keep_default_na=False)
            frame = frame.set_index(frame.columns[0])
            if not frame.index.is_unique or set(frame.index) != set(obs_names):
                raise CytoTraceFailed("CytoTRACE 2 returned incomplete or duplicate cell IDs")
            scores = frame["CytoTRACE2_Score"].astype(float)
            if not scores.between(0, 1).all():
                raise CytoTraceFailed("CytoTRACE 2 returned missing or invalid potency scores")
            (work / result.parent).mkdir(exist_ok=True)
            (exchange / result).replace(work / result)
            (exchange / "cytotrace2-runtime.json").replace(work / "cytotrace2-runtime.json")
            return frame
    except CytoTraceComputeError:
        raise
    except subprocess.CalledProcessError as exc:
        raise CytoTraceFailed(
            f"CytoTRACE 2 worker exited {exc.returncode}. Check CELLQUORUM_CYTOTRACE2_PYTHON "
            f"and the worker log at {log}"
        ) from exc
    except (FileNotFoundError, PermissionError) as exc:
        raise CytoTraceUnavailable(
            f"Check CELLQUORUM_CYTOTRACE2_PYTHON and worker files: {exc}"
        ) from exc
    except Exception as exc:
        raise CytoTraceFailed(f"cytotrace2 run failed: {exc}") from exc


def _score_counts(
    counts: np.ndarray | sp.spmatrix,
    var_names: list[str],
    obs_names: list[str],
    work: Path,
    settings: dict,
) -> None:
    try:
        from cytotrace2_py.cytotrace2_py import cytotrace2
    except ImportError as exc:
        raise CytoTraceUnavailable(
            f"Set CELLQUORUM_CYTOTRACE2_PYTHON to the CytoTRACE 2 environment: {exc}"
        ) from exc
    mask = _feature_gene_mask(var_names, settings["species"])
    if mask is not None:
        counts = counts[:, mask]
        var_names = [v for v, keep in zip(var_names, mask, strict=True) if keep]
    input_path = work / "cytotrace2_input.tsv"
    _write_counts_tsv(counts, var_names, obs_names, input_path)
    # Upstream passes output_dir to a shell. Use a fixed name in the private
    # working directory so paths with spaces are safe and stale outputs impossible.
    cytotrace2(str(input_path), output_dir="cytotrace2_results", disable_plotting=True, **settings)
    if version("cytotrace2-py") == "1.1.0.4":
        models = files("cytotrace2_py") / "resources" / "models"
        missing = [i for i in range(1, 20) if not (models / f"model{i}.pt").is_file()]
        if missing:
            raise CytoTraceFailed(f"Incomplete CytoTRACE 2 ensemble, missing models {missing}")
    packages = ("cytotrace2-py", "numpy", "pandas", "scanpy", "torch")
    runtime = {name: version(name) for name in packages}
    Path("cytotrace2-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")


if __name__ == "__main__":
    exchange = Path.cwd()
    workdir = Path(sys.argv[1])
    names, cells, options = json.loads((exchange / "input.json").read_text())
    sparse_input = exchange / "counts.npz"
    counts = (
        sp.load_npz(sparse_input)
        if sparse_input.exists()
        else np.load(exchange / "counts.npy", mmap_mode="r")
    )
    _score_counts(counts, names, cells, workdir, options)


__all__ = [
    "CytoTraceComputeError",
    "CytoTraceUnavailable",
    "NoCounts",
    "CytoTraceFailed",
    "resolve_counts",
    "run_cytotrace2",
]
