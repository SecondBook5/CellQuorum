"""CytoTRACE 2 developmental-potency compute, behind an import guard.

CytoTRACE 2 ships a file-based entrypoint (a genes x cells table in, a
per-cell results table out) and downloads pretrained model weights on first
use. The heavy call is isolated in :func:`run_cytotrace2` so the method layer
can mock it and so every failure is retyped into a recoverable
``CytoTraceComputeError`` subclass (skip-not-crash).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


class CytoTraceComputeError(Exception):
    """Base for recoverable CytoTRACE 2 failures (→ MethodSkip)."""


class CytoTraceUnavailable(CytoTraceComputeError):
    """cytotrace2-py is not importable."""


class NoCounts(CytoTraceComputeError):
    """No usable counts matrix (empty genes/cells) to score."""


class CytoTraceFailed(CytoTraceComputeError):
    """The CytoTRACE 2 run raised."""


def resolve_counts(adata: ad.AnnData, counts_layer: str | None) -> np.ndarray:
    """Return a dense cells x genes counts array (configured layer or ``.X``).

    Sparse inputs are densified; the CytoTRACE 2 entrypoint reads a plain table.
    Raises ``NoCounts`` when the chosen source is empty or cannot be densified.
    """
    if counts_layer:
        if counts_layer not in adata.layers:
            raise NoCounts(f"counts layer '{counts_layer}' absent")
        mat = adata.layers[counts_layer]
    else:
        mat = adata.X
    try:
        dense = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
        dense = np.asarray(dense, dtype="float64")
    except (ValueError, TypeError) as exc:
        raise NoCounts(f"counts matrix is not numeric: {exc}") from exc
    if dense.size == 0 or dense.shape[0] == 0 or dense.shape[1] == 0:
        raise NoCounts("counts matrix is empty")
    return dense


def run_cytotrace2(
    counts: np.ndarray,
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
    try:
        from cytotrace2_py.cytotrace2_py import cytotrace2
    except Exception as exc:  # noqa: BLE001 — any import failure → skip
        raise CytoTraceUnavailable(f"cytotrace2-py unavailable: {exc}") from exc

    work = Path(workdir)
    try:
        work.mkdir(parents=True, exist_ok=True)
        # CytoTRACE 2 expects genes as rows, cells as columns.
        table = pd.DataFrame(counts.T, index=list(var_names), columns=list(obs_names))
        input_path = work / "cytotrace2_input.tsv"
        table.to_csv(input_path, sep="\t")
        out_dir = work / "cytotrace2_results"
        cytotrace2(
            str(input_path),
            species=species,
            output_dir=str(out_dir),
            seed=int(seed),
            disable_plotting=True,
            disable_parallelization=bool(disable_parallelization),
            batch_size=int(batch_size),
            smooth_batch_size=int(smooth_batch_size),
        )
        results_path = out_dir / "cytotrace2_results.txt"
        frame = pd.read_csv(results_path, sep="\t", index_col=0)
    except CytoTraceComputeError:
        raise
    except Exception as exc:  # noqa: BLE001 — retype any run failure
        raise CytoTraceFailed(f"cytotrace2 run failed: {exc}") from exc
    return frame


__all__ = [
    "CytoTraceComputeError",
    "CytoTraceUnavailable",
    "NoCounts",
    "CytoTraceFailed",
    "resolve_counts",
    "run_cytotrace2",
]
