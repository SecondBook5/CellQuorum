"""SoupX ambient-RNA correction — per-library R invocation + import.

Runs the bundled SoupX R script (ported from the validated le_kc script) via the
Rscript run_script adapter, one 10x library at a time, and imports the corrected
integer counts back into AnnData. Ambient correction runs before QC/integration.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import anndata as ad
import scipy.io as sio
import scipy.sparse as sp

from cellquorum.core.exceptions import CellQuorumBackendError, CellQuorumDataError

# Path to the bundled SoupX R script.
_SOUPX_R = Path(__file__).parent.parent / "backends" / "r_scripts" / "soupx_per_library.R"


class SoupXError(CellQuorumDataError):
    """Raised when SoupX output cannot be parsed or imported."""


def parse_rho(stdout: str) -> float:
    """
    Parse the contamination fraction from the R script's ``RHO=`` line.

    Args:
        stdout: Captured stdout from the SoupX R script.

    Returns:
        The estimated contamination fraction (rho).

    Raises:
        SoupXError: If no RHO= line is present.
    """

    # Find the RHO= marker the script prints.
    match = re.search(r"RHO=([0-9.eE+-]+)", stdout)
    if match is None:
        raise SoupXError("SoupX R output did not contain a RHO= line.")
    return float(match.group(1))


def run_soupx_library(
    raw_h5: str | Path,
    filtered_h5: str | Path,
    out_dir: str | Path,
    backend: Any,
    *,
    resolution: float,
    round_to_int: bool,
    timeout: int,
) -> float:
    """
    Run SoupX on one library via the Rscript adapter; return rho.

    Args:
        raw_h5: Path to raw_feature_bc_matrix.h5.
        filtered_h5: Path to filtered_feature_bc_matrix.h5.
        out_dir: Directory to write corrected matrix.mtx.gz + features/barcodes.
        backend: RscriptBackend.
        resolution: Quick-cluster resolution for autoEstCont.
        round_to_int: Whether to round corrected counts to integers.
        timeout: R timeout (seconds).

    Returns:
        The estimated contamination fraction (rho).

    Raises:
        CellQuorumBackendError: If the R run fails.
    """

    # Invoke the bundled SoupX script with positional args.
    result = backend.run_script(
        _SOUPX_R,
        [str(raw_h5), str(filtered_h5), str(out_dir), str(resolution), str(round_to_int).upper()],
        timeout=timeout,
    )
    if result.returncode != 0:
        raise CellQuorumBackendError(f"SoupX failed for {raw_h5}: {result.stderr.strip()[:500]}")
    rho = parse_rho(result.stdout)
    # Persist rho next to the corrected matrix so a resumed run can reuse this
    # library's contamination fraction without re-running SoupX.
    write_rho_sidecar(out_dir, rho)
    return rho


# Files a completed SoupX library output must contain to be reusable on resume.
_CORRECTED_OUTPUTS: tuple[str, ...] = (
    "matrix.mtx.gz",
    "features.tsv.gz",
    "barcodes.tsv.gz",
)

# Sidecar filename holding the per-library contamination fraction (rho).
_RHO_SIDECAR = "rho.txt"


def corrected_output_exists(out_dir: str | Path) -> bool:
    """
    Return whether a directory holds a complete corrected SoupX matrix.

    Args:
        out_dir: Per-library output directory.

    Returns:
        True when every required corrected-matrix file is present.
    """

    d = Path(out_dir)
    return all((d / name).is_file() for name in _CORRECTED_OUTPUTS)


def write_rho_sidecar(out_dir: str | Path, rho: float) -> None:
    """
    Write the per-library contamination fraction to a small sidecar file.

    Args:
        out_dir: Per-library output directory (created if absent).
        rho: Estimated contamination fraction.
    """

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / _RHO_SIDECAR).write_text(f"{rho:.6f}\n")


def read_rho_sidecar(out_dir: str | Path) -> float | None:
    """
    Read the per-library rho sidecar, or None when it is absent/unreadable.

    Args:
        out_dir: Per-library output directory.

    Returns:
        The stored contamination fraction, or None when unavailable.
    """

    path = Path(out_dir) / _RHO_SIDECAR
    if not path.is_file():
        return None
    try:
        return float(path.read_text().strip())
    except (ValueError, OSError):
        return None


def import_corrected_matrix(out_dir: str | Path, sample_id: str) -> ad.AnnData:
    """
    Read a corrected per-library matrix into AnnData (counts layer).

    Args:
        out_dir: Directory containing matrix.mtx.gz + features.tsv.gz + barcodes.tsv.gz.
        sample_id: Sample id to namespace barcodes.

    Returns:
        AnnData (cells x genes) with layers["counts"] = corrected integer counts.
    """

    import pandas as pd

    # Read the Matrix Market file (genes x cells) and transpose to cells x genes.
    d = Path(out_dir)
    mat = sio.mmread(str(d / "matrix.mtx.gz")).tocsr().T
    genes = pd.read_csv(d / "features.tsv.gz", header=None)[0].to_numpy()
    barcodes = pd.read_csv(d / "barcodes.tsv.gz", header=None)[0].to_numpy()

    # Build the AnnData; namespace barcodes by sample.
    adata = ad.AnnData(X=sp.csr_matrix(mat))
    adata.var_names = genes
    adata.obs_names = [f"{sample_id}_{bc}" for bc in barcodes]
    adata.var_names_make_unique()
    # Record sample_id as an explicit obs column (robust — do NOT reconstruct it
    # downstream by splitting namespaced barcodes, which can themselves contain '_').
    adata.obs["sample_id"] = sample_id
    adata.layers["counts"] = adata.X.copy()
    return adata


__all__ = [
    "SoupXError",
    "corrected_output_exists",
    "import_corrected_matrix",
    "parse_rho",
    "read_rho_sidecar",
    "run_soupx_library",
    "write_rho_sidecar",
]
