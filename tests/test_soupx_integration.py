"""End-to-end SoupX on one real CellRanger library (skips if data/R absent).

This is the real proof that SoupX works end-to-end inside CellQuorum on the
manuscript data — the ambient-RNA correction both the keratinocyte and mast-cell
papers were blocked on. It is intentionally a skippable integration test (needs
the multi-GB /mnt/e CellRanger matrices and minutes of R), NOT a unit test.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

# One known LE library (raw + filtered present per the le_kc project).
_LIB = Path("/mnt/e/lymphedema_cellranger/Set1_norm_LE/LE1_v8/outs")
_RAW = _LIB / "raw_feature_bc_matrix.h5"
_FILT = _LIB / "filtered_feature_bc_matrix.h5"


def _soupx_available() -> bool:
    """Return True when Rscript and the SoupX R package are both available."""

    # Rscript must be on PATH.
    if shutil.which("Rscript") is None:
        return False

    # SoupX must be installed in the R library.
    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            'quit(status=!requireNamespace("SoupX", quietly=TRUE))',
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not (_RAW.is_file() and _FILT.is_file() and _soupx_available()),
    reason="real CellRanger data or Rscript+SoupX unavailable",
)


def test_soupx_on_real_library(tmp_path):
    """SoupX corrects one real library and imports as integer counts."""

    from cellquorum.stages.ambient_correction.soupx import (
        import_corrected_matrix,
        run_soupx_library,
    )
    from cellquorum.backends.rscript import RscriptBackend

    # Run SoupX on the real raw+filtered pair.
    out_dir = tmp_path / "LE1"
    rho = run_soupx_library(
        _RAW,
        _FILT,
        out_dir,
        RscriptBackend(),
        resolution=0.5,
        round_to_int=True,
        timeout=1800,
    )

    # rho should be a small, plausible contamination fraction (LE median ~0.015).
    assert 0.0 < rho < 0.2

    # The corrected matrix imports as integer counts.
    adata = import_corrected_matrix(out_dir, "P1_LE")
    assert adata.n_obs > 0 and adata.n_vars > 0
    counts = adata.layers["counts"]
    dense = counts[:50].toarray() if hasattr(counts, "toarray") else counts[:50]
    assert np.allclose(dense, np.round(dense))  # integer counts
