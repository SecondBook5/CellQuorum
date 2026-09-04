"""End-to-end SoupX on one real CellRanger library (skips unless data + R present).

This is the real proof that SoupX works end-to-end inside CellQuorum on the
manuscript data — the ambient-RNA correction both the keratinocyte and mast-cell
papers were blocked on. It is intentionally a skippable integration test (needs
multi-GB CellRanger matrices and minutes of R), NOT a unit test.

The Cell Ranger root is named by the ``CELLQUORUM_TEST_CELLRANGER_ROOT`` environment
variable rather than hardcoded, so this test is runnable by anyone with the data
instead of only on the machine it was written on. See ``tests/conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _external_data import require_cellranger_library, require_r_package

# Real data plus minutes of R: integration, R-backed, and slow. The markers let this be
# deselected wholesale (`-m "not integration"`) even where the data is present, which
# `skipif` alone cannot do.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.r,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def soupx_library():
    """Resolve one real Cell Ranger library with both matrices present.

    Resolved in a fixture rather than at module scope so the filesystem checks happen
    at run time. A module-scope check runs during collection for every session, even
    when this test is deselected.

    Returns:
        Tuple of (raw matrix path, filtered matrix path).
    """

    # One known LE library (raw + filtered present per the le_kc project).
    library = require_cellranger_library(
        "Set1_norm_LE",
        "LE1_v8",
        "outs",
        needs=("raw_feature_bc_matrix.h5", "filtered_feature_bc_matrix.h5"),
    )

    # Hand back both matrices SoupX needs.
    return library / "raw_feature_bc_matrix.h5", library / "filtered_feature_bc_matrix.h5"


def test_soupx_on_real_library(tmp_path, soupx_library):
    """SoupX corrects one real library and imports as integer counts."""

    # Probing for the R package here (not at import) keeps the Rscript subprocess off
    # the collection path; the result is cached across every test that asks.
    require_r_package("SoupX")

    from cellquorum.backends.rscript import RscriptBackend
    from cellquorum.stages.ambient_correction.soupx import (
        import_corrected_matrix,
        run_soupx_library,
    )

    raw, filtered = soupx_library

    # Run SoupX on the real raw+filtered pair.
    out_dir = tmp_path / "LE1"
    rho = run_soupx_library(
        raw,
        filtered,
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
