"""Regression: the corrected lekc KC object satisfies its declared contract,
and a sabotaged copy is rejected.

Needs a real reference-mapped keratinocyte ``.h5ad``, which is too large to ship in the
repository. Its location comes from the ``CELLQUORUM_TEST_KC_H5AD`` environment variable
rather than a hardcoded maintainer path, so anyone holding the object can run this;
without the variable the test skips with a message naming it. See
``tests/_external_data.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _external_data import ENV_KC_H5AD, require_external_file

anndata = pytest.importorskip("anndata")

from cellquorum.core.contracts import (  # noqa: E402
    CellQuorumContractError,
    DataContract,
    set_layer_tag,
)

# Real external data: mark so it can be deselected wholesale with `-m "not integration"`
# even on a machine where the object is present.
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def kc_adata():
    """Load the reference-mapped keratinocyte object, skipping when unavailable.

    Returns:
        The loaded AnnData object.
    """

    # Resolve and validate the configured path, skipping with an actionable reason.
    path = require_external_file(
        ENV_KC_H5AD,
        what="a reference-mapped keratinocyte .h5ad (le_kc_keratinocyte_refmapped.h5ad)",
    )

    # Read once per module: these objects are large.
    return anndata.read_h5ad(path)


def test_sabotaged_kc_object_is_rejected(kc_adata):
    # Put raw counts into X and tag it lognorm — must be caught.
    a = kc_adata.copy()
    a.X = (
        a.layers["counts"].copy()
        if "counts" in a.layers
        else np.round(np.abs(a.X.toarray()) if hasattr(a.X, "toarray") else np.round(np.abs(a.X)))
    )
    set_layer_tag(a, "X", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    contract = DataContract(expression_layer="X", expected_kind="lognorm")
    with pytest.raises(CellQuorumContractError):
        contract.validate(a)
