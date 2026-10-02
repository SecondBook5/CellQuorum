"""PCA routing: scclr-normalized layer -> scclr sparse PCA; else scanpy PCA."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from cellquorum.core.contracts import set_layer_tag
from cellquorum.stages.preprocessing.dimensionality.pca import PCAMethod


class _Paths:
    def __init__(self, tmp: Path):
        self.figures = tmp
        self.scratch = tmp


class _Ctx:
    def __init__(self, adata, tmp, *, backend_registry=None):
        self._adata = adata
        self.paths = _Paths(tmp)
        self.config = {}
        self.backend_registry = backend_registry

    def require_adata(self):
        return self._adata


def _scclr_registry_or_skip():
    from cellquorum.backends.registry import build_default_backend_registry

    registry = build_default_backend_registry()
    if not registry.get("scclr").status().available:
        pytest.skip("scclr environment unavailable (isolated micromamba env not built)")
    return registry


def _scclr_normalized_adata(tmp: Path, registry):
    """Build an adata whose normalized layer was produced by the scclr backend."""

    from cellquorum.stages.preprocessing.config import NormalizationConfig
    from cellquorum.stages.preprocessing.normalization import normalize_adata

    rng = np.random.default_rng(0)
    counts = rng.negative_binomial(2, 0.15, size=(80, 30)).astype(np.float32)
    adata = ad.AnnData(X=counts)
    cfg = NormalizationConfig(
        recipe="cellquorum_pf_log1p_pf_v1", output_layer="cellquorum_normalized"
    )
    result = normalize_adata(adata, cfg, backend=registry.get("scclr"), scratch_dir=tmp)
    return result.adata


def test_scclr_layer_routes_to_scclr_pca(tmp_path):
    """A scclr-normalized layer (with row_center) uses scclr's sparse PCA."""
    registry = _scclr_registry_or_skip()
    adata = _scclr_normalized_adata(tmp_path, registry)
    # The scclr normalization must have left a row_center obs column.
    assert "cellquorum_normalized_row_center" in adata.obs.columns

    result = PCAMethod().run(
        adata,
        {"n_pcs": 10, "max_pcs": 20, "input_layer": "cellquorum_normalized"},
        context=_Ctx(adata, tmp_path, backend_registry=registry),
    )
    from cellquorum.methods.base import MethodSkip

    assert not isinstance(result, MethodSkip)
    assert result.metrics["compute"] == "scclr"
    assert result.adata.obsm["X_pca"].shape[0] == 80
    assert result.adata.obsm["X_pca"].shape[1] == 10


def test_scclr_pca_honours_the_highly_variable_mask(tmp_path):
    """The scclr route must restrict itself to the HVGs, not ignore the mask.

    ``_run_scclr_pca`` took no ``mask_var`` argument at all, while its caller dutifully built
    one. So on every run of the default PFlog1pPF recipe -- which is what puts a row_center
    on the object and selects this route -- ``use_highly_variable: true`` was constructed and
    then dropped, and the basis came from all genes.

    The proof is that the mask changes the answer: an embedding built from 8 of 30 genes
    cannot equal one built from all 30. It also pins the component cap, since asking for more
    components than the subset has genes is an error deep inside the SVD.
    """
    registry = _scclr_registry_or_skip()
    adata = _scclr_normalized_adata(tmp_path, registry)

    hvg = np.zeros(adata.n_vars, dtype=bool)
    hvg[:8] = True
    adata.var["highly_variable"] = hvg

    masked = PCAMethod().run(
        adata.copy(),
        {
            "n_pcs": 5,
            "max_pcs": 20,
            "input_layer": "cellquorum_normalized",
            "use_highly_variable": True,
        },
        context=_Ctx(adata, tmp_path, backend_registry=registry),
    )
    all_genes = PCAMethod().run(
        adata.copy(),
        {
            "n_pcs": 5,
            "max_pcs": 20,
            "input_layer": "cellquorum_normalized",
            "use_highly_variable": False,
        },
        context=_Ctx(adata, tmp_path, backend_registry=registry),
    )

    assert masked.metrics["compute"] == "scclr"
    assert masked.adata.obsm["X_pca"].shape == (80, 5)
    assert not np.allclose(
        masked.adata.obsm["X_pca"], all_genes.adata.obsm["X_pca"]
    ), "the HVG mask made no difference, so the scclr path is ignoring it again"


def test_scclr_pca_caps_components_at_the_subset_width(tmp_path):
    """Requesting more components than the HVG subset has genes must not reach the SVD."""
    registry = _scclr_registry_or_skip()
    adata = _scclr_normalized_adata(tmp_path, registry)

    hvg = np.zeros(adata.n_vars, dtype=bool)
    hvg[:6] = True
    adata.var["highly_variable"] = hvg

    result = PCAMethod().run(
        adata,
        {
            "n_pcs": "auto",
            "max_pcs": 20,
            "input_layer": "cellquorum_normalized",
            "use_highly_variable": True,
        },
        context=_Ctx(adata, tmp_path, backend_registry=registry),
    )
    assert result.adata.obsm["X_pca"].shape[1] <= 5


def test_asking_for_hvgs_that_do_not_exist_is_refused(tmp_path):
    """A config demanding HVGs must fail, not quietly build the basis from every gene.

    This is the whole failure mode: `stages.feature_selection: true` was skipped by a second
    switch, so the flag never appeared, and PCA carried on over ~33,000 genes for 40 minutes
    while reporting success.
    """
    from cellquorum.core.exceptions import CellQuorumStageError

    rng = np.random.default_rng(2)
    lognorm = np.abs(rng.normal(loc=2.0, scale=1.5, size=(60, 20)).astype(np.float32))
    adata = ad.AnnData(X=lognorm)
    adata.layers["cellquorum_normalized"] = lognorm
    set_layer_tag(
        adata, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_log1p_cp10k_v1"
    )

    with pytest.raises(CellQuorumStageError, match="feature.selection"):
        PCAMethod().run(
            adata,
            {
                "n_pcs": 5,
                "input_layer": "cellquorum_normalized",
                "use_highly_variable": True,
            },
            context=_Ctx(adata, tmp_path),
        )


def test_hvgs_are_used_automatically_when_present(tmp_path):
    """Default (unset) means "follow feature_selection", so the flag needs no restating."""
    rng = np.random.default_rng(3)
    lognorm = np.abs(rng.normal(loc=2.0, scale=1.5, size=(60, 20)).astype(np.float32))
    adata = ad.AnnData(X=lognorm)
    adata.layers["cellquorum_normalized"] = lognorm
    set_layer_tag(
        adata, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_log1p_cp10k_v1"
    )
    hvg = np.zeros(20, dtype=bool)
    hvg[:6] = True
    adata.var["highly_variable"] = hvg

    followed = PCAMethod().run(
        adata.copy(),
        {"n_pcs": 4, "input_layer": "cellquorum_normalized"},
        context=_Ctx(adata, tmp_path),
    )
    forced_off = PCAMethod().run(
        adata.copy(),
        {"n_pcs": 4, "input_layer": "cellquorum_normalized", "use_highly_variable": False},
        context=_Ctx(adata, tmp_path),
    )
    assert not np.allclose(
        followed.adata.obsm["X_pca"], forced_off.adata.obsm["X_pca"]
    ), "an unset use_highly_variable ignored the flag feature_selection produced"


def test_standard_layer_routes_to_scanpy_pca(tmp_path):
    """A standard lognorm layer (no row_center) uses the scanpy PCA path."""
    rng = np.random.default_rng(1)
    lognorm = rng.normal(loc=2.0, scale=1.5, size=(120, 40)).astype(np.float32)
    lognorm[:60, :5] += 3.0
    adata = ad.AnnData(X=np.abs(lognorm))
    adata.layers["cellquorum_normalized"] = np.abs(lognorm)
    set_layer_tag(
        adata, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_log1p_cp10k_v1"
    )

    result = PCAMethod().run(
        adata,
        {"n_pcs": 8, "max_pcs": 20, "input_layer": "cellquorum_normalized"},
        context=_Ctx(adata, tmp_path),
    )
    from cellquorum.methods.base import MethodSkip

    assert not isinstance(result, MethodSkip)
    # No row_center -> standard scanpy path (cpu/gpu), never "scclr".
    assert result.metrics["compute"] != "scclr"
    assert result.adata.obsm["X_pca"].shape == (120, 8)
