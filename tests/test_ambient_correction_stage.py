"""AmbientCorrectionStage: skip contract + the corrected-counts wiring.

The wiring test is the one that matters: it proves the stage returns the
SoupX-corrected counts as result.adata (which the executor threads downstream),
not the stale input — the defect this test suite exists to prevent recurring.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from _external_data import ENV_CELLRANGER_ROOT, require_cellranger_library, require_r_package

from cellquorum.core.stage import StageResult
from cellquorum.stages.ambient_correction.config import AmbientCorrectionConfig
from cellquorum.stages.ambient_correction.stage import AmbientCorrectionStage


class _Ctx:
    def __init__(self, config):
        self.config = config
        self.adata = None

    def require_adata(self):
        raise AssertionError("ambient_correction must not require adata")


def test_stage_skips_when_disabled():
    cfg = type("C", (), {"ambient_correction": type("A", (), {"enabled": False})()})()
    result = AmbientCorrectionStage().run(_Ctx(cfg))
    assert isinstance(result, StageResult)
    assert result.metrics.get("skipped") is True


# ---- Real-data wiring test ---- #
#
# Needs real Cell Ranger matrices, located via the CELLQUORUM_TEST_CELLRANGER_ROOT
# environment variable rather than a hardcoded maintainer path, so anyone with the data
# can run it. See tests/_external_data.py. It skips with an actionable message when the
# variable is unset.


class _RegistryStub:
    """Minimal backend registry exposing get('rscript')."""

    def get(self, name):
        from cellquorum.backends.rscript import RscriptBackend

        if name == "rscript":
            return RscriptBackend()
        raise KeyError(name)


class _WiringCtx:
    """Context that feeds the stage a 1-library manifest + rscript backend."""

    def __init__(self, config, manifest, objects_dir):
        self.config = config
        self.adata = None
        self.manifest = manifest
        self.backend_registry = _RegistryStub()
        self.paths = type("P", (), {"objects": objects_dir})()

    def require_adata(self):
        raise AssertionError("ambient_correction must not require adata")


@pytest.mark.integration
@pytest.mark.r
@pytest.mark.slow
def test_stage_returns_corrected_counts_on_result_adata(tmp_path):
    """The stage must return SoupX-corrected counts as result.adata, not the input."""

    # Resolve the real library, skipping cleanly when the data or R package is absent.
    # Both checks happen here rather than in a module-scope skipif so the filesystem
    # stat and the Rscript subprocess stay off the collection path.
    require_cellranger_library(
        "Set1_norm_LE",
        "LE1_v8",
        "outs",
        needs=("raw_feature_bc_matrix.h5", "filtered_feature_bc_matrix.h5"),
    )
    require_r_package("SoupX")

    # The stage resolves libraries relative to the configured Cell Ranger root, so pass
    # the root itself rather than the individual library directory.
    cellranger_root = os.environ[ENV_CELLRANGER_ROOT]

    # A one-row manifest pointing at a real library.
    manifest = pd.DataFrame(
        {
            "sample_id": ["P1_LE"],
            "cellranger_path": ["Set1_norm_LE/LE1_v8"],
            "include": ["true"],
        }
    )
    config = type(
        "C",
        (),
        {
            "ambient_correction": AmbientCorrectionConfig(
                enabled=True,
                method="soupx",
                cellranger_root=cellranger_root,
                cluster_resolution=0.5,
                round_to_int=True,
                timeout_seconds=1800,
            )
        },
    )()
    ctx = _WiringCtx(config, manifest, tmp_path)

    result = AmbientCorrectionStage().run(ctx)

    # The corrected counts must be ON result.adata (the object threaded downstream),
    # not None and not a stale input.
    assert result.adata is not None
    assert "counts" in result.adata.layers
    assert result.adata.n_obs > 0
    # Metadata wiring: sample_id must be present on the corrected object.
    assert "sample_id" in result.adata.obs.columns
    assert set(result.adata.obs["sample_id"]) == {"P1_LE"}
    # Provenance: the contamination fraction is recorded and plausible.
    fractions = result.metrics["contamination_fractions"]
    assert "P1_LE" in fractions
    assert 0.0 < fractions["P1_LE"] < 0.2


# ---- Regression tests for null cellranger_path handling ---- #


def test_resolve_manifest_skips_path_only_manifest_without_crash():
    """Path-only manifests (cellranger_path column all-null) must return [] gracefully."""

    import types

    from cellquorum.stages.ambient_correction.stage import _resolve_manifest

    # A path-only manifest: the cellranger_path column exists (per schema) but
    # all values are None — mimics the to_dataframe() output after the schema change.
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "path": ["s1.h5ad", "s2.h5ad"],
            "cellranger_path": [None, None],
        }
    )
    # Must return [] (not crash) so the stage skips gracefully.
    assert _resolve_manifest(types.SimpleNamespace(manifest=df)) == []


def test_resolve_manifest_keeps_only_rows_with_cellranger_path():
    """Mixed manifests: keep only rows with non-null cellranger_path values."""

    import types

    from cellquorum.stages.ambient_correction.stage import _resolve_manifest

    # A mixed manifest: one row has a real cellranger_path, one is null.
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "cellranger_path": ["A/s1", None],
        }
    )
    rows = _resolve_manifest(types.SimpleNamespace(manifest=df))
    # Must return only the non-null row.
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "s1"
    assert rows[0]["cellranger_path"] == "A/s1"
