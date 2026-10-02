"""build_pipeline_context must load config.paths.manifest into context.manifest."""

from __future__ import annotations

import pandas as pd

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.pipeline import build_pipeline_context


def _base_config(tmp_path, manifest_path):
    return CellQuorumConfig.model_validate(
        {
            "project": {"name": "ctx_manifest_test"},
            "paths": {
                "output_dir": str(tmp_path / "run"),
                "manifest": str(manifest_path) if manifest_path else None,
            },
        }
    )


def test_context_loads_manifest_when_configured(tmp_path):
    manifest_csv = tmp_path / "m.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "cellranger_path": ["A/s1", "A/s2"],
            "condition": ["Normal", "LE"],
            "batch": ["B1", "B1"],
        }
    ).to_csv(manifest_csv, index=False)

    context = build_pipeline_context(_base_config(tmp_path, manifest_csv))

    assert context.manifest is not None
    assert list(context.manifest["sample_id"]) == ["s1", "s2"]
    assert "cellranger_path" in context.manifest.columns


def test_context_manifest_is_none_when_unset(tmp_path):
    context = build_pipeline_context(_base_config(tmp_path, None))
    assert context.manifest is None
