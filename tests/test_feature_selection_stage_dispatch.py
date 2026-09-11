"""feature_selection's method-name default must match FeatureSelectionConfig.method.

FeatureSelectionConfig.method defaults to "seurat_v3", but FeatureSelectionStage and
HVGMethod each hardcoded their own fallback of "seurat" (the v1, lognorm-based flavor)
for a bare dict missing the key. A caller that built config from something other than
the validated Pydantic model -- or a future direct `_run`/`input_contract` call -- would
silently get a different flavor, and a different required input layer, than the one
FeatureSelectionConfig actually defaults to.
"""

from __future__ import annotations

from cellquorum.stages.preprocessing.feature_selection.hvg import HVGMethod
from cellquorum.stages.preprocessing.feature_selection.stage import FeatureSelectionStage


def test_stage_default_method_matches_the_config_default():
    stage = FeatureSelectionStage()

    assert stage._select_method_name({}) == "seurat_v3"


def test_hvg_method_input_contract_default_matches_the_config_default():
    contract = HVGMethod().input_contract({})

    # seurat_v3 is a count flavor: requires the counts layer, not lognorm.
    assert contract.required_layers == ["counts"]
    assert contract.expected_kind == "counts"
