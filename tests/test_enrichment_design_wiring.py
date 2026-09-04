from __future__ import annotations

from cellquorum.config.design import DesignConfig
from cellquorum.stages.comparative.enrichment.stage import EnrichmentStage


class _Cfg:
    organism = "mouse"
    design = DesignConfig(
        donor_col="patient_id",
        condition_col="condition",
        case="Disease",
        control="Normal",
        paired=True,
    )


class _Ctx:
    config = _Cfg()


def test_augment_injects_design_and_organism_and_default_methods():
    aug = EnrichmentStage()._augment_config(_Ctx(), {})
    assert aug["case"] == "Disease"
    assert aug["control"] == "Normal"
    assert aug["condition_col"] == "condition"
    assert aug["donor_col"] == "patient_id"
    assert aug["paired"] is True
    assert aug["organism"] == "mouse"
    assert [m["method"] for m in aug["methods"]] == ["gsea", "ora", "gsva", "activity"]


def test_augment_no_design_leaves_case_unset_but_sets_organism():
    class _C:
        organism = "human"
        design = None

    class _X:
        config = _C()

    aug = EnrichmentStage()._augment_config(_X(), {})
    assert aug.get("case") is None
    assert aug["organism"] == "human"
