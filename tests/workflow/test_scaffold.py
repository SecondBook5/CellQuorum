from __future__ import annotations

from cellquorum.cli.workflow import scaffold
from cellquorum.config.models import StageSelectionConfig


def test_scaffold_has_seven_table0_methods() -> None:
    assert scaffold.SCAFFOLD == [
        "pseudobulk",
        "subclustering",
        "pathway_enrichment",
        "rna_velocity",
        "phate_pseudotime",
        "cell_cell_communication",
        "progeny",
    ]


def test_every_mapped_stage_is_a_real_stage_flag() -> None:
    legal = set(StageSelectionConfig.model_fields)
    for method, stages in scaffold.SCAFFOLD_METHOD_STAGES.items():
        assert stages, f"{method} maps to no stages"
        for stage in stages:
            assert stage in legal, f"{method} -> unknown stage flag {stage!r}"
    for stage in scaffold.MANDATORY_STAGES:
        assert stage in legal, f"mandatory stage {stage!r} is not a real flag"


def test_every_scaffold_method_is_mapped() -> None:
    assert set(scaffold.SCAFFOLD_METHOD_STAGES) == set(scaffold.SCAFFOLD)


def test_optional_stages_exclude_mandatory() -> None:
    assert scaffold.MANDATORY_STAGES
    assert scaffold.ALL_OPTIONAL_STAGES.isdisjoint(scaffold.MANDATORY_STAGES)
    # Optional set is exactly the legal flags minus the mandatory ones.
    legal = set(StageSelectionConfig.model_fields)
    assert scaffold.ALL_OPTIONAL_STAGES == frozenset(legal) - set(scaffold.MANDATORY_STAGES)
