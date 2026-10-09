from types import SimpleNamespace

from cellquorum.stages.cell_cell_communication.stage import CellCellCommunicationStage


def test_design_pairing_and_cohort_donor_are_inherited():
    ctx = SimpleNamespace(
        config=SimpleNamespace(
            design=SimpleNamespace(
                paired=True,
                donor_col="legacy_donor",
                condition_col="condition",
                case="LE",
                control="Normal",
            ),
            cohort=SimpleNamespace(
                donor_key="patient", sample_key="sample", condition_key="condition"
            ),
        )
    )
    cfg = CellCellCommunicationStage()._augment_config(ctx, {})
    assert cfg["paired"] is True
    assert cfg["donor_col"] == "patient"
    assert cfg["sample_col"] == "sample"
    assert cfg["case"] == "LE"


def test_explicit_unpaired_override_is_preserved():
    ctx = SimpleNamespace(
        config=SimpleNamespace(design=SimpleNamespace(paired=True, donor_col="donor"), cohort=None)
    )
    cfg = CellCellCommunicationStage()._augment_config(ctx, {"paired": False})
    assert cfg["paired"] is False
