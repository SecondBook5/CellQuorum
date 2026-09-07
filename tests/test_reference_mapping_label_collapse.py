"""Two label resolutions must come from ONE prediction, not two mappings.

scANVI is semi-supervised: the label set shapes the latent space and the classifier both, so a
coarse-trained model is a different model rather than another readout of the same one. Mapping
twice therefore costs two full trainings AND lets the two resolutions contradict each other —
a cell called ``LEC`` at coarse resolution and ``Fibroblast CCL19+`` at fine. Collapsing a fine
prediction to its parent cannot produce that, which is the property these tests pin.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.annotation.reference_mapping.scarches import (
    _collapse_labels,
    _label_hierarchy,
)


def _atlas(pairs: list[tuple[str, str]]) -> ad.AnnData:
    """A minimal reference carrying a fine and a coarse annotation column."""
    fine, coarse = zip(*pairs, strict=True)
    obs = pd.DataFrame({"fine": list(fine), "coarse": list(coarse)})
    obs.index = pd.Index([f"cell_{i}" for i in range(len(obs))])
    return ad.AnnData(X=np.zeros((len(obs), 2), dtype="float32"), obs=obs)


# ═══ The hierarchy must be learned, and validated ══════════════════════════════════


def test_a_clean_hierarchy_is_learned_from_the_atlas() -> None:
    """The map comes from the reference's own two columns, not from a hand-written table.

    Hard-coding 86 subtype-to-lineage pairs would go stale the first time the atlas is
    re-released, and would be wrong silently.
    """
    atlas = _atlas(
        [
            ("Capillary EC FABP4 hi", "VEC"),
            ("Venule EC ACKR1 hi", "VEC"),
            ("Fibroblast CCL19+", "Fibroblasts"),
            ("LEC", "LEC"),
        ]
    )
    assert _label_hierarchy(atlas, "fine", "coarse") == {
        "Capillary EC FABP4 hi": "VEC",
        "Venule EC ACKR1 hi": "VEC",
        "Fibroblast CCL19+": "Fibroblasts",
        "LEC": "LEC",
    }


def test_an_ambiguous_hierarchy_is_refused_rather_than_resolved() -> None:
    """A fine label under two parents has no collapse, and guessing would invent annotation."""
    atlas = _atlas(
        [
            ("Ambiguous EC", "VEC"),
            ("Ambiguous EC", "LEC"),
            ("Fibroblast CCL19+", "Fibroblasts"),
        ]
    )
    with pytest.raises(CellQuorumDataError, match="not a clean parent"):
        _label_hierarchy(atlas, "fine", "coarse")


def test_a_missing_column_names_what_is_available() -> None:
    """The atlas is someone else's file; the error has to say what it actually contains."""
    atlas = _atlas([("LEC", "LEC")])
    with pytest.raises(CellQuorumDataError, match="Available"):
        _label_hierarchy(atlas, "fine", "Cell_type_granular")


# ═══ The collapse itself ═══════════════════════════════════════════════════════════


def test_predictions_collapse_to_their_parents() -> None:
    hierarchy = {"Capillary EC FABP4 hi": "VEC", "Venule EC IL6+": "VEC", "LEC": "LEC"}
    coarse, unmapped = _collapse_labels(
        ["Capillary EC FABP4 hi", "LEC", "Venule EC IL6+"], hierarchy
    )
    assert coarse == ["VEC", "LEC", "VEC"]
    assert unmapped == set()


def test_confusion_inside_a_family_does_not_change_the_coarse_call() -> None:
    """The reason the derived column is more robust than the column it derives from.

    Mistaking one venule-EC subtype for a capillary-EC subtype is a granular error and a coarse
    non-event. That asymmetry is why collapsing beats a separately trained coarse model, which
    could get the lineage itself wrong.
    """
    hierarchy = {
        "Capillary EC FABP4 hi": "VEC",
        "Venule EC IL6+": "VEC",
        "Arterial EC IGFBP3+": "VEC",
    }
    truth = ["Venule EC IL6+"] * 3
    confused = ["Capillary EC FABP4 hi", "Arterial EC IGFBP3+", "Venule EC IL6+"]

    assert confused != truth, "the fine labels disagree"
    assert _collapse_labels(confused, hierarchy)[0] == _collapse_labels(truth, hierarchy)[0]


def test_an_unmapped_label_keeps_its_name_and_is_reported() -> None:
    """A cell must never be silently un-annotated.

    A predicted label absent from the hierarchy means the model produced something the atlas's
    coarse column does not cover. Blanking it would look like a QC exclusion; keeping the fine
    name is honest, and the caller warns.
    """
    coarse, unmapped = _collapse_labels(["LEC", "Something New"], {"LEC": "LEC"})
    assert coarse == ["LEC", "Something New"]
    assert unmapped == {"Something New"}


# ═══ Against the real atlas ════════════════════════════════════════════════════════


@pytest.mark.integration
def test_the_real_skin_atlas_has_a_clean_hierarchy() -> None:
    """The property this design depends on, checked against the actual reference.

    Marked integration because it needs the 2.3 GB atlas, which is named by an environment
    variable rather than a path: a literal made the suite depend on one machine's drive
    letter, so it skipped silently everywhere else while looking like coverage.

    If a future atlas release breaks the tree, `_label_hierarchy` raises at run time — but
    finding out here is cheaper than finding out after scANVI has trained.
    """
    from _external_data import require_external_file

    path = require_external_file(
        "CELLQUORUM_TEST_SKIN_ATLAS",
        what="the CellxGene atopic-dermatitis skin atlas .h5ad (Cell_type + "
        "Cell_type_granular columns)",
    )

    atlas = ad.read_h5ad(path, backed="r")
    hierarchy = _label_hierarchy(atlas, "Cell_type_granular", "Cell_type")

    assert len(hierarchy) == 86
    assert hierarchy["LEC"] == "LEC"
    # The four populations this cohort is being analysed for.
    parents = set(hierarchy.values())
    assert {"LEC", "VEC", "Fibroblasts", "Keratinocytes"} <= parents
    assert sum(1 for parent in hierarchy.values() if parent == "VEC") == 13
