"""The cluster-artifact audit, on a synthetic atlas whose artifacts are known by construction.

The fixture is built to contain the two clusters that make this problem hard rather than easy:
a genuinely low-RNA population (a neutrophil-shaped cluster: shallow, confidently annotated,
lineage-pure, spread across libraries) and a donor-private real population (one library, but
deep and confident). An audit that flags either of those would delete real cells, so the tests
assert on them as hard as on the debris cluster itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellquorum.stats.cluster_artifacts import (
    AMBIENT_DEBRIS,
    AUDIT_COLUMNS,
    LIBRARY_ARTIFACT,
    cluster_artifact_audit,
    debris_clusters,
    verify_declared_debris,
)

LINEAGES = ["LEC", "VEC", "Fibroblasts", "Keratinocytes", "T_NK", "Mast", "Neutrophils"]
LIBRARIES = [f"P{i}_{arm}" for i in range(1, 5) for arm in ("Normal", "LE")]

#: (cluster, n_cells, median genes, confidence, lineage rule, library rule)
#: ``lineage`` None means "one cell type per the cluster's own name"; a list means mixed.
CLEAN = "real_deep"
LOW_RNA = "real_neutrophil"
PRIVATE = "real_donor_private"
DEBRIS = "graveyard"
DOUBLETS = "real_doublet_ish"
TINY = "too_small"


def build_atlas(*, seed: int = 0) -> pd.DataFrame:
    """A per-cell frame with six clusters whose verdicts are decided in advance."""
    rng = np.random.default_rng(seed)
    blocks = []

    def block(cluster, n, genes, conf, lineages, libraries, *, radius):
        return pd.DataFrame(
            {
                "cluster": cluster,
                "n_genes": rng.normal(genes, genes * 0.1, n).clip(50),
                "confidence": rng.normal(conf, 0.02, n).clip(0.0, 1.0),
                "lineage": rng.choice(lineages, n),
                "library": rng.choice(libraries, n),
                "radius": rng.normal(radius, 1.0, n),
            }
        )

    # Four ordinary populations: deep, confident, pure, spread over every library.
    for name, lineage in [(CLEAN, "LEC"), (f"{CLEAN}_2", "Fibroblasts")]:
        blocks.append(block(name, 4000, 2400, 0.99, [lineage], LIBRARIES, radius=10.0))

    # Genuinely low-RNA biology. Shallow enough to trip the complexity mark on its own, and
    # nothing else. Flagging this as debris is the failure mode the conjunction exists to avoid.
    blocks.append(block(LOW_RNA, 900, 500, 0.97, ["Neutrophils"], LIBRARIES, radius=9.0))

    # A real population one donor happens to own. Library-dominated and otherwise pristine.
    blocks.append(block(PRIVATE, 700, 2300, 0.98, ["Mast"], ["P3_LE"], radius=12.0))

    # Doublet-ish / transitional: confident annotation fails and the lineage is mixed, but the
    # cells are deep and spread across libraries. A lead, not debris.
    blocks.append(
        block(DOUBLETS, 600, 2200, 0.55, ["LEC", "VEC", "Fibroblasts"], LIBRARIES, radius=8.0)
    )

    # The ambient graveyard: shallow, unassignable, every lineage at once, and mostly one
    # library — the conjunction, plus the dominance that says which library to blame.
    blocks.append(
        block(
            DEBRIS,
            1200,
            600,
            0.52,
            LINEAGES,
            ["P2_LE"] * 8 + LIBRARIES,
            radius=4.0,
        )
    )

    # Below min_cells: the audit should decline rather than guess from eleven medians.
    blocks.append(block(TINY, 11, 600, 0.4, LINEAGES, ["P2_LE"], radius=4.0))

    frame = pd.concat(blocks, ignore_index=True)
    frame["condition"] = np.where(frame["library"].str.endswith("_LE"), "Lymphedema", "Normal")
    return frame


def audit(frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """The audit over a fixture frame, indexed by cluster for readable assertions."""
    table = cluster_artifact_audit(
        frame["cluster"],
        complexity=frame["n_genes"],
        confidence=frame["confidence"],
        lineage=frame["lineage"],
        library=frame["library"],
        condition=frame["condition"],
        embedding=frame[["radius"]].to_numpy(),
        **kwargs,
    )
    return table.set_index("cluster")


@pytest.fixture
def atlas() -> pd.DataFrame:
    return build_atlas()


def test_the_ambient_conjunction_finds_the_graveyard_and_nothing_else(atlas):
    table = audit(atlas)

    debris = table.index[table["verdict"].isin([AMBIENT_DEBRIS, LIBRARY_ARTIFACT])].tolist()
    assert debris == [DEBRIS], table[["verdict", "marks"]].to_dict("index")
    assert table.loc[DEBRIS, "verdict"] == AMBIENT_DEBRIS
    assert table.loc[DEBRIS, "n_marks"] == 4
    assert table.loc[DEBRIS, "n_lineages"] == len(LINEAGES)
    assert table.loc[DEBRIS, "lineage_coverage"] == pytest.approx(1.0)


def test_a_genuinely_low_rna_lineage_is_a_lead_and_not_debris(atlas):
    """The neutrophil-shaped cluster is shallow and real. One mark, and it must stay one."""
    table = audit(atlas)
    row = table.loc[LOW_RNA]

    assert row["complexity_collapsed"]
    assert row["verdict"] == "low_complexity"
    assert row["n_marks"] == 1
    assert not row["confidence_collapsed"]
    assert not row["lineage_promiscuous"]
    assert LOW_RNA not in debris_clusters(table.reset_index())


def test_a_donor_private_population_is_named_not_condemned(atlas):
    """One library and nothing else wrong is a real population in a small cohort."""
    table = audit(atlas)
    row = table.loc[PRIVATE]

    assert row["library_dominated"]
    assert row["dominant_library"] == "P3_LE"
    assert row["verdict"] == "single_library", row["marks"]
    assert PRIVATE not in debris_clusters(table.reset_index())


def test_mixed_but_deep_is_an_annotation_problem_not_a_data_one(atlas):
    table = audit(atlas)
    row = table.loc[DOUBLETS]

    assert row["confidence_collapsed"] and row["lineage_promiscuous"]
    assert not row["complexity_collapsed"]
    assert row["verdict"] == "ambiguous_annotation"


def test_condition_dominance_is_reported_and_never_scored(atlas):
    """A cluster that is one arm is either the finding or the artifact; the audit says so.

    The graveyard is overwhelmingly one condition here purely because it is overwhelmingly one
    (case) library. The verdict must be reachable without the condition column at all, or the
    audit is capable of deleting a disease-specific population for being one.
    """
    table = audit(atlas)
    assert table.loc[DEBRIS, "dominant_condition"] == "Lymphedema"
    assert table.loc[DEBRIS, "dominant_condition_frac"] > 0.7

    without = cluster_artifact_audit(
        atlas["cluster"],
        complexity=atlas["n_genes"],
        confidence=atlas["confidence"],
        lineage=atlas["lineage"],
        library=atlas["library"],
    ).set_index("cluster")
    assert (without["verdict"] == table["verdict"].reindex(without.index)).all()


def test_embedding_position_is_reported_and_never_scored(atlas):
    """Moving the graveyard to the rim of the embedding must not change any verdict."""
    moved = atlas.copy()
    moved.loc[moved["cluster"] == DEBRIS, "radius"] += 40.0

    before, after = audit(atlas), audit(moved)
    assert (before["verdict"] == after["verdict"]).all()
    column = "embedding_radius_rank_pct"
    assert after.loc[DEBRIS, column] > before.loc[DEBRIS, column]


def test_a_cluster_too_small_to_judge_is_declined(atlas):
    table = audit(atlas)
    assert table.loc[TINY, "verdict"] == "insufficient_cells"
    assert TINY not in debris_clusters(table.reset_index())


def test_without_a_confidence_column_nothing_reaches_the_debris_verdict(atlas):
    """Missing evidence must not be silently read as absent evidence.

    Two of the three intrinsic marks are unavailable without an annotation confidence, so no
    cluster can satisfy the conjunction. The audit has to say that in ``marks`` rather than
    returning a table of ``clean`` verdicts that reads like a clean atlas.
    """
    table = cluster_artifact_audit(
        atlas["cluster"],
        complexity=atlas["n_genes"],
        lineage=atlas["lineage"],
        library=atlas["library"],
    ).set_index("cluster")

    assert AMBIENT_DEBRIS not in set(table["verdict"])
    assert table.loc[DEBRIS, "verdict"] == LIBRARY_ARTIFACT
    assert all("no confidence column" in marks for marks in table["marks"])
    assert table["median_confidence"].isna().all()


def test_thresholds_are_arguments_and_the_measured_values_travel_with_them(atlas):
    """Every boolean in the table is reproducible from a number in the same row."""
    table = audit(atlas, collapse_ratio=0.05)
    assert not table["complexity_collapsed"].any()
    assert table.loc[DEBRIS, "verdict"] != AMBIENT_DEBRIS

    strict = audit(atlas, confidence_floor=0.999)
    assert strict["confidence_collapsed"].all()

    for _, row in audit(atlas).iterrows():
        assert row["complexity_collapsed"] == (row["complexity_ratio"] < 0.5)
        assert row["library_dominated"] == (row["dominant_library_frac"] > 0.5)


def test_the_table_leads_with_what_needs_looking_at(atlas):
    table = cluster_artifact_audit(
        atlas["cluster"], complexity=atlas["n_genes"], confidence=atlas["confidence"]
    )
    assert list(table.columns) == AUDIT_COLUMNS
    assert table.iloc[0]["verdict"] == "insufficient_cells"
    assert table.iloc[-1]["verdict"] == "clean"


# ---------------------------------------------------------------------------
# the cross-partition rule


def test_a_mask_written_against_another_partition_stops_the_run(atlas):
    """The whole point. An id that is not a cluster here is a mask from somewhere else."""
    table = audit(atlas).reset_index()

    with pytest.raises(ValueError, match="not clusters of this partition"):
        verify_declared_debris(table, ["18", "30", "40"])

    comparison = verify_declared_debris(table, ["18", DEBRIS], strict=False)
    absent = comparison.set_index("cluster").loc["18"]
    assert absent["verdict"] == "absent"
    assert not absent["agrees"]
    assert absent["n_cells"] == 0


def test_masking_a_clean_cluster_is_surfaced_as_a_disagreement(atlas):
    """Applying the wrong ids to the right object deletes real cells; say how many."""
    table = audit(atlas).reset_index()
    comparison = verify_declared_debris(table, [CLEAN, DEBRIS]).set_index("cluster")

    assert comparison.loc[CLEAN, "declared"] and not comparison.loc[CLEAN, "audited"]
    assert not comparison.loc[CLEAN, "agrees"]
    assert comparison.loc[CLEAN, "n_cells"] == 4000
    assert comparison.loc[DEBRIS, "agrees"]


def test_leaving_the_real_artifact_out_of_the_mask_is_surfaced_too(atlas):
    """A mask that misses the debris is worse than no mask: it reads as having handled it."""
    table = audit(atlas).reset_index()
    comparison = verify_declared_debris(table, []).set_index("cluster")

    assert comparison.loc[DEBRIS, "audited"] and not comparison.loc[DEBRIS, "declared"]
    assert not comparison.loc[DEBRIS, "agrees"]


def test_an_agreeing_mask_agrees_on_every_row(atlas):
    table = audit(atlas).reset_index()
    comparison = verify_declared_debris(table, debris_clusters(table))
    assert comparison["agrees"].all()
    assert len(comparison) == 1


def test_unlabelled_cells_are_not_a_cluster(atlas):
    frame = atlas.copy()
    frame.loc[frame.index[:50], "cluster"] = None
    table = audit(frame)

    assert "None" not in table.index and "nan" not in table.index
    assert int(table["n_cells"].sum()) == len(frame) - 50

    with pytest.raises(ValueError, match="no cell carries a usable cluster label"):
        cluster_artifact_audit(pd.Series([None] * 5), complexity=pd.Series([1.0] * 5))


def test_misaligned_inputs_are_refused_rather_than_broadcast(atlas):
    with pytest.raises(ValueError, match="complexity has"):
        cluster_artifact_audit(atlas["cluster"], complexity=atlas["n_genes"].iloc[:-1])
    with pytest.raises(ValueError, match="confidence has"):
        cluster_artifact_audit(
            atlas["cluster"],
            complexity=atlas["n_genes"],
            confidence=atlas["confidence"].iloc[:-1],
        )
    with pytest.raises(ValueError, match="embedding has"):
        cluster_artifact_audit(
            atlas["cluster"],
            complexity=atlas["n_genes"],
            embedding=atlas[["radius"]].to_numpy()[:-1],
        )
