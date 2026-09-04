"""Synthetic-fixture spec for the ``module_remodeling`` comparative stage.

``state_scoring`` already produces per-cell module activity — ``obs["state_<program>"]``
and ``obsm["X_state_aucell"]`` — and stops there: **it emits tables and zero figures**.
So a run that scores 11 curated modules across a subtype axis currently ends with the
module activity computed and nothing that shows it. ``cellquorum.stats`` supplies the
statistics (donor-aware LMM effect sizes, PERMANOVA, signature-argmax subtyping, the
signed contrast index, concordance, correlations) as plain numpy/pandas functions, and
nothing in the engine calls them. This file specifies the stage that closes that gap:
scores in, statistics out, figures on disk.

Everything here is built inline from numpy/pandas — no real data, no R, no skip
markers, so these always run and never trip the real-data skipif leak. The fixture is
deliberately *not* endothelial: programs are ``prog_a…prog_e`` and groups are ``g0/g1``,
because the reusability boundary is part of the contract (spec §8 — the engine stage is
generic; the 11 endothelial modules and the "LEC Module Remodeling" title live in the
hypothesis repo's config). One test enforces that boundary structurally.
"""

from __future__ import annotations

import ast
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.backends.registry import build_default_backend_registry
from cellquorum.config.models import CellQuorumConfig, StageSelectionConfig
from cellquorum.core.stages import all_stage_specs
from cellquorum.methods.base import MethodSkip

PROGRAMS = ["prog_a", "prog_b", "prog_c", "prog_d", "prog_e"]

# The planted truth: prog_a rises with disease in group g0 only; prog_b falls
# everywhere; prog_c/d/e are null. Recovering exactly this is the statistical test.
PLANTED_UP = ("g0", "prog_a")
PLANTED_DOWN = "prog_b"

# Condition-independent identity of each cluster, which is what the signature
# labelling reads. Without it c1 has no elevated marker at all and its argmax is
# decided by noise, so the labelling assertion would pass or fail by seed —
# a flaky test masquerading as a contract.
CLUSTER_IDENTITY = {"c0": "prog_a", "c1": "prog_c"}
IDENTITY_LIFT = 1.2


@pytest.fixture
def scored_adata():
    """A paired cohort carrying a ``state_scoring``-shaped score matrix.

    Six donors, each contributing cells to both arms (the paired design the LEC
    manuscript runs), split across two clusters that map onto two groups. The
    score matrix lives in ``obsm`` with its column order in ``uns``, exactly as
    ``state_scoring``'s AUCell path writes it — the stage must read it from there
    rather than from a hardcoded column list.
    """
    rng = np.random.default_rng(11)
    rows, scores = [], []
    for donor in [f"D{i}" for i in range(6)]:
        donor_offset = rng.normal(0.0, 0.3, size=len(PROGRAMS))
        for condition in ("Normal", "Lymphedema"):
            for cluster in ("c0", "c1"):
                group = "g0" if cluster == "c0" else "g1"
                identity = PROGRAMS.index(CLUSTER_IDENTITY[cluster])
                for _ in range(25):
                    v = donor_offset + rng.normal(0.0, 0.25, size=len(PROGRAMS))
                    # Cluster identity is present in both arms, so it labels the
                    # cluster without contributing to any condition contrast.
                    v[identity] += IDENTITY_LIFT
                    if condition == "Lymphedema":
                        if group == PLANTED_UP[0]:
                            v[PROGRAMS.index(PLANTED_UP[1])] += 1.6
                        v[PROGRAMS.index(PLANTED_DOWN)] -= 1.6
                    scores.append(v)
                    rows.append(
                        {
                            "donor_id": donor,
                            "condition": condition,
                            "sample_id": f"{donor}_{condition}",
                            "cluster": cluster,
                        }
                    )
    obs = pd.DataFrame(rows)
    obs.index = [f"cell{i}" for i in range(len(obs))]
    matrix = np.vstack(scores)

    adata = ad.AnnData(X=np.zeros((len(obs), 4), dtype=np.float32), obs=obs)
    adata.obsm["X_state_aucell"] = matrix
    adata.uns["state_aucell"] = {"programs": list(PROGRAMS)}
    return adata


@pytest.fixture
def mock_context(tmp_path):
    """Stage context with the four run directories and a backend registry."""

    class Paths:
        root = tmp_path
        scratch = tmp_path / "scratch"
        results = tmp_path / "results"
        figures = tmp_path / "figures"

    class Context:
        paths = Paths()
        backend_registry = build_default_backend_registry()

    return Context()


def _config(**overrides):
    """The generic config: no endothelial biology, only column names + design."""
    config = {
        "score_key": "X_state_aucell",
        "programs_uns_key": "state_aucell",
        "cluster_col": "cluster",
        "donor_col": "donor_id",
        "condition_col": "condition",
        "sample_col": "sample_id",
        "case": "Lymphedema",
        "control": "Normal",
        "paired": True,
        # Two synthetic "subtypes", each defined by the programs that mark it —
        # the generic form of CV / Structural / Inflamed.
        "subtype_signatures": {"g0": ["prog_a"], "g1": ["prog_c"]},
        "module_categories": {
            "axis": ["prog_a", "prog_b"],
            "other": ["prog_c", "prog_d", "prog_e"],
        },
        "index_up": ["prog_a"],
        "index_down": ["prog_b"],
        "key_added": "module_subtype",
        "index_key": "contrast_index",
        "seed": 1337,
        "n_permutations": 199,
    }
    config.update(overrides)
    return config


def _run(adata, context, **overrides):
    from cellquorum.stages.comparative.module_remodeling.remodeling_method import (
        ModuleRemodelingMethod,
    )

    return ModuleRemodelingMethod().run(adata, _config(**overrides), context)


def _table(result, name: str) -> pd.DataFrame:
    artifact = next(a for a in result.artifacts if a.name == name)
    return pd.read_csv(artifact.path)


# --------------------------------------------------------------------------- #
# registration                                                                 #
# --------------------------------------------------------------------------- #
def test_module_remodeling_is_a_registered_stage_with_a_config_block():
    """A stage nobody can switch on is not a stage.

    The three couplings that make it reachable from a manifest: a catalog entry,
    a ``stages:`` toggle, and a config sub-block.
    """
    spec = next((s for s in all_stage_specs() if s.name == "module_remodeling"), None)
    assert spec is not None, "module_remodeling missing from the stage catalog"
    assert spec.is_implemented, "registered but has no factory"
    assert spec.config_flag in StageSelectionConfig.model_fields
    assert spec.config_field in CellQuorumConfig.model_fields


def test_every_config_key_the_stage_reads_is_a_field_of_its_config_model():
    """A key the model does not declare can never arrive, and never says so.

    ``ModuleRemodelingConfig`` is strict, so a manifest setting a key the model has
    not declared is rejected loudly — but the reverse is silent: the method calling
    ``config.get("min_depth_pairs")`` against a model that only has
    ``min_depth_pair`` reads ``None`` forever and quietly uses the default, on every
    dataset, with nothing in the output saying the setting was ignored. Deriving the
    expected key list from the source rather than restating it means adding a knob
    to the method and forgetting the model is a failing test, not a support ticket.
    """
    package = (
        Path(__file__).resolve().parents[1] / "src/cellquorum/stages/comparative/module_remodeling"
    )
    from cellquorum.stages.comparative.module_remodeling.config import ModuleRemodelingConfig

    declared = set(ModuleRemodelingConfig.model_fields)
    read: dict[str, str] = {}
    for module in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "config"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                read[node.args[0].value] = f"{module.name}:{node.lineno}"

    undeclared = {key: where for key, where in read.items() if key not in declared}
    assert not undeclared, "config keys the model does not declare (they will always be None): " + (
        ", ".join(f"{key} at {where}" for key, where in sorted(undeclared.items()))
    )


def test_module_remodeling_runs_after_the_tables_it_consumes():
    """It reads DE, GSEA, state scores and the subclustering partition.

    Ordering is the only thing that guarantees those exist when it runs, so it is
    part of the contract rather than an implementation detail.
    """
    orders = {s.name: s.order for s in all_stage_specs()}
    for upstream in ("state_scoring", "subclustering", "differential_expression", "enrichment"):
        assert orders["module_remodeling"] > orders[upstream], f"must run after {upstream}"


def test_module_remodeling_reads_the_design_the_project_already_declares():
    """The design is declared once, in ``design:``/``cohort:``, not restated here.

    Every other comparative stage bridges the project-level design block into its
    own config, so a manifest names the donor column, the condition column and the
    case/control tokens exactly once. Without that bridge this stage would demand a
    second copy under ``module_remodeling:`` — and a second copy is a second thing
    that can disagree with the first, which is how a stage ends up silently testing
    a different contrast from the DE table beside it.
    """
    from cellquorum.core.stages import all_stage_specs as _specs

    spec = next(s for s in _specs() if s.name == "module_remodeling")
    stage = spec.factory()

    config = CellQuorumConfig.model_validate(
        {
            "design": {
                "donor_col": "donor_id",
                "condition_col": "condition",
                "case": "Lymphedema",
                "control": "Normal",
                "paired": True,
            },
            "cohort": {"sample_key": "sample_id", "donor_key": "donor_id"},
        }
    )

    class Ctx:
        pass

    ctx = Ctx()
    ctx.config = config

    augmented = stage._augment_config(ctx, {})
    assert augmented["donor_col"] == "donor_id"
    assert augmented["condition_col"] == "condition"
    assert augmented["sample_col"] == "sample_id"
    assert augmented["case"] == "Lymphedema"
    assert augmented["control"] == "Normal"
    assert augmented["paired"] is True

    # An explicit stage value still wins: a study that scores modules on a
    # different axis from its primary contrast must be able to say so.
    override = stage._augment_config(ctx, {"condition_col": "timepoint", "case": "week12"})
    assert override["condition_col"] == "timepoint"
    assert override["case"] == "week12"
    assert override["control"] == "Normal", "only the overridden keys change"


# --------------------------------------------------------------------------- #
# statistics                                                                   #
# --------------------------------------------------------------------------- #
def test_module_remodeling_recovers_the_planted_group_specific_shift(scored_adata, mock_context):
    """The end-to-end statistical claim: right module, right group, right sign.

    prog_a was planted up in g0 only and prog_b down in both. A group-specific
    effect surviving FDR in one group and not the other is precisely the
    "is this remodeling subtype-restricted?" question the flagship figure asks,
    so the stage has to get it right before the figure means anything.
    """
    result = _run(scored_adata, mock_context)
    assert not isinstance(result, MethodSkip), getattr(result, "reason", result)

    effects = _table(result, "mr_effect_sizes").set_index(["group", "program"])
    for column in ("effect", "ci_low", "ci_high", "p_value", "fdr", "n_donors", "method"):
        assert column in effects.columns, f"house bar requires {column} on every row"

    up = effects.loc[PLANTED_UP]
    assert up["effect"] > 0.5 and up["fdr"] < 0.05
    assert effects.loc[("g1", "prog_a")]["fdr"] > 0.05, "planted g0-only effect leaked into g1"

    for group in ("g0", "g1"):
        assert effects.loc[(group, PLANTED_DOWN)]["effect"] < -0.5

    assert (effects["method"] == "lmm").all(), "a paired 6-donor design must fit the mixed model"


def test_module_remodeling_reports_a_multivariate_effect_per_group(scored_adata, mock_context):
    """PERMANOVA on per-sample module vectors, seeded and recorded."""
    result = _run(scored_adata, mock_context)
    permanova = _table(result, "mr_permanova")
    assert set(permanova["group"]) == {"g0", "g1"}
    for column in ("pseudo_F", "R2", "p_value", "n_perm", "seed"):
        assert column in permanova.columns
    assert (permanova["seed"] == 1337).all(), "the seed must be recorded, not just used"
    assert (permanova["R2"] > 0.0).all()


def test_module_remodeling_is_deterministic_under_a_fixed_seed(scored_adata, mock_context):
    """Two runs of the same config produce byte-identical statistics."""
    first = _table(_run(scored_adata, mock_context), "mr_permanova")
    second = _table(_run(scored_adata, mock_context), "mr_permanova")
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------- #
# obs additions + subtype derivation                                           #
# --------------------------------------------------------------------------- #
def test_module_remodeling_labels_clusters_and_writes_the_axis_to_obs(scored_adata, mock_context):
    """Subtype labels and the contrast index become obs columns, not just CSVs.

    Downstream figures (UMAP overlays, ridgelines) read obs, and the checkpoint
    carries obs forward; a label that exists only in a CSV cannot be plotted on
    an embedding.
    """
    result = _run(scored_adata, mock_context)
    adata = result.adata

    assert "module_subtype" in adata.obs.columns
    assert "contrast_index" in adata.obs.columns
    assert adata.obs["contrast_index"].notna().all()
    # prog_a marks g0 and prog_c marks g1, so the two clusters must not collapse
    # onto one label — an argmax that assigns everything to one subtype is the
    # failure mode the ambiguity guard exists to make visible.
    assigned = set(adata.obs["module_subtype"].unique()) - {"unassigned"}
    assert len(assigned) == 2, f"clusters did not separate: {assigned}"

    per_cluster = adata.obs.groupby("cluster", observed=True)["module_subtype"].nunique()
    assert (per_cluster == 1).all(), "a cluster-level label must be constant within its cluster"

    index_by_condition = adata.obs.groupby("condition", observed=True)["contrast_index"].mean()
    assert (
        index_by_condition["Lymphedema"] > index_by_condition["Normal"]
    ), "index_up rose and index_down fell in disease; the signed index must follow"


def test_module_remodeling_records_an_ambiguous_cluster_as_unassigned(scored_adata, mock_context):
    """A margin threshold nothing can clear must yield labels, not a crash.

    ``min_margin`` above any achievable separation forces every cluster to be
    undecidable. The recorded answer is ``unassigned`` for all of them — an
    explicit "cannot tell", which is what keeps a borderline subtype out of the
    statistics instead of silently into one arm of it.
    """
    result = _run(scored_adata, mock_context, min_margin=1e6)
    assert not isinstance(result, MethodSkip), getattr(result, "reason", result)
    assert set(result.adata.obs["module_subtype"].unique()) == {"unassigned"}
    assert any("unassigned" in note for note in result.notes), "silence would hide the guard"


# --------------------------------------------------------------------------- #
# figures — the actual gap                                                     #
# --------------------------------------------------------------------------- #
def test_module_remodeling_ships_the_flagship_dot_grid(scored_adata, mock_context):
    """The figure that does not exist anywhere in the engine today.

    Module (rows, grouped by category) x group (columns) dot-grid: colour is the
    LMM disease-minus-control effect, size is -log10(FDR). ``state_scoring``
    computes the module activity and draws nothing, so this is the only place the
    curated modules become a panel. It is the manuscript's lead figure, which is
    why it is asserted as a stage artifact and not left to a repo-level script:
    a figure produced by hand once is a figure that silently stops matching the
    tables next to it.
    """
    result = _run(scored_adata, mock_context)
    figures = [a for a in result.artifacts if a.kind in {"figure", "png", "pdf"}]
    assert figures, "the stage emitted no figure at all"

    flagship = next((a for a in figures if "dotgrid" in a.path.name), None)
    assert flagship is not None, f"no dot-grid among {[a.path.name for a in figures]}"
    assert flagship.path.exists() and flagship.path.stat().st_size > 0
    # The house writer's vector twin, so the panel is editable for the figure file.
    assert flagship.path.with_suffix(".pdf").exists()


def test_module_remodeling_names_the_axes_a_reader_has_to_read(
    scored_adata, mock_context, monkeypatch
):
    """A column of bare cluster numbers over an unnamed axis is not a finished panel.

    On the real arm the flagship's columns came out as 1..8 with nothing anywhere on
    the figure saying they are subclusters, and its rows printed the obs column
    names — ``integrin_focal_adhesion`` is an identifier, not a row label. Both are
    the study's wording, so both arrive as config: the engine's own default is the
    column name with its underscores opened up, never an invented display name.
    """
    from cellquorum.stages.comparative.module_remodeling import mr_figures

    captured: dict = {}
    real = mr_figures.plot_module_dotgrid

    def spy(effects, **kwargs):
        captured.update(kwargs)
        return real(effects, **kwargs)

    monkeypatch.setattr(mr_figures, "plot_module_dotgrid", spy)

    labels = {"prog_a": "Programme A (marker)"}
    result = _run(
        scored_adata,
        mock_context,
        group_label="Synthetic subtype",
        program_labels=labels,
    )
    assert result.artifacts, f"stage produced nothing: {result.notes}"
    assert captured.get("group_label") == "Synthetic subtype"
    assert captured.get("program_labels") == labels

    captured.clear()
    _run(scored_adata, mock_context)
    # Default: the group column, made readable. Not blank — an unlabelled axis is
    # the defect — and not guessed at either.
    assert captured.get("group_label") == "module subtype"


def test_program_labels_fall_back_to_the_column_name_made_readable():
    """The display rule both program-labelled panels share.

    A mapping the study supplies wins; anything it omits still renders, with
    underscores opened up rather than printed as an identifier.
    """
    from cellquorum.stages.comparative.module_remodeling import mr_figures

    assert mr_figures.display_labels(
        ["endomt_lec", "protective_flow"], {"endomt_lec": "EndoMT (LEC)"}
    ) == ["EndoMT (LEC)", "protective flow"]


def test_module_remodeling_figures_recompute_nothing(scored_adata, mock_context):
    """Every figure has a table beside it that fully determines it.

    The replot rule: run directories are the rendering source, so a figure must
    be reproducible from the CSVs without re-running the statistics. Concretely
    — each figure's underlying numbers ship as a table artifact.
    """
    result = _run(scored_adata, mock_context)
    names = {a.name for a in result.artifacts}
    assert {
        "mr_effect_sizes",
        "mr_permanova",
        "mr_correlation",
        "mr_correlation_tests",
        "mr_subtypes",
    } <= names


def test_module_remodeling_emits_the_correlation_and_index_tables(scored_adata, mock_context):
    result = _run(scored_adata, mock_context)
    corr = _table(result, "mr_correlation")
    # Square, symmetric, unit diagonal over the five programs.
    corr = corr.set_index(corr.columns[0])
    assert list(corr.columns) == PROGRAMS and list(corr.index) == PROGRAMS
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)

    subtypes = _table(result, "mr_subtypes")
    assert {"cluster", "label", "margin"} <= set(subtypes.columns)


# --------------------------------------------------------------------------- #
# the correlation table's unit                                                 #
# --------------------------------------------------------------------------- #
def test_the_testable_correlation_is_at_the_sample_level_not_the_cell_level(
    scored_adata, mock_context
):
    """The defect the second table exists to fix.

    ``module_correlation.csv`` is ``scores.corr()`` over every cell, so any p-value
    attached to it would use 600 cells as 600 independent observations of 12
    samples. The tested version has to say so: the unit is named in the frame, and
    ``n_units`` counts samples. A table that reports the cell count here is
    pseudoreplicating whatever it says in its header.
    """
    result = _run(scored_adata, mock_context)
    tests = _table(result, "mr_correlation_tests")

    n_samples = scored_adata.obs["sample_id"].nunique()
    assert n_samples < scored_adata.n_obs, "fixture cannot distinguish the two units"
    assert set(tests["unit"].astype(str)) == {"sample_id"}
    assert set(tests["n_units"].astype(int)) == {n_samples}

    # Every unordered pair once — not the full square, and not the diagonal.
    n_programs = len(PROGRAMS)
    assert len(tests) == n_programs * (n_programs - 1) // 2
    assert not (tests["program_a"] == tests["program_b"]).any()
    assert {"r", "p_value", "fdr", "r_adjusted", "fdr_adjusted", "shared_genes"} <= set(
        tests.columns
    )


def test_the_correlation_table_follows_the_same_program_order_as_the_matrix(
    scored_adata, mock_context
):
    """Both tables are replot sources for panels sharing one axis, so both are ordered.

    ``module_categories`` puts prog_a/prog_b before prog_c/d/e; a long-form table in
    groupby order would draw its rows in a different order from the matrix's axes.
    """
    result = _run(scored_adata, mock_context)
    tests = _table(result, "mr_correlation_tests")
    expected = ["prog_a", "prog_b", "prog_c", "prog_d"]
    assert list(dict.fromkeys(tests["program_a"].astype(str))) == expected
    # prog_a-major, with program_b in the shared order inside the first block.
    first_block = tests.loc[tests["program_a"].astype(str) == "prog_a", "program_b"]
    assert list(first_block.astype(str)) == ["prog_b", "prog_c", "prog_d", "prog_e"]


def test_no_sample_column_warns_that_cells_are_not_independent(scored_adata, mock_context):
    """Degrading to the cell level is allowed; doing it silently is not.

    Not every study configures a sample column, so the table is still written — but
    a reader who cites its p-values has to be told they are anticonservative, and
    the warning is where that lands in the run summary. (A *wrong* column name is a
    different failure: the input contract rejects it before the stage runs.)
    """
    result = _run(scored_adata, mock_context, sample_col=None)
    assert not isinstance(result, MethodSkip), getattr(result, "reason", result)
    joined = "\n".join(result.warnings)
    assert "cell level" in joined
    assert "anticonservative" in joined

    tests = _table(result, "mr_correlation_tests")
    assert set(tests["unit"].astype(str)) == {"row"}
    assert set(tests["n_units"].astype(int)) == {scored_adata.n_obs}


def test_the_shared_gene_count_travels_from_the_scoring_stage(scored_adata, mock_context):
    """Programs that share genes correlate arithmetically, so the count has to ship.

    ``state_scoring`` records the genes each program was actually scored on; this
    stage reads them so a reader can tell an r of 0.83 between two lists sharing
    seven genes from an r of 0.83 between two disjoint ones.
    """
    adata = scored_adata.copy()
    adata.uns["state_aucell"] = {
        "programs": list(PROGRAMS),
        "genes": {
            "prog_a": ["VIM", "FN1", "ACTA2"],
            "prog_b": ["VIM", "FN1"],  # 2 shared with prog_a
            "prog_c": ["CLDN5"],
            "prog_d": ["PROX1"],
            "prog_e": ["KLF2"],
        },
    }
    tests = _table(_run(adata, mock_context), "mr_correlation_tests")
    by_pair = {
        (str(row.program_a), str(row.program_b)): (int(row.shared_genes), bool(row.shares_genes))
        for row in tests.itertuples()
    }
    assert by_pair[("prog_a", "prog_b")] == (2, True)
    assert by_pair[("prog_a", "prog_c")] == (0, False)


def test_absent_gene_lists_mark_the_overlap_unknown_rather_than_zero(scored_adata, mock_context):
    """ "We did not check" must not render as "the programs are disjoint".

    The fixture's ``uns`` has no gene lists, which is the state of every run made
    before the scoring stage recorded them. The sentinel keeps that distinguishable
    from a checked, genuinely empty intersection.
    """
    tests = _table(_run(scored_adata, mock_context), "mr_correlation_tests")
    assert set(tests["shared_genes"].astype(int)) == {-1}
    assert not tests["shares_genes"].any()


def test_module_remodeling_tests_the_contrast_index_it_plots(
    scored_adata, mock_context, monkeypatch
):
    """The index panel makes a claim, so the claim ships tested and named.

    Two violins per group with nothing else on the panel asks the reader to
    eyeball an overlap and call it a shift. The test is the same machinery the
    module effects use — condition fixed effect, donor random intercept, BH across
    the groups — so the index cannot disagree with the modules it is built from
    about what a shift is. And the y axis says which index: "signed contrast
    index" describes the formula, not the quantity, and every study's is different.
    """
    from cellquorum.stages.comparative.module_remodeling import mr_figures

    captured: dict = {}
    real = mr_figures.plot_contrast_index

    def spy(values, **kwargs):
        captured.update(kwargs)
        return real(values, **kwargs)

    monkeypatch.setattr(mr_figures, "plot_contrast_index", spy)
    result = _run(scored_adata, mock_context)

    table = _table(result, "mr_index_effects")
    assert set(table["group"].astype(str)) == {"g0", "g1"}
    assert {"effect", "ci_low", "ci_high", "p_value", "fdr", "method", "n_donors"} <= set(
        table.columns
    )
    # The planted shift is on prog_a, which is the index's up side, so the group
    # carrying it must come out positive rather than merely "tested".
    planted = table.loc[table["group"].astype(str) == "g0", "effect"]
    assert float(planted.iloc[0]) > 0, f"index effect not recovered: {table}"

    # The panel is annotated from that table, not from a second computation.
    # ``approx`` because the comparison crosses a CSV: pandas' C parser is a bit-off
    # 1 ulp on values like 2.8424540814948263e-08, so exact equality here tests the
    # parser rather than the stage. A second computation would not agree to 1e-12.
    assert captured.get("significance") == pytest.approx(
        {str(row.group): float(row.fdr) for row in table.itertuples()}, rel=1e-12
    )
    # Default label: the index's own obs key, made readable.
    assert captured.get("index_label") == "contrast index"


# --------------------------------------------------------------------------- #
# loud skips                                                                   #
# --------------------------------------------------------------------------- #
def test_module_remodeling_orders_groups_and_modules_the_way_the_figure_reads(
    scored_adata, mock_context
):
    """Column and row order are part of the figure, not a byproduct of a groupby.

    A cluster partition arrives as numbers, and the statistics return groups in
    whatever order the grouping produced — on the LEC arm that was 1, 3, 6, 2, 4,
    8, 5, 7. Columns in that order make a reader hunt for cluster 2, and a
    lexicographic fix is no better once there are ten clusters (1, 10, 2, ...).
    Rows follow the configured categories for the same reason: the table is the
    replot source, so it has to carry the order the panel is drawn in.
    """
    adata = scored_adata.copy()
    # Ten numeric clusters assigned in a scrambled order, so all three candidate
    # orders differ: order of appearance is the permutation below, lexicographic is
    # 1, 10, 2, ..., and only natural order is 1..10. A fixture whose cells happen
    # to appear in cluster order cannot tell those apart and passes either way.
    appearance = [3, 10, 1, 7, 2, 9, 4, 8, 5, 6]
    codes = np.array([appearance[i % 10] for i in range(adata.n_obs)], dtype=float)
    adata.obs["choir"] = codes

    result = _run(adata, mock_context, group_col="choir", cluster_col=None)
    assert not isinstance(result, MethodSkip), getattr(result, "reason", result)

    effects = _table(result, "mr_effect_sizes")
    groups = list(dict.fromkeys(effects["group"].astype(str)))
    assert groups == [str(i) for i in range(1, 11)], f"columns out of order: {groups}"

    permanova = _table(result, "mr_permanova")
    assert list(dict.fromkeys(permanova["group"].astype(str))) == groups

    # Row order follows module_categories: the axis pair first, then the rest.
    programs = list(dict.fromkeys(effects["program"]))
    assert programs == ["prog_a", "prog_b", "prog_c", "prog_d", "prog_e"]


def test_module_remodeling_honours_an_explicit_group_order(scored_adata, mock_context):
    """A study that has a meaningful group order can declare it.

    Natural order is only the right default. Once the clusters are named subtypes,
    the defensible column order is biological (capillary before collecting, say),
    and that is a config statement, not something the engine can infer.
    """
    result = _run(scored_adata, mock_context, group_order=["g1", "g0"])
    effects = _table(result, "mr_effect_sizes")
    assert list(dict.fromkeys(effects["group"].astype(str))) == ["g1", "g0"]


def test_module_remodeling_does_not_invent_a_group_from_unassigned_cells(
    scored_adata, mock_context
):
    """Cells the partition left unassigned are not a subtype.

    A real CHOIR partition is a float column with ``NaN`` for the cells it could
    not place — 8 of 1864 on the LEC arm. ``astype(str)`` turns those into the
    string ``"nan"``, which then travels the whole way through as a group: an
    eleventh column of crosses on the flagship figure, an all-``NaN`` PERMANOVA
    row, and eleven rows of the effect table spent on it. It reads as a subtype
    with no detectable biology, which is a claim about the data rather than an
    artifact of the column's dtype.
    """
    adata = scored_adata.copy()
    partition = pd.Series(
        np.where(adata.obs["cluster"].to_numpy() == "c0", 1.0, 2.0),
        index=adata.obs_names,
        dtype=float,
    )
    partition.iloc[:7] = np.nan  # the cells CHOIR could not place
    adata.obs["choir"] = partition

    result = _run(adata, mock_context, group_col="choir", cluster_col=None)
    assert not isinstance(result, MethodSkip), getattr(result, "reason", result)

    effects = _table(result, "mr_effect_sizes")
    groups = {str(g) for g in effects["group"]}
    assert groups == {"1", "2"}, f"phantom group in the effect table: {groups}"
    # "1.0" is not a subtype name. A float column read straight into a manuscript
    # figure's column headers is the same defect one step less visible.
    assert not any("." in g or g.lower() == "nan" for g in groups)

    permanova = _table(result, "mr_permanova")
    assert {str(g) for g in permanova["group"]} == {"1", "2"}
    assert permanova["R2"].notna().all(), "an all-NaN row is a group that should not exist"

    assert any(
        "7" in note and "unassigned" in note.lower() for note in result.notes
    ), f"dropping cells silently is the other half of the bug: {result.notes}"


def test_a_run_that_labels_nothing_writes_no_label_table(scored_adata, mock_context):
    """An empty CSV is not a record of "no labelling happened".

    Grouping on an existing obs column skips the naming step entirely, and the
    stage used to write ``module_subtypes.csv`` anyway — zero rows, and three of
    the real table's six columns. From the run directory alone that was
    indistinguishable from a labelling run that named nothing, which is a very
    different fact, and any reader reaching for ``top_z`` hit a KeyError. The LEC
    arm shipped exactly that file.
    """
    adata = scored_adata.copy()
    adata.obs["choir"] = np.where(adata.obs["cluster"].to_numpy() == "c0", 1.0, 2.0)

    result = _run(adata, mock_context, group_col="choir", cluster_col=None)
    assert not isinstance(result, MethodSkip), getattr(result, "reason", result)

    assert "mr_subtypes" not in {a.name for a in result.artifacts}
    assert any(
        "no signature labelling" in note for note in result.notes
    ), f"the absence has to be stated, not inferred from a missing file: {result.notes}"
    assert result.metrics["n_unassigned_clusters"] == 0


def test_a_labelling_run_writes_the_full_label_table(scored_adata, mock_context):
    """And when labelling does run, the table carries every column it computed."""
    result = _run(scored_adata, mock_context)

    table = _table(result, "mr_subtypes")
    assert list(table.columns) == [
        "cluster",
        "label",
        "top_signature",
        "top_z",
        "second_z",
        "margin",
    ]
    assert len(table) == scored_adata.obs["cluster"].nunique()


def test_module_remodeling_reads_program_names_that_came_back_from_disk(
    scored_adata, mock_context, tmp_path
):
    """The score matrix normally arrives via a checkpoint, not from memory.

    ``uns["state_aucell"]["programs"]`` is written as a Python list and read back as
    an ``object``-dtype numpy array, so any truthiness test on it
    (``if not names``) raises "truth value of an array ... is ambiguous". Every
    fixture in this file builds the object in memory, where the value is still a
    list — which is exactly why the stage passed 14 tests and then failed on the
    first real run, before a single statistic was computed.

    Round-tripping through h5ad is the only way a synthetic fixture reproduces
    that, so this test does it.
    """
    path = tmp_path / "roundtrip.h5ad"
    scored_adata.write_h5ad(path)
    from_disk = ad.read_h5ad(path)
    assert isinstance(
        from_disk.uns["state_aucell"]["programs"], np.ndarray
    ), "premise check: anndata no longer returns an array here, so this test is moot"

    result = _run(from_disk, mock_context)
    assert not isinstance(result, MethodSkip), getattr(result, "reason", result)
    effects = _table(result, "mr_effect_sizes")
    assert set(effects["program"]) == set(PROGRAMS)


def test_module_remodeling_skips_loudly_without_a_score_matrix(scored_adata, mock_context):
    """No ``state_scoring`` upstream is a recorded skip, never a crash or a blank."""
    bare = scored_adata.copy()
    del bare.obsm["X_state_aucell"]
    result = _run(bare, mock_context)
    assert isinstance(result, MethodSkip)
    assert "X_state_aucell" in result.reason


def test_module_remodeling_skips_loudly_when_the_design_is_unset(scored_adata, mock_context):
    result = _run(scored_adata, mock_context, case=None, control=None)
    assert isinstance(result, MethodSkip)
    assert "case" in result.reason.lower() or "control" in result.reason.lower()


def test_module_remodeling_skips_loudly_when_the_cluster_column_is_absent(
    scored_adata, mock_context
):
    result = _run(scored_adata, mock_context, cluster_col="not_a_column")
    assert isinstance(result, MethodSkip)
    assert "not_a_column" in result.reason


# --------------------------------------------------------------------------- #
# depth audit                                                                  #
# --------------------------------------------------------------------------- #
# Every program score here is a continuous per-cell quantity, so every one of them
# rises with library depth; if depth also differs between the arms, a condition
# effect has a second explanation that donor pairing does not touch. The engine
# owns that check because leaving it to each driver means it runs when the analyst
# remembers, which is not a property of the engine at all.
DEPTH_COUPLED = "prog_d"  # null in the fixture, and not an identity or index program


DEPTH_MASKED = "prog_e"  # the other null program, so the two plants cannot interfere


def _with_depth(
    adata,
    *,
    depth_shift: float,
    coupled=(DEPTH_COUPLED,),
    masked=(),
    seed: int = 5,
):
    """Give the fixture a depth column, optionally imbalanced and optionally coupled.

    ``depth_shift`` is added to every case cell, which is the arm imbalance. A program
    named in ``coupled`` is overwritten with ``log1p(depth)`` plus noise, so its entire
    condition effect comes from the imbalance and nothing else — the adjustment is
    against ``log1p`` too, so a correct audit should remove essentially all of it.

    A program named in ``masked`` gets the same depth term *plus* a fall with condition
    of exactly the size the depth term inflates it by, so the two cancel: it has no
    unadjusted condition effect and its whole effect is recoverable. That is the other
    direction of the same confound, and an audit that only ever removes claims never
    reports it.
    """
    rng = np.random.default_rng(seed)
    out = adata.copy()
    is_case = (out.obs["condition"] == "Lymphedema").to_numpy()
    depth = rng.normal(2000.0, 250.0, size=out.n_obs) + depth_shift * is_case
    out.obs["n_genes_by_counts"] = depth
    matrix = np.asarray(out.obsm["X_state_aucell"], dtype=float).copy()
    lifted = np.log1p(depth)
    for program in coupled:
        matrix[:, PROGRAMS.index(program)] = lifted + rng.normal(0.0, 0.02, out.n_obs)
    hidden = float(lifted[is_case].mean() - lifted[~is_case].mean())
    for program in masked:
        matrix[:, PROGRAMS.index(program)] = (
            lifted - hidden * is_case + rng.normal(0.0, 0.02, out.n_obs)
        )
    out.obsm["X_state_aucell"] = matrix
    return out


def test_no_depth_column_warns_instead_of_skipping_the_audit_quietly(scored_adata, mock_context):
    """An unaudited run must say so, in the warnings and on the headline.

    The fixture carries no QC metrics, which is the state a hand-built object arrives
    in. The stage still has to produce its other five tables — the audit is not a
    precondition for the effect sizes — but a run summary that reports significant
    programs without mentioning that none of them was audited invites a reader to
    quote the first number and never reach the second.
    """
    result = _run(scored_adata, mock_context)
    assert not isinstance(result, MethodSkip), getattr(result, "reason", result)

    assert "mr_depth_audit" not in {a.name for a in result.artifacts}
    assert "mr_effect_sizes" in {a.name for a in result.artifacts}
    assert any(
        "depth audit skipped" in w and "n_genes_by_counts" in w for w in result.warnings
    ), f"the candidates it looked for have to be named so the fix is obvious: {result.warnings}"
    assert "depth unaudited" in result.notes[0]
    assert result.metrics["depth_col"] is None
    assert result.metrics["n_depth_driven"] == 0


def test_a_configured_depth_column_that_is_absent_does_not_fall_back(scored_adata, mock_context):
    """Naming a column the object lacks is a skip, never a substitution.

    Falling back to ``n_genes_by_counts`` here would audit a different covariate from
    the one the manifest asked for and label the result as though the request had been
    honoured — so a passing audit would be read as evidence about the named column.
    That is worse than not auditing, because it cannot be noticed from the output.
    """
    adata = _with_depth(scored_adata, depth_shift=600.0)
    result = _run(adata, mock_context, depth_col="umis_per_cell")

    assert result.metrics["depth_col"] is None
    assert "mr_depth_audit" not in {a.name for a in result.artifacts}
    assert any(
        "umis_per_cell" in w and "absent" in w for w in result.warnings
    ), f"the skip has to name the column that was asked for: {result.warnings}"


def test_the_audit_covers_every_program_and_the_index_it_plots(scored_adata, mock_context):
    """One row per program plus the contrast index, in the figures' program order.

    The index is the quantity a manuscript sentence is usually stated on, and being a
    z-scored combination of depth-sensitive scores does not make it less
    depth-sensitive — so auditing the programs and not the index would leave the
    headline claim unchecked.
    """
    adata = _with_depth(scored_adata, depth_shift=600.0)
    result = _run(adata, mock_context)
    assert not isinstance(result, MethodSkip), getattr(result, "reason", result)

    audit = _table(result, "mr_depth_audit")
    assert list(audit["metric"]) == [*PROGRAMS, "contrast_index"], (
        "the audit shares the tables' and panels' axis order, or a reader lines up "
        f"two differently ordered CSVs by hand: {list(audit['metric'])}"
    )
    for column in ("spearman_rho_vs_depth", "raw_delta", "adjusted_delta", "verdict", "reason"):
        assert column in audit.columns
    assert result.metrics["depth_col"] == "n_genes_by_counts"


def test_a_depth_driven_program_is_flagged_and_a_real_one_is_not(scored_adata, mock_context):
    """The audit has to separate the two, not flag everything that tracks depth.

    ``prog_d`` is nothing but ``log1p(depth)``, and the arms differ in depth, so its
    condition effect is entirely a library-size readout. ``prog_a`` and ``prog_b``
    carry the planted effects and were built independent of depth. An audit that
    flags all three is unusable — it would retract the finding along with the
    artifact — and one that flags none is not an audit.
    """
    adata = _with_depth(scored_adata, depth_shift=600.0)
    result = _run(adata, mock_context)

    audit = _table(result, "mr_depth_audit").set_index("metric")
    assert audit.loc[DEPTH_COUPLED, "verdict"] == "depth_driven", audit.loc[
        DEPTH_COUPLED, ["raw_delta", "adjusted_delta", "verdict", "reason"]
    ].to_dict()
    for program in ("prog_a", PLANTED_DOWN):
        evidence = audit.loc[program, ["raw_delta", "adjusted_delta", "reason"]].to_dict()
        assert audit.loc[program, "verdict"] == "robust", (
            f"{program} was constructed independent of depth; flagging it retracts a "
            f"true effect: {evidence}"
        )

    assert result.metrics["n_depth_driven"] == 1
    assert any(
        "depth audit" in w and DEPTH_COUPLED in w for w in result.warnings
    ), f"a contradicted program has to be named in the warnings: {result.warnings}"
    assert f"{1} program(s) depth-driven" in result.notes[0]


def test_depth_balanced_arms_are_reported_as_safe_not_as_silence(scored_adata, mock_context):
    """A cohort whose arms match on depth cannot have a depth-driven effect.

    ``prog_d`` still tracks depth perfectly here — the coupling is identical to the
    test above — and the only difference is that the arms are balanced, which is the
    leg that gates the other two. Reporting the coupling as a problem here would
    manufacture alarm on a clean cohort; reporting nothing at all would leave a
    reader unable to tell an audited run from an unaudited one.
    """
    adata = _with_depth(scored_adata, depth_shift=0.0)
    result = _run(adata, mock_context)

    audit = _table(result, "mr_depth_audit")
    assert (audit["verdict"] == "depth_balanced").all(), audit[["metric", "verdict"]].to_dict()
    assert not audit["depth_is_confounded"].any()
    assert result.metrics["n_depth_driven"] == 0
    assert "none depth-driven" in result.notes[0]
    assert any(
        "depth-balanced" in note for note in result.notes
    ), f"the informative negative has to be stated: {result.notes}"


def test_a_program_depth_was_hiding_is_surfaced_as_a_lead(scored_adata, mock_context):
    """The audit has to report what removing depth reveals, not only what it kills.

    ``prog_e`` here falls with condition by exactly the amount the deeper case
    libraries inflate it, so the unadjusted paired test sees nothing and the adjusted
    one sees the whole fall. An audit that filed that under "no raw effect, nothing to
    audit" would throw away the one program it had found rather than protected, and
    the stage would never mention it — so the verdict has to reach the table, the
    metrics and the headline, and it has to be worded as a lead, because the
    unadjusted test is the one this table's FDR family was corrected over.
    """
    adata = _with_depth(scored_adata, depth_shift=600.0, coupled=(), masked=(DEPTH_MASKED,))
    result = _run(adata, mock_context)

    audit = _table(result, "mr_depth_audit").set_index("metric")
    row = audit.loc[DEPTH_MASKED]
    assert row["verdict"] == "depth_masked", audit[["verdict", "raw_t_p", "adjusted_t_p"]].to_dict()
    assert float(row["raw_t_p"]) >= 0.05, "the fixture must hide the effect before adjustment"
    assert float(row["adjusted_t_p"]) < 0.05
    assert float(row["adjusted_delta"]) < 0.0, "the recovered effect is the fall depth cancelled"

    assert result.metrics["n_depth_masked"] == 1
    assert result.metrics["n_depth_driven"] == 0
    assert any(
        DEPTH_MASKED in note and "lead" in note for note in result.notes
    ), f"the surfaced program has to be named and framed as a lead: {result.notes}"
    assert "1 depth-masked" in result.notes[0]


# --------------------------------------------------------------------------- #
# reusability boundary (spec §8)                                               #
# --------------------------------------------------------------------------- #
def test_module_remodeling_hardcodes_no_study_biology():
    """The engine stage stays generic; the endothelial specifics stay in config.

    This is what makes one capability serve the LEC manuscript and the next
    hypothesis repo both. A gene symbol or a lineage name compiled into the
    stage is the exact thing that turns a reusable stage into a one-off script,
    and it is invisible in review because it works — for one study.

    Docstrings are exempt: naming the motivating analysis is documentation. Only
    code-level string constants are checked.
    """
    package = (
        Path(__file__).resolve().parents[1] / "src/cellquorum/stages/comparative/module_remodeling"
    )
    if not package.is_dir():
        pytest.fail(f"stage package not found: {package}")

    banned = ("lec", "endomt", "lymphedema", "prox1", "lyve1", "ccl21", "pecam1", "mesenchymal")
    offenders = []
    for module in sorted(package.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue
            lowered = node.value.lower()
            for term in banned:
                if term in lowered:
                    offenders.append(f"{module.name}:{node.lineno}: {term!r} in {node.value!r}")

    assert not offenders, "study biology hardcoded in the generic stage:\n" + "\n".join(offenders)
