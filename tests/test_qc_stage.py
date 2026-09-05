"""Tests for the CellQuorum QC pipeline stage."""

from __future__ import annotations

# Import JSON helpers for reading stage-written summary artifacts.
# Import Path for pytest tmp_path fixture annotations.
from pathlib import Path

# Import SimpleNamespace for lightweight context/config tests.
from types import SimpleNamespace

# Import AnnData for test input objects.
import anndata as ad

# Import NumPy for deterministic test matrices.
import numpy as np

# Import pandas for AnnData metadata and decision tables.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import top-level CellQuorum config for stage-selection tests.
from cellquorum.config.models import CellQuorumConfig

# Import pipeline context and path contracts.
from cellquorum.core.context import PipelineContext, PipelinePaths

# Import QC stage utilities under test.
from cellquorum.stages.qc._annotate import (
    add_metric_columns_to_axis,
    annotate_adata_with_qc_metrics,
)
from cellquorum.stages.qc._context import (
    coerce_qc_config,
    get_context_adata,
    get_qc_output_dir,
    is_qc_stage_enabled,
    resolve_qc_config,
)
from cellquorum.stages.qc._report import (
    build_disabled_qc_stage_result,
    build_qc_stage_summary_extra,
    build_stage_artifacts_from_manifest,
    describe_qc_artifact,
    infer_artifact_kind,
)

# Import QC artifact manifest for artifact conversion tests.
from cellquorum.stages.qc.artifacts import QCArtifactManifest
from cellquorum.stages.qc.config import QCConfig

# Import QC configuration.
from cellquorum.stages.qc.floors import FloorResult

# Import QC decision result container.
from cellquorum.stages.qc.stage import (
    QCStage,
    QCStageError,
)


def make_stage_test_adata() -> ad.AnnData:
    """
    Build a small AnnData object for QC stage tests.

    The data are chosen so fixed QC thresholds keep cell_1 and cell_3, remove
    cell_2, keep MT-ND1 and ACTB, and remove RPS3 and MALAT1.

    Returns:
        Small AnnData object.
    """

    # Build a deterministic count matrix.
    matrix = np.array(
        [
            [5.0, 5.0, 0.0, 0.0],
            [9.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )

    # Build observation metadata.
    obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])

    # Build variable metadata.
    var = pd.DataFrame(index=["MT-ND1", "ACTB", "RPS3", "MALAT1"])

    # Return AnnData.
    return ad.AnnData(X=matrix, obs=obs, var=var)


def make_stage_qc_config() -> QCConfig:
    """
    Build a deterministic QC configuration for stage tests.

    Args:

    Returns:
        QCConfig with fixed thresholds only and simple output settings.
    """

    # Return the deterministic stage QC configuration.
    return QCConfig(
        metrics={"percent_top": [2]},
        floors={
            "min_genes_per_cell": 2,
            "min_cells_per_gene": 2,
        },
        outputs={
            "write_h5ad": False,
            "write_figures": False,
        },
    )


def make_context(
    tmp_path: Path,
    *,
    adata: ad.AnnData | None = None,
    config: object | None = None,
) -> PipelineContext:
    """
    Build a PipelineContext for QC stage tests.

    Args:
        tmp_path: pytest temporary path.
        adata: Optional AnnData object.
        config: Optional runtime config.

    Returns:
        PipelineContext with initialized output paths.
    """

    # Build standardized pipeline paths.
    paths = PipelinePaths.from_output_dir(tmp_path / "run")

    # Create the path directories.
    paths.ensure_directories()

    # Return a PipelineContext.
    return PipelineContext(
        config=CellQuorumConfig() if config is None else config,
        paths=paths,
        adata=make_stage_test_adata() if adata is None else adata,
        run_id="stage-test-run",
        random_seed=123,
    )


def make_decision_result() -> FloorResult:
    """A small floor result for artifact/stage tests.

    Replaces a ``QCDecisionResult`` fixture. The shape is deliberately simpler: floors produce a
    keep mask and a reason, not a boolean column per threshold rule, because a barcode either
    cleared the detection limit or it did not — there is nothing to attribute.
    """
    cells = pd.Index(["cell_0", "cell_1", "cell_2"])
    genes = pd.Index(["gene_0", "gene_1", "gene_2", "gene_3"])
    cell_keep = pd.Series([True] * 2 + [False] * 1, index=cells)
    reason = pd.Series([""] * 2 + ["fewer_than_100_genes"] * 1, index=cells)
    gene_keep = pd.Series([True] * 2 + [False] * 2, index=genes)
    return FloorResult(
        cell_keep=cell_keep,
        gene_keep=gene_keep,
        reason=reason,
        summary={
            "n_cells": 3,
            "n_cells_below_floor": 1,
            "n_genes": 4,
            "n_genes_below_floor": 2,
        },
        warnings=["floor warning"],
    )


def test_qc_stage_has_stable_name_and_default_output_subdir() -> None:
    """
    Verify QCStage exposes a stable stage name and output subdirectory.

    The pipeline stage contract uses the stage name in plans and provenance.
    """

    # Build the stage.
    stage = QCStage()

    # Confirm the stable stage name.
    assert stage.name == "qc"

    # Confirm the default output subdirectory.
    assert stage.output_subdir == "qc"


def test_resolve_qc_config_prefers_explicit_override(tmp_path: Path) -> None:
    """
    Verify explicit QCStage config overrides context configuration.

    Stage-level overrides are useful in direct tests and programmatic runs.
    """

    # Build a context with default config.
    context = make_context(tmp_path)

    # Build a disabled override.
    override = QCConfig(enabled=False)

    # Resolve the QC config.
    resolved = resolve_qc_config(context, override=override)

    # Confirm the explicit override was used.
    assert resolved is override


def test_resolve_qc_config_accepts_context_qc_config(tmp_path: Path) -> None:
    """
    Verify context.config can be a QCConfig directly.

    This keeps direct module testing simple without requiring top-level config.
    """

    # Build a QC config.
    qc_config = make_stage_qc_config()

    # Build a context whose config is already QCConfig.
    context = make_context(tmp_path, config=qc_config)

    # Resolve the QC config.
    resolved = resolve_qc_config(context)

    # Confirm the context QCConfig was used.
    assert resolved is qc_config


def test_resolve_qc_config_accepts_object_qc_field(tmp_path: Path) -> None:
    """
    Verify context.config.qc can be resolved into QCConfig.

    This supports future top-level configs once QC is embedded directly.
    """

    # Build a context with object-style QC config.
    context = make_context(
        tmp_path,
        config=SimpleNamespace(qc={"floors": {"min_genes_per_cell": 1}}),
    )

    # Resolve the QC config.
    resolved = resolve_qc_config(context)

    # Confirm object config was validated.
    assert isinstance(resolved, QCConfig)
    assert resolved.floors.min_genes_per_cell == 1


def test_coerce_qc_config_rejects_invalid_value() -> None:
    """
    Verify QC config coercion rejects unsupported values.

    Unsupported config values should fail before stage execution begins.
    """

    # Confirm invalid config values fail clearly.
    with pytest.raises(QCStageError, match="QC configuration must be"):
        coerce_qc_config(["not", "config"])


def test_is_qc_stage_enabled_respects_qc_config_flag(tmp_path: Path) -> None:
    """
    Verify QCConfig.enabled disables the stage.

    Module-level disabled flags should prevent execution even when top-level
    stages allow QC.
    """

    # Build a context.
    context = make_context(tmp_path)

    # Confirm disabled QC config disables the stage.
    assert is_qc_stage_enabled(context, QCConfig(enabled=False)) is False


def test_is_qc_stage_enabled_respects_top_level_stage_selection(tmp_path: Path) -> None:
    """
    Verify top-level stages.qc=false disables the stage.

    The planner should usually avoid running disabled stages, but the stage also
    defends against direct execution.
    """

    # Build top-level config with QC disabled.
    config = CellQuorumConfig(stages={"qc": False})

    # Build a context.
    context = make_context(tmp_path, config=config)

    # Confirm top-level stage selection disables the stage.
    assert is_qc_stage_enabled(context, QCConfig(enabled=True)) is False


def test_get_context_adata_returns_active_anndata(tmp_path: Path) -> None:
    """
    Verify AnnData retrieval works through PipelineContext.require_adata.

    QCStage should use the formal context helper when available.
    """

    # Build a context.
    context = make_context(tmp_path)

    # Retrieve AnnData.
    adata = get_context_adata(context)

    # Confirm the active AnnData object was returned.
    assert adata is context.adata


def test_get_context_adata_rejects_missing_anndata(tmp_path: Path) -> None:
    """
    Verify AnnData retrieval fails clearly when context lacks AnnData.

    QC cannot run before ingestion/loading has produced an AnnData object.
    """

    # Build a context with missing AnnData.
    context = make_context(tmp_path, adata=None)
    context.adata = None

    # Confirm missing AnnData fails clearly.
    with pytest.raises(QCStageError, match="requires an AnnData object"):
        get_context_adata(context)


def test_get_context_adata_rejects_invalid_direct_adata() -> None:
    """
    Verify direct context.adata fallback validates AnnData type.

    Lightweight context objects should still be type checked.
    """

    # Build a direct context with invalid adata.
    context = SimpleNamespace(adata={"not": "anndata"})

    # Confirm invalid AnnData fails clearly.
    with pytest.raises(QCStageError, match="context.adata"):
        get_context_adata(context)


def test_get_qc_output_dir_resolves_under_results(tmp_path: Path) -> None:
    """
    Verify QC output directory resolves under context.paths.results.

    QC artifacts should live in the standardized results namespace.
    """

    # Build a context.
    context = make_context(tmp_path)

    # Resolve the QC output directory.
    output_dir = get_qc_output_dir(context, "qc")

    # Confirm the path is under results.
    assert output_dir == context.paths.results / "qc"


def test_get_qc_output_dir_rejects_missing_paths() -> None:
    """
    Verify QC output resolution requires context paths.

    Stages should not invent output locations without PipelinePaths.
    """

    # Build a context without paths.
    context = SimpleNamespace(config=QCConfig(), adata=make_stage_test_adata())

    # Confirm missing paths fail clearly.
    with pytest.raises(QCStageError, match="context.paths"):
        get_qc_output_dir(context, "qc")


def test_build_disabled_qc_stage_result_returns_noop_result() -> None:
    """
    Verify disabled QC produces an explicit no-op StageResult.

    Disabled stages should be visible in result metrics and notes.
    """

    # Build AnnData.
    adata = make_stage_test_adata()

    # Build disabled result.
    result = build_disabled_qc_stage_result(
        adata=adata,
        stage_name="qc",
        qc_config=QCConfig(enabled=False),
    )

    # Confirm AnnData was returned unchanged.
    assert result.adata is adata

    # Confirm no artifacts were emitted.
    assert result.artifacts == []

    # Confirm a note explains the no-op.
    assert result.notes == ["QC stage skipped because QC is disabled."]

    # Confirm metrics describe the disabled state.
    assert result.metrics["enabled"] is False
    assert result.metrics["reason"] == "qc_disabled"


def test_build_stage_artifacts_from_manifest_converts_written_artifacts(tmp_path: Path) -> None:
    """
    Verify QC artifact manifest conversion produces StageArtifact records.

    StageResult artifacts should use stage-prefixed names and inferred file
    kinds.
    """

    # Build a manifest with representative artifacts.
    manifest = QCArtifactManifest(
        output_dir=tmp_path,
        artifacts={
            "cell_metrics": tmp_path / "cell_metrics.csv",
            "summary": tmp_path / "qc_summary.json",
            "qc_h5ad": tmp_path / "qc.h5ad",
        },
    )

    # Convert to stage artifacts.
    artifacts = build_stage_artifacts_from_manifest(manifest)

    # Confirm stage-prefixed names.
    assert [artifact.name for artifact in artifacts] == [
        "qc_cell_metrics",
        "qc_summary",
        "qc_qc_h5ad",
    ]

    # Confirm artifact kinds.
    assert [artifact.kind for artifact in artifacts] == ["csv", "json", "h5ad"]


def test_infer_artifact_kind_maps_known_suffixes(tmp_path: Path) -> None:
    """
    Verify artifact kind inference maps known QC file suffixes.

    StageArtifact.kind should be compact and predictable.
    """

    # Confirm known suffix mappings.
    assert infer_artifact_kind(tmp_path / "table.csv") == "csv"
    assert infer_artifact_kind(tmp_path / "summary.json") == "json"
    assert infer_artifact_kind(tmp_path / "object.h5ad") == "h5ad"

    # Confirm unknown suffix fallback.
    assert infer_artifact_kind(tmp_path / "artifact.txt") == "file"


def test_describe_qc_artifact_returns_known_and_fallback_descriptions() -> None:
    """
    Verify QC artifact descriptions are human-readable.

    Descriptions appear in StageResult summaries and provenance.
    """

    # Confirm known artifact description.
    assert describe_qc_artifact("cell_metrics") == "Cell-level QC metric table."

    # Confirm fallback artifact description.
    assert describe_qc_artifact("unknown") == "QC artifact: unknown."


def test_qc_stage_run_disabled_returns_no_artifacts(tmp_path: Path) -> None:
    """
    Verify disabled QC stage returns an explicit no-op result.

    No artifacts should be written when QC is disabled.
    """

    # Build context and disabled stage.
    context = make_context(tmp_path)
    stage = QCStage(config=QCConfig(enabled=False))

    # Run the stage.
    result = stage.run(context)

    # Confirm no artifacts were emitted.
    assert result.artifacts == []

    # Confirm disabled metrics were returned.
    assert result.metrics["enabled"] is False

    # Confirm the original shape is preserved.
    assert result.adata.shape == (3, 4)


def test_qc_stage_run_respects_top_level_stage_disabled_flag(tmp_path: Path) -> None:
    """
    Verify direct QCStage execution respects top-level stages.qc=false.

    This protects direct execution paths in addition to planner gating.
    """

    # Build top-level config with QC disabled.
    config = CellQuorumConfig(stages={"qc": False})

    # Build context and stage.
    context = make_context(tmp_path, config=config)
    stage = QCStage(config=make_stage_qc_config())

    # Run the stage.
    result = stage.run(context)

    # Confirm the stage was skipped as a no-op.
    assert result.metrics["enabled"] is False
    assert result.artifacts == []


def test_qc_stage_run_rejects_missing_anndata(tmp_path: Path) -> None:
    """
    Verify full stage execution rejects missing AnnData.

    QC cannot execute before a loading/ingestion stage has populated context.adata.
    """

    # Build context with no AnnData.
    context = make_context(tmp_path)
    context.adata = None

    # Build stage.
    stage = QCStage(config=make_stage_qc_config())

    # Confirm stage execution fails clearly.
    with pytest.raises(QCStageError, match="requires an AnnData object"):
        stage.run(context)


def test_qc_stage_run_rejects_missing_paths() -> None:
    """
    Verify full stage execution rejects contexts without paths.

    QC artifact writing requires standardized PipelinePaths.
    """

    # Build a lightweight context without paths.
    context = SimpleNamespace(
        config=QCConfig(),
        adata=make_stage_test_adata(),
    )

    # Build stage.
    stage = QCStage(config=make_stage_qc_config())

    # Confirm missing paths fail clearly.
    with pytest.raises(QCStageError, match="context.paths"):
        stage.run(context)


# ---------------------------------------------------------------------------
# Metric annotation used to be flag-not-clobber, and these two tests asserted
# that. It was reversed deliberately, because on a per-lineage arm carved out of
# an atlas the inherited columns are not a competing opinion but a description of
# a different object: gene-level metrics are aggregates OVER cells, so the clean
# LEC input (2,125 cells) carries var['n_cells_by_counts'] up to 200,072. QC
# recomputed them correctly from this object's own matrix and then discarded them,
# so the final h5ad shipped whole-atlas gene metrics to anything reading var.
# The fresh value now wins; a value that already agrees is left alone and silent.
# ---------------------------------------------------------------------------


def test_add_metric_columns_to_axis_replaces_a_disagreeing_column() -> None:
    """A pre-existing metric column that disagrees is stale, and QC overrules it."""

    # Build an axis frame that already carries a QC-metric-named column.
    axis_frame = pd.DataFrame(
        {"total_counts": [111.0, 222.0], "existing_only": [1, 2]},
        index=["cell_1", "cell_2"],
    )

    # Build an aligned metric table that collides on total_counts and adds a new one.
    metrics = pd.DataFrame(
        {"total_counts": [5.0, 6.0], "pct_counts_mito": [10.0, 20.0]},
        index=["cell_1", "cell_2"],
    )

    replaced = add_metric_columns_to_axis(axis_frame=axis_frame, metrics=metrics)

    # The freshly computed value wins.
    assert list(axis_frame["total_counts"]) == [5.0, 6.0]
    # The non-conflicting metric column is added, and unrelated columns are untouched.
    assert list(axis_frame["pct_counts_mito"]) == [10.0, 20.0]
    assert list(axis_frame["existing_only"]) == [1, 2]
    # The replacement is reported, and names both magnitudes so it is auditable.
    assert len(replaced) == 1
    assert replaced[0].startswith("total_counts")
    assert "222" in replaced[0] and "6" in replaced[0]


def test_add_metric_columns_to_axis_is_silent_when_values_already_agree() -> None:
    """Per-cell metrics are invariant under cell subsetting, so they match exactly.

    Reporting those was noise on every clean-input run, and it made the genuinely
    stale gene-level columns look equally harmless.
    """

    axis_frame = pd.DataFrame({"total_counts": [5.0, 6.0]}, index=["cell_1", "cell_2"])
    metrics = pd.DataFrame({"total_counts": [5.0, 6.0]}, index=["cell_1", "cell_2"])

    assert add_metric_columns_to_axis(axis_frame=axis_frame, metrics=metrics) == []
    assert list(axis_frame["total_counts"]) == [5.0, 6.0]


def test_add_metric_columns_to_axis_treats_nan_as_agreeing() -> None:
    """A metric that is NaN in both places has not changed; NaN != NaN must not fool it."""

    axis_frame = pd.DataFrame({"pct_counts_mito": [np.nan, 1.0]}, index=["c1", "c2"])
    metrics = pd.DataFrame({"pct_counts_mito": [np.nan, 1.0]}, index=["c1", "c2"])

    assert add_metric_columns_to_axis(axis_frame=axis_frame, metrics=metrics) == []


def test_annotate_adata_with_qc_metrics_reports_a_stale_gene_level_column() -> None:
    """The real failure, at real scale: 2 cells cannot have a gene seen in 200,072.

    The var message must say WHY, since 'gene-level metrics are aggregates over
    cells' is the fact that makes an inherited value wrong rather than merely old.
    """

    from cellquorum.stages.qc.metrics import QCMetricsResult

    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame({"total_counts": [5.0, 6.0]}, index=["cell_1", "cell_2"]),
        var=pd.DataFrame({"n_cells_by_counts": [200072, 150000]}, index=["gene_1", "gene_2"]),
    )

    metrics_result = QCMetricsResult(
        cell_metrics=pd.DataFrame(
            # obs total_counts AGREES, so it must not be reported.
            {"total_counts": [5.0, 6.0], "pct_counts_mito": [10.0, 20.0]},
            index=["cell_1", "cell_2"],
        ),
        gene_metrics=pd.DataFrame({"n_cells_by_counts": [2, 1]}, index=["gene_1", "gene_2"]),
        feature_masks=pd.DataFrame(index=["gene_1", "gene_2"]),
        summary={},
    )

    warnings = annotate_adata_with_qc_metrics(adata=adata, metrics_result=metrics_result)

    # The stale gene-level column is corrected to this object's own counts.
    assert list(adata.var["n_cells_by_counts"]) == [2, 1]
    assert list(adata.obs["pct_counts_mito"]) == [10.0, 20.0]

    joined = " ".join(warnings)
    assert "n_cells_by_counts" in joined
    assert "200072" in joined
    assert "aggregates over cells" in joined
    # The agreeing obs column produced no warning at all.
    assert not any("obs metric" in w for w in warnings)


def test_resolve_qc_config_accepts_mapping_qc_field(tmp_path: Path) -> None:
    """
    Verify context.config['qc'] can be resolved into QCConfig.

    Dictionary-like contexts are useful for lightweight adapters.
    """

    # Build a context with dictionary-style QC config.
    context = make_context(
        tmp_path,
        config={
            "qc": {
                "enabled": False,
            }
        },
    )

    # Resolve the QC config.
    resolved = resolve_qc_config(context)

    # Confirm mapping config was validated.
    assert isinstance(resolved, QCConfig)
    assert resolved.enabled is False


def test_build_qc_stage_summary_extra_uses_context_metadata(tmp_path: Path) -> None:
    """
    Verify stage summary extras include run metadata and QC mode.

    These values are written into qc_summary.json through the artifact writer.
    """

    # Build a context.
    context = make_context(tmp_path)

    # Build QC config.
    config = make_stage_qc_config()

    # Build summary extra payload.
    payload = build_qc_stage_summary_extra(
        context=context,
        qc_config=config,
        stage_name="qc",
    )

    # Confirm context metadata is present.
    assert payload["stage_name"] == "qc"
    assert payload["run_id"] == "stage-test-run"
    assert payload["random_seed"] == 123
    assert payload["enabled_metric_families"] == ["floors", "doublets", "ambient_rna"]
