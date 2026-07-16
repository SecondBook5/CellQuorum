"""Tests for the CellQuorum QC pipeline stage."""

from __future__ import annotations

# Import JSON helpers for reading stage-written summary artifacts.
import json

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

# Import QC artifact manifest for artifact conversion tests.
from cellquorum.qc.artifacts import QCArtifactManifest

# Import QC configuration.
from cellquorum.qc.config import QCConfig

# Import QC decision result container.
from cellquorum.qc.decisions import QCDecisionResult

# Import QC stage utilities under test.
from cellquorum.qc.stage import (
    QCStage,
    QCStageError,
    add_metric_columns_to_axis,
    annotate_adata_with_qc_decisions,
    annotate_adata_with_qc_metrics,
    build_disabled_qc_stage_result,
    build_qc_output_adata,
    build_qc_stage_summary_extra,
    build_stage_artifacts_from_manifest,
    coerce_qc_config,
    describe_qc_artifact,
    filter_adata_by_qc_decisions,
    get_context_adata,
    get_qc_output_dir,
    infer_artifact_kind,
    is_qc_stage_enabled,
    resolve_qc_config,
    validate_decision_index_alignment,
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


def make_stage_qc_config(*, mode: str = "report_only") -> QCConfig:
    """
    Build a deterministic QC configuration for stage tests.

    Args:
        mode: QC execution mode.

    Returns:
        QCConfig with fixed thresholds only and simple output settings.
    """

    # Return the deterministic stage QC configuration.
    return QCConfig(
        mode=mode,
        threshold_strategy="fixed",
        metrics={"percent_top": [2]},
        basic={
            "min_genes_per_cell": 2,
            "min_cells_per_gene": 2,
            "max_mito_percent": 60.0,
        },
        mad={"enabled": False},
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


def make_decision_result() -> QCDecisionResult:
    """
    Build a deterministic decision result for direct stage helper tests.

    Returns:
        QCDecisionResult aligned to make_stage_test_adata().
    """

    # Build cell decisions.
    cell_decisions = pd.DataFrame(
        {
            "keep": [True, False, True],
            "fail_any_qc": [False, True, False],
            "failed_rules": ["", "fixed_min_genes_per_cell", ""],
            "fixed_min_genes_per_cell": [False, True, False],
        },
        index=["cell_1", "cell_2", "cell_3"],
    )

    # Build gene decisions.
    gene_decisions = pd.DataFrame(
        {
            "keep": [True, True, False, False],
            "fail_any_qc": [False, False, True, True],
            "failed_rules": ["", "", "fixed_min_cells_per_gene", "fixed_min_cells_per_gene"],
            "fixed_min_cells_per_gene": [False, False, True, True],
        },
        index=["MT-ND1", "ACTB", "RPS3", "MALAT1"],
    )

    # Return the structured decision result.
    return QCDecisionResult(
        cell_decisions=cell_decisions,
        gene_decisions=gene_decisions,
        summary={
            "n_cells": 3,
            "n_cells_kept": 2,
            "n_cells_failed": 1,
            "n_genes": 4,
            "n_genes_kept": 2,
            "n_genes_failed": 2,
        },
        warnings=[],
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
                "mode": "report_only",
            }
        },
    )

    # Resolve the QC config.
    resolved = resolve_qc_config(context)

    # Confirm mapping config was validated.
    assert isinstance(resolved, QCConfig)
    assert resolved.enabled is False


def test_resolve_qc_config_accepts_object_qc_field(tmp_path: Path) -> None:
    """
    Verify context.config.qc can be resolved into QCConfig.

    This supports future top-level configs once QC is embedded directly.
    """

    # Build a context with object-style QC config.
    context = make_context(
        tmp_path,
        config=SimpleNamespace(
            qc={"mode": "filter", "threshold_strategy": "fixed", "mad": {"enabled": False}}
        ),
    )

    # Resolve the QC config.
    resolved = resolve_qc_config(context)

    # Confirm object config was validated.
    assert isinstance(resolved, QCConfig)
    assert resolved.mode == "filter"


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


def test_validate_decision_index_alignment_accepts_exact_match() -> None:
    """
    Verify decision index alignment accepts exact ordered matches.

    Annotation and filtering rely on positionally aligned decision tables.
    """

    # Confirm exact matching indices do not raise.
    validate_decision_index_alignment(
        expected=["a", "b"],
        observed=["a", "b"],
        label="cell_decisions",
    )


def test_validate_decision_index_alignment_rejects_mismatch() -> None:
    """
    Verify decision index alignment rejects mismatched names.

    Mismatched decisions could filter the wrong cells or genes.
    """

    # Confirm mismatched indices fail clearly.
    with pytest.raises(QCStageError, match="index does not match"):
        validate_decision_index_alignment(
            expected=["a", "b"],
            observed=["b", "a"],
            label="cell_decisions",
        )


def test_annotate_adata_with_qc_decisions_adds_obs_and_var_columns() -> None:
    """
    Verify QC decisions are added to AnnData.obs and AnnData.var.

    The stage should retain audit columns even in report-only mode.
    """

    # Build AnnData and decisions.
    adata = make_stage_test_adata()
    decisions = make_decision_result()

    # Annotate AnnData.
    annotated = annotate_adata_with_qc_decisions(adata, decisions)

    # Confirm input AnnData was not mutated.
    assert "cellquorum_qc_keep" not in adata.obs.columns

    # Confirm cell-level QC annotations were added.
    assert annotated.obs["cellquorum_qc_keep"].tolist() == [True, False, True]
    assert annotated.obs["cellquorum_qc_fail_any_qc"].tolist() == [False, True, False]

    # Confirm gene-level QC annotations were added.
    assert annotated.var["cellquorum_qc_keep"].tolist() == [True, True, False, False]
    assert annotated.var["cellquorum_qc_fail_any_qc"].tolist() == [False, False, True, True]


def test_filter_adata_by_qc_decisions_subsets_cells_and_genes() -> None:
    """
    Verify QC filtering subsets AnnData by decision keep masks.

    The filter step should remove failed cells and failed genes together.
    """

    # Build AnnData and decisions.
    adata = make_stage_test_adata()
    decisions = make_decision_result()

    # Filter AnnData.
    filtered = filter_adata_by_qc_decisions(adata, decisions)

    # Confirm the expected cells were kept.
    assert list(filtered.obs_names) == ["cell_1", "cell_3"]

    # Confirm the expected genes were kept.
    assert list(filtered.var_names) == ["MT-ND1", "ACTB"]

    # Confirm the filtered shape.
    assert filtered.shape == (2, 2)


def test_build_qc_output_adata_report_only_preserves_shape() -> None:
    """
    Verify report-only QC annotates but does not filter AnnData.

    Report-only mode should be non-mutating with respect to the data matrix.
    """

    # Build AnnData and decisions.
    adata = make_stage_test_adata()
    decisions = make_decision_result()

    # Build report-only output AnnData.
    output = build_qc_output_adata(
        adata=adata,
        decision_result=decisions,
        config=make_stage_qc_config(mode="report_only"),
    )

    # Confirm shape was preserved.
    assert output.shape == adata.shape

    # Confirm annotations were added.
    assert "cellquorum_qc_keep" in output.obs.columns
    assert "cellquorum_qc_keep" in output.var.columns


def test_build_qc_output_adata_filter_mode_filters_shape() -> None:
    """
    Verify filter-mode QC annotates and filters AnnData.

    Filter mode should return only cells and genes marked keep=True.
    """

    # Build AnnData and decisions.
    adata = make_stage_test_adata()
    decisions = make_decision_result()

    # Build filtered output AnnData.
    output = build_qc_output_adata(
        adata=adata,
        decision_result=decisions,
        config=make_stage_qc_config(mode="filter"),
    )

    # Confirm cells and genes were filtered.
    assert output.shape == (2, 2)

    # Confirm annotations remain after filtering.
    assert output.obs["cellquorum_qc_keep"].tolist() == [True, True]
    assert output.var["cellquorum_qc_keep"].tolist() == [True, True]


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


def test_build_qc_stage_summary_extra_uses_context_metadata(tmp_path: Path) -> None:
    """
    Verify stage summary extras include run metadata and QC mode.

    These values are written into qc_summary.json through the artifact writer.
    """

    # Build a context.
    context = make_context(tmp_path)

    # Build QC config.
    config = make_stage_qc_config(mode="both")

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
    assert payload["mode"] == "both"
    assert payload["threshold_strategy"] == "fixed"
    assert payload["enabled_metric_families"] == ["basic", "doublets", "ambient_rna"]


def test_qc_stage_run_report_only_writes_artifacts_and_preserves_shape(tmp_path: Path) -> None:
    """
    Verify the full QC stage runs in report-only mode.

    Report-only mode should write QC artifacts, annotate AnnData, preserve the
    input shape, and return structured stage metrics.
    """

    # Build context and stage.
    context = make_context(tmp_path)
    stage = QCStage(config=make_stage_qc_config(mode="report_only"))

    # Run the QC stage.
    result = stage.run(context)

    # Confirm report-only mode preserved shape.
    assert result.adata.shape == (3, 4)

    # Confirm QC annotations exist.
    assert "cellquorum_qc_keep" in result.adata.obs.columns
    assert "cellquorum_qc_keep" in result.adata.var.columns

    # Confirm expected cells and genes were marked keep/fail.
    assert result.adata.obs["cellquorum_qc_keep"].tolist() == [True, False, True]
    assert result.adata.var["cellquorum_qc_keep"].tolist() == [True, True, False, False]

    # Confirm no warnings were emitted with the deterministic test config.
    assert result.warnings == []

    # Confirm stage notes summarize QC.
    assert result.notes[0] == "QC completed in report_only mode."

    # Confirm stage metrics include input and output shapes.
    assert result.metrics["input_shape"] == {"n_obs": 3, "n_vars": 4}
    assert result.metrics["output_shape"] == {"n_obs": 3, "n_vars": 4}

    # Confirm expected artifacts were reported.
    artifact_names = {artifact.name for artifact in result.artifacts}
    assert artifact_names == {
        "qc_cell_metrics",
        "qc_gene_metrics",
        "qc_feature_masks",
        "qc_thresholds",
        "qc_cell_decisions",
        "qc_gene_decisions",
        "qc_summary",
    }

    # Confirm the summary artifact exists and contains stage extra metadata.
    summary_artifact = next(
        artifact for artifact in result.artifacts if artifact.name == "qc_summary"
    )
    summary = json.loads(summary_artifact.path.read_text(encoding="utf-8"))
    assert summary["extra"]["stage_name"] == "qc"
    assert summary["extra"]["run_id"] == "stage-test-run"


def test_qc_stage_run_filter_mode_returns_filtered_anndata(tmp_path: Path) -> None:
    """
    Verify the full QC stage filters AnnData in filter mode.

    Filter mode should remove failed cells and failed genes according to explicit
    QC decision tables.
    """

    # Build context and stage.
    context = make_context(tmp_path)
    stage = QCStage(config=make_stage_qc_config(mode="filter"))

    # Run the QC stage.
    result = stage.run(context)

    # Confirm AnnData was filtered.
    assert result.adata.shape == (2, 2)

    # Confirm expected cells and genes remain.
    assert list(result.adata.obs_names) == ["cell_1", "cell_3"]
    assert list(result.adata.var_names) == ["MT-ND1", "ACTB"]

    # Confirm filtering note was emitted.
    assert result.notes[-1] == (
        "QC filtering changed AnnData shape from 3 cells x 4 genes to " "2 cells x 2 genes."
    )

    # Confirm stage metrics include filtered output shape.
    assert result.metrics["output_shape"] == {"n_obs": 2, "n_vars": 2}


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


def test_add_metric_columns_to_axis_preserves_existing_columns() -> None:
    """
    Verify metric annotation never overwrites a pre-existing obs/var column.

    An upstream tool may have populated ``total_counts`` with authoritative
    values; QC metric annotation must preserve them and report the conflict.
    """

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

    conflicts = add_metric_columns_to_axis(axis_frame=axis_frame, metrics=metrics)

    # The pre-existing column is preserved, not overwritten.
    assert list(axis_frame["total_counts"]) == [111.0, 222.0]
    # The non-conflicting metric column is added.
    assert list(axis_frame["pct_counts_mito"]) == [10.0, 20.0]
    # The conflict is reported for the caller to surface as a warning.
    assert conflicts == ["total_counts"]


def test_annotate_adata_with_qc_metrics_warns_on_preserved_columns() -> None:
    """
    Verify annotate_adata_with_qc_metrics returns warnings for preserved columns.
    """

    from cellquorum.qc.metrics import QCMetricsResult

    # Build an adata whose obs already carries a metric-named column.
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame({"total_counts": [111.0, 222.0]}, index=["cell_1", "cell_2"]),
        var=pd.DataFrame(index=["gene_1", "gene_2"]),
    )

    metrics_result = QCMetricsResult(
        cell_metrics=pd.DataFrame(
            {"total_counts": [5.0, 6.0], "pct_counts_mito": [10.0, 20.0]},
            index=["cell_1", "cell_2"],
        ),
        gene_metrics=pd.DataFrame(index=["gene_1", "gene_2"]),
        feature_masks=pd.DataFrame(index=["gene_1", "gene_2"]),
        summary={},
    )

    warnings = annotate_adata_with_qc_metrics(adata=adata, metrics_result=metrics_result)

    # Pre-existing obs values preserved; new metric added; conflict warned.
    assert list(adata.obs["total_counts"]) == [111.0, 222.0]
    assert list(adata.obs["pct_counts_mito"]) == [10.0, 20.0]
    assert any("total_counts" in w for w in warnings)
