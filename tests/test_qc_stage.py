"""Tests for the CellQuorum QC pipeline stage."""

from __future__ import annotations

# Import dataclass for small config wrapper tests.
from dataclasses import dataclass

# Import Path for filesystem-based stage tests.
from pathlib import Path

# Import AnnData for tiny stage input objects.
import anndata as ad

# Import NumPy for deterministic matrices.
import numpy as np

# Import pandas for AnnData metadata and decision tables.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import pipeline context and path contracts.
from cellquorum.core.context import PipelineContext, PipelinePaths

# Import generic stage result contract.
from cellquorum.core.stage import StageArtifact, StageResult

# Import QC configuration.
from cellquorum.qc.config import QCConfig

# Import QC decision result container.
from cellquorum.qc.decisions import QCDecisionResult

# Import QC stage utilities under test.
from cellquorum.qc.stage import (
    QCStage,
    QCStageError,
    QCWorkflowResult,
    apply_qc_filter_to_adata,
    build_qc_stage_notes,
    collect_qc_stage_warnings,
    deduplicate_strings,
    describe_qc_artifact,
    infer_artifact_kind,
    prepare_adata_for_qc,
    resolve_qc_config,
    run_qc_workflow,
    stage_artifacts_from_qc_manifest,
    summarize_adata_shape_for_stage,
    validate_decision_index_alignment,
)

# Import threshold record for hand-built decision tests.


@dataclass
class ConfigWithQC:
    """
    Minimal top-level config wrapper with a qc attribute.

    Args:
        qc: QC configuration payload.
    """

    # Store a QC config payload.
    qc: object


def make_stage_adata() -> ad.AnnData:
    """
    Build a tiny AnnData object for QC stage tests.

    The fixed thresholds in the test config will fail one low-quality cell and
    one low-detected gene while keeping a non-empty filtered result.

    Returns:
        Small AnnData object.
    """

    # Build a deterministic count matrix.
    matrix = np.array(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 0.0],
        ]
    )

    # Build observation metadata.
    obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3", "cell_4"])

    # Build variable metadata with one mitochondrial gene.
    var = pd.DataFrame(index=["MT-ND1", "ACTB", "MALAT1"])

    # Return the AnnData object.
    return ad.AnnData(X=matrix, obs=obs, var=var)


def make_stage_config(**overrides: object) -> QCConfig:
    """
    Build a deterministic QC stage test configuration.

    Args:
        **overrides: Optional top-level QCConfig overrides.

    Returns:
        QCConfig suitable for tiny test matrices.
    """

    # Build the base configuration.
    payload: dict[str, object] = {
        "mode": "report_only",
        "threshold_strategy": "fixed",
        "basic": {
            "min_genes_per_cell": 1,
            "max_genes_per_cell": None,
            "min_counts_per_cell": 2,
            "max_counts_per_cell": None,
            "min_cells_per_gene": 1,
            "max_mito_percent": 90.0,
            "max_ribo_percent": None,
            "max_hemoglobin_percent": None,
        },
        "mad": {
            "enabled": False,
        },
        "outputs": {
            "write_figures": False,
            "write_h5ad": True,
        },
    }

    # Apply top-level overrides.
    payload.update(overrides)

    # Return the validated config.
    return QCConfig.model_validate(payload)


def make_pipeline_context(
    *,
    tmp_path: Path,
    adata: ad.AnnData | None = None,
    config: object | None = None,
) -> PipelineContext:
    """
    Build a PipelineContext for QC stage tests.

    Args:
        tmp_path: Temporary directory root.
        adata: Optional AnnData object.
        config: Optional context configuration.

    Returns:
        PipelineContext with ensured output directories.
    """

    # Build standardized pipeline paths.
    paths = PipelinePaths.from_output_dir(tmp_path / "run")

    # Ensure all directories exist.
    paths.ensure_directories()

    # Return the pipeline context.
    return PipelineContext(
        config=make_stage_config() if config is None else config,
        paths=paths,
        adata=make_stage_adata() if adata is None else adata,
        run_id="test-run",
        random_seed=123,
    )


def test_resolve_qc_config_prefers_explicit_config(tmp_path: Path) -> None:
    """
    Verify explicit QCStage config overrides context config.

    Stage-level explicit config should be the most deterministic option.
    """

    # Build context with default config.
    context = make_pipeline_context(tmp_path=tmp_path)

    # Build explicit disabled config.
    explicit = make_stage_config(enabled=False)

    # Resolve the config.
    observed = resolve_qc_config(context=context, explicit_config=explicit)

    # Confirm the explicit config was used.
    assert observed is explicit
    assert observed.enabled is False


def test_resolve_qc_config_accepts_context_qcconfig(tmp_path: Path) -> None:
    """
    Verify config resolution accepts PipelineContext.config as QCConfig.

    This is the simplest notebook/test usage pattern.
    """

    # Build a QC config.
    config = make_stage_config(mode="both")

    # Build context using that config directly.
    context = make_pipeline_context(tmp_path=tmp_path, config=config)

    # Resolve the config.
    observed = resolve_qc_config(context=context)

    # Confirm the context config was used.
    assert observed is config


def test_resolve_qc_config_accepts_qc_attribute_object(tmp_path: Path) -> None:
    """
    Verify config resolution accepts context.config.qc as QCConfig.

    This supports future top-level runtime configs that expose a qc attribute.
    """

    # Build a QC config.
    config = make_stage_config(mode="both")

    # Build wrapped config.
    context = make_pipeline_context(tmp_path=tmp_path, config=ConfigWithQC(qc=config))

    # Resolve the config.
    observed = resolve_qc_config(context=context)

    # Confirm the qc attribute was used.
    assert observed is config


def test_resolve_qc_config_accepts_qc_attribute_mapping(tmp_path: Path) -> None:
    """
    Verify config resolution validates mapping-style qc attributes.

    This supports plain dictionaries and partially bootstrapped config objects.
    """

    # Build context with a qc mapping.
    context = make_pipeline_context(
        tmp_path=tmp_path,
        config=ConfigWithQC(
            qc={
                "mode": "filter",
                "threshold_strategy": "fixed",
                "mad": {"enabled": False},
            }
        ),
    )

    # Resolve the config.
    observed = resolve_qc_config(context=context)

    # Confirm the mapping was validated.
    assert isinstance(observed, QCConfig)
    assert observed.mode == "filter"
    assert observed.threshold_strategy == "fixed"


def test_resolve_qc_config_accepts_mapping_with_qc_key(tmp_path: Path) -> None:
    """
    Verify config resolution accepts dictionary config['qc'] payloads.

    This is useful for CLI/Hydra integration before the full config model includes
    a first-class QC field.
    """

    # Build context with a qc dictionary.
    context = make_pipeline_context(
        tmp_path=tmp_path,
        config={
            "qc": {
                "mode": "both",
                "threshold_strategy": "fixed",
                "mad": {"enabled": False},
            }
        },
    )

    # Resolve the config.
    observed = resolve_qc_config(context=context)

    # Confirm mapping config was validated.
    assert observed.mode == "both"


def test_resolve_qc_config_rejects_invalid_qc_attribute(tmp_path: Path) -> None:
    """
    Verify config resolution rejects unsupported qc attributes.

    Invalid config objects should fail before stage execution.
    """

    # Build context with an invalid qc attribute.
    context = make_pipeline_context(tmp_path=tmp_path, config=ConfigWithQC(qc=123))

    # Confirm invalid qc attribute fails clearly.
    with pytest.raises(QCStageError, match="context.config.qc must be"):
        resolve_qc_config(context=context)


def test_prepare_adata_for_qc_makes_duplicate_names_unique_when_configured() -> None:
    """
    Verify duplicate-name preparation repairs names when configured.

    The QC stage should make names unique before metric calculation when the
    duplicate-name policy says make_unique.
    """

    # Build an AnnData object with duplicate names.
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(index=["cell", "cell"]),
        var=pd.DataFrame(index=["gene", "gene"]),
    )

    # Build config that repairs both axes.
    config = make_stage_config(
        duplicate_names={
            "obs_names": "make_unique",
            "var_names": "make_unique",
        }
    )

    # Prepare AnnData for QC.
    prepared, notes = prepare_adata_for_qc(adata, config)

    # Confirm a copy was created.
    assert prepared is not adata

    # Confirm observation names were made unique.
    assert prepared.obs_names.is_unique

    # Confirm variable names were made unique.
    assert prepared.var_names.is_unique

    # Confirm preparation notes were emitted.
    assert notes == [
        "Made duplicate AnnData.obs_names unique before QC metric calculation.",
        "Made duplicate AnnData.var_names unique before QC metric calculation.",
    ]


def test_prepare_adata_for_qc_returns_original_when_no_repair_needed() -> None:
    """
    Verify AnnData preparation avoids unnecessary copies.

    Copying large single-cell objects unnecessarily would be expensive.
    """

    # Build a normal AnnData object.
    adata = make_stage_adata()

    # Prepare AnnData for QC.
    prepared, notes = prepare_adata_for_qc(adata, make_stage_config())

    # Confirm original object was reused.
    assert prepared is adata

    # Confirm no preparation notes were emitted.
    assert notes == []


def test_apply_qc_filter_to_adata_filters_cells_and_genes() -> None:
    """
    Verify QC filtering applies cell and gene keep masks by position.

    Filtering should retain only rows and columns marked keep=True.
    """

    # Build AnnData.
    adata = make_stage_adata()

    # Build cell decisions.
    cell_decisions = pd.DataFrame(
        {
            "keep": [True, False, True, False],
            "fail_any_qc": [False, True, False, True],
            "failed_rules": ["", "rule", "", "rule"],
        },
        index=adata.obs_names,
    )

    # Build gene decisions.
    gene_decisions = pd.DataFrame(
        {
            "keep": [False, True, True],
            "fail_any_qc": [True, False, False],
            "failed_rules": ["rule", "", ""],
        },
        index=adata.var_names,
    )

    # Build decision result.
    decision_result = QCDecisionResult(
        cell_decisions=cell_decisions,
        gene_decisions=gene_decisions,
        summary={},
    )

    # Apply filtering.
    filtered = apply_qc_filter_to_adata(
        adata=adata,
        decision_result=decision_result,
        config=make_stage_config(mode="filter"),
    )

    # Confirm filtered shape.
    assert filtered.shape == (2, 2)

    # Confirm retained cells.
    assert list(filtered.obs_names) == ["cell_1", "cell_3"]

    # Confirm retained genes.
    assert list(filtered.var_names) == ["ACTB", "MALAT1"]


def test_apply_qc_filter_to_adata_returns_original_when_not_filtering() -> None:
    """
    Verify report-only mode does not filter AnnData.

    QC report mode should calculate decisions without mutating the data object.
    """

    # Build AnnData.
    adata = make_stage_adata()

    # Build empty-ish decision result.
    decision_result = QCDecisionResult(
        cell_decisions=pd.DataFrame(index=adata.obs_names),
        gene_decisions=pd.DataFrame(index=adata.var_names),
        summary={},
    )

    # Apply report-only behavior.
    observed = apply_qc_filter_to_adata(
        adata=adata,
        decision_result=decision_result,
        config=make_stage_config(mode="report_only"),
    )

    # Confirm original object was returned.
    assert observed is adata


def test_validate_decision_index_alignment_accepts_matching_index() -> None:
    """
    Verify decision alignment accepts matching indexes.

    Filtering uses positional masks, so the names and lengths must match exactly.
    """

    # Build expected index.
    expected = pd.Index(["a", "b"])

    # Confirm matching index passes.
    validate_decision_index_alignment(
        expected_index=expected,
        observed_index=pd.Index(["a", "b"]),
        axis_name="obs",
    )


def test_validate_decision_index_alignment_rejects_length_mismatch() -> None:
    """
    Verify decision alignment rejects length mismatches.

    A length mismatch would filter the wrong rows or columns.
    """

    # Confirm length mismatch fails clearly.
    with pytest.raises(QCStageError, match="has length"):
        validate_decision_index_alignment(
            expected_index=pd.Index(["a", "b"]),
            observed_index=pd.Index(["a"]),
            axis_name="obs",
        )


def test_validate_decision_index_alignment_rejects_value_mismatch() -> None:
    """
    Verify decision alignment rejects name mismatches.

    A name mismatch means the decision table is stale or from a different object.
    """

    # Confirm value mismatch fails clearly.
    with pytest.raises(QCStageError, match="does not match"):
        validate_decision_index_alignment(
            expected_index=pd.Index(["a", "b"]),
            observed_index=pd.Index(["a", "c"]),
            axis_name="var",
        )


def test_run_qc_workflow_report_only_writes_artifacts_without_filtering(tmp_path: Path) -> None:
    """
    Verify the standalone QC workflow runs in report-only mode.

    Report-only mode should write artifacts and return the unfiltered AnnData
    object while still producing decisions.
    """

    # Build input AnnData.
    adata = make_stage_adata()

    # Run the workflow.
    result = run_qc_workflow(
        adata=adata,
        config=make_stage_config(mode="report_only"),
        output_dir=tmp_path / "qc",
        write_artifacts=True,
        summary_extra={"run_id": "test"},
    )

    # Confirm structured workflow result.
    assert isinstance(result, QCWorkflowResult)

    # Confirm report-only mode did not filter.
    assert result.adata is adata
    assert result.adata.shape == (4, 3)

    # Confirm artifacts were written.
    assert result.artifact_manifest is not None
    assert result.artifact_manifest.get_path("summary").exists()
    assert result.artifact_manifest.get_path("cell_metrics").exists()

    # Confirm decisions exist.
    assert result.decision_result.summary["n_cells"] == 4
    assert result.decision_result.summary["n_genes"] == 3

    # Confirm stage metrics include shapes.
    assert result.metrics["input_shape"] == {"n_obs": 4, "n_vars": 3}
    assert result.metrics["output_shape"] == {"n_obs": 4, "n_vars": 3}


def test_run_qc_workflow_filter_mode_filters_cells_and_genes(tmp_path: Path) -> None:
    """
    Verify the standalone QC workflow filters AnnData in filter mode.

    The test config should remove the low-count cell and the gene detected in
    only that failed cell.
    """

    # Build input AnnData.
    adata = make_stage_adata()

    # Run the workflow in filter mode.
    result = run_qc_workflow(
        adata=adata,
        config=make_stage_config(mode="filter"),
        output_dir=tmp_path / "qc",
        write_artifacts=True,
    )

    # Confirm a filtered copy was returned.
    assert result.adata is not adata

    # Confirm filtered shape.
    assert result.adata.shape == (3, 2)

    # Confirm low-count cell was removed.
    assert list(result.adata.obs_names) == ["cell_1", "cell_2", "cell_3"]

    # Confirm gene detected only in the failed low-count cell was removed.
    assert list(result.adata.var_names) == ["ACTB", "MALAT1"]

    # Confirm filtering note was emitted.
    assert "QC filtering retained 3/4 cells and 2/3 genes." in result.notes


def test_run_qc_workflow_without_artifacts_allows_missing_output_dir() -> None:
    """
    Verify standalone workflow can run without artifact writing.

    This makes unit tests and notebook experiments faster.
    """

    # Run the workflow without artifacts.
    result = run_qc_workflow(
        adata=make_stage_adata(),
        config=make_stage_config(),
        output_dir=None,
        write_artifacts=False,
    )

    # Confirm no artifact manifest was produced.
    assert result.artifact_manifest is None

    # Confirm metrics were still produced.
    assert result.metrics["enabled"] is True


def test_run_qc_workflow_requires_output_dir_when_writing_artifacts() -> None:
    """
    Verify artifact writing requires an explicit output directory.

    This prevents artifacts from being silently written to the process working
    directory.
    """

    # Confirm missing output_dir fails clearly when writing artifacts.
    with pytest.raises(QCStageError, match="requires an output_dir"):
        run_qc_workflow(
            adata=make_stage_adata(),
            config=make_stage_config(),
            output_dir=None,
            write_artifacts=True,
        )


def test_run_qc_workflow_rejects_invalid_config() -> None:
    """
    Verify standalone workflow rejects invalid config objects.

    Invalid config types should fail before any QC work begins.
    """

    # Confirm invalid config fails clearly.
    with pytest.raises(QCStageError, match="QCConfig object"):
        run_qc_workflow(
            adata=make_stage_adata(),
            config={"bad": "config"},  # type: ignore[arg-type]
            write_artifacts=False,
        )


def test_run_qc_workflow_rejects_invalid_anndata() -> None:
    """
    Verify standalone workflow rejects invalid AnnData objects.

    The stage wrapper should provide a clear error before metric calculation.
    """

    # Confirm invalid AnnData input fails clearly.
    with pytest.raises(QCStageError, match="AnnData object"):
        run_qc_workflow(
            adata={"bad": "adata"},  # type: ignore[arg-type]
            config=make_stage_config(),
            write_artifacts=False,
        )


def test_qc_stage_run_returns_stage_result_and_artifacts(tmp_path: Path) -> None:
    """
    Verify QCStage.run returns a generic StageResult.

    This is the main test that the QC module satisfies the pipeline stage
    contract.
    """

    # Build pipeline context.
    context = make_pipeline_context(tmp_path=tmp_path)

    # Build the QC stage.
    stage = QCStage()

    # Run the stage.
    result = stage.run(context)

    # Confirm a generic StageResult was returned.
    assert isinstance(result, StageResult)

    # Confirm report-only mode did not filter.
    assert result.adata.shape == (4, 3)

    # Confirm artifacts were converted to StageArtifact records.
    assert result.artifacts
    assert all(isinstance(artifact, StageArtifact) for artifact in result.artifacts)

    # Confirm expected artifact names are present.
    assert {artifact.name for artifact in result.artifacts} == {
        "cell_metrics",
        "gene_metrics",
        "feature_masks",
        "thresholds",
        "cell_decisions",
        "gene_decisions",
        "qc_h5ad",
        "summary",
    }

    # Confirm artifacts were written under results/qc.
    assert all(
        context.paths.results / "qc" in artifact.path.parents for artifact in result.artifacts
    )

    # Confirm stage metrics are populated.
    assert result.metrics["enabled"] is True
    assert result.metrics["mode"] == "report_only"


def test_qc_stage_run_filters_when_configured(tmp_path: Path) -> None:
    """
    Verify QCStage.run returns filtered AnnData in filter mode.

    The QC stage should use the same filtering behavior as run_qc_workflow.
    """

    # Build filter-mode context.
    context = make_pipeline_context(
        tmp_path=tmp_path,
        config=make_stage_config(mode="filter"),
    )

    # Run the stage.
    result = QCStage().run(context)

    # Confirm filtered shape.
    assert result.adata.shape == (3, 2)

    # Confirm output-shape metrics.
    assert result.metrics["output_shape"] == {"n_obs": 3, "n_vars": 2}


def test_qc_stage_run_returns_noop_when_disabled(tmp_path: Path) -> None:
    """
    Verify disabled QC returns a no-op StageResult.

    Disabled stages should be explicit and auditable without touching artifacts.
    """

    # Build disabled config.
    config = make_stage_config(enabled=False)

    # Build pipeline context.
    context = make_pipeline_context(tmp_path=tmp_path, config=config)

    # Run the stage.
    result = QCStage().run(context)

    # Confirm original AnnData was returned.
    assert result.adata is context.adata

    # Confirm no artifacts were written.
    assert result.artifacts == []

    # Confirm skip note was emitted.
    assert result.notes == ["QC stage was skipped because qc.enabled is false."]

    # Confirm disabled metrics were emitted.
    assert result.metrics["enabled"] is False


def test_qc_stage_run_rejects_invalid_context() -> None:
    """
    Verify QCStage.run rejects non-PipelineContext objects.

    Pipeline stages should fail clearly when called with the wrong object.
    """

    # Confirm invalid context fails clearly.
    with pytest.raises(QCStageError, match="PipelineContext"):
        QCStage().run({"not": "context"})


def test_qc_stage_run_requires_anndata(tmp_path: Path) -> None:
    """
    Verify QCStage.run requires AnnData in the context.

    Context.require_adata should produce the missing-input error.
    """

    # Build context without AnnData.
    context = make_pipeline_context(tmp_path=tmp_path, adata=None)

    # Explicitly remove AnnData from the context.
    context.adata = None

    # Confirm missing AnnData fails.
    with pytest.raises(RuntimeError, match="does not contain an AnnData object"):
        QCStage().run(context)


def test_stage_artifacts_from_qc_manifest_converts_written_artifacts(tmp_path: Path) -> None:
    """
    Verify QC artifact manifests convert into StageArtifact records.

    Generic pipeline reports should not need to know about QCArtifactManifest.
    """

    # Run a workflow to create a real manifest.
    workflow = run_qc_workflow(
        adata=make_stage_adata(),
        config=make_stage_config(),
        output_dir=tmp_path / "qc",
        write_artifacts=True,
    )

    # Convert manifest to stage artifacts.
    artifacts = stage_artifacts_from_qc_manifest(workflow.artifact_manifest)  # type: ignore[arg-type]

    # Confirm artifacts were converted.
    assert artifacts

    # Confirm artifact kinds were inferred.
    assert {artifact.kind for artifact in artifacts} == {"csv", "json", "h5ad"}

    # Confirm descriptions are populated.
    assert all(artifact.description for artifact in artifacts)


def test_infer_artifact_kind_maps_known_suffixes() -> None:
    """
    Verify artifact kind inference maps known file suffixes.

    This keeps StageArtifact metadata concise and consistent.
    """

    # Confirm CSV kind.
    assert infer_artifact_kind(Path("x.csv")) == "csv"

    # Confirm JSON kind.
    assert infer_artifact_kind(Path("x.json")) == "json"

    # Confirm h5ad kind.
    assert infer_artifact_kind(Path("x.h5ad")) == "h5ad"

    # Confirm fallback kind.
    assert infer_artifact_kind(Path("x.txt")) == "file"


def test_describe_qc_artifact_returns_known_and_fallback_descriptions() -> None:
    """
    Verify artifact descriptions are stable for known artifact names.

    Descriptions appear in stage summaries and reports.
    """

    # Confirm known description.
    assert describe_qc_artifact("cell_metrics") == "Cell-level QC metric table."

    # Confirm fallback description.
    assert describe_qc_artifact("custom") == "QC artifact: custom."


def test_summarize_adata_shape_for_stage_returns_counts() -> None:
    """
    Verify AnnData shape summaries are JSON-friendly.

    Stage metrics should contain simple integer counts.
    """

    # Build AnnData.
    adata = make_stage_adata()

    # Confirm shape summary.
    assert summarize_adata_shape_for_stage(adata) == {"n_obs": 4, "n_vars": 3}


def test_build_qc_stage_notes_reports_filtering_and_report_only() -> None:
    """
    Verify stage notes explain whether filtering happened.

    Notes should be human-readable enough for reports.
    """

    # Build input and output AnnData.
    input_adata = make_stage_adata()
    output_adata = input_adata[:3, :2].copy()

    # Build filter-mode notes.
    filter_notes = build_qc_stage_notes(
        config=make_stage_config(mode="filter"),
        input_adata=input_adata,
        output_adata=output_adata,
        preparation_notes=["prepared"],
    )

    # Confirm filter notes include retention.
    assert filter_notes == [
        "prepared",
        "QC completed in filter mode.",
        "QC filtering retained 3/4 cells and 2/3 genes.",
    ]

    # Build report-only notes.
    report_notes = build_qc_stage_notes(
        config=make_stage_config(mode="report_only"),
        input_adata=input_adata,
        output_adata=input_adata,
        preparation_notes=[],
    )

    # Confirm report-only notes mention no filtering.
    assert report_notes == [
        "QC completed in report_only mode.",
        "QC decisions were reported but AnnData was not filtered.",
    ]


def test_collect_qc_stage_warnings_deduplicates_sources(tmp_path: Path) -> None:
    """
    Verify warning collection combines and de-duplicates all QC warning sources.

    Warnings can originate from validation, metrics, thresholds, decisions, and
    artifacts.
    """

    # Run a workflow to get realistic result objects.
    workflow = run_qc_workflow(
        adata=make_stage_adata(),
        config=make_stage_config(outputs={"write_figures": True, "write_h5ad": False}),
        output_dir=tmp_path / "qc",
        write_artifacts=True,
    )

    # Collect warnings again from workflow pieces.
    warnings = collect_qc_stage_warnings(
        validation_summary=workflow.validation_summary,
        metrics_result=workflow.metrics_result,
        threshold_result=workflow.threshold_result,
        decision_result=workflow.decision_result,
        artifact_manifest=workflow.artifact_manifest,
    )

    # Confirm artifact warning is present only once.
    assert (
        warnings.count(
            "QCOutputConfig.write_figures is true, but QC figure generation is not "
            "implemented in artifacts.py yet."
        )
        == 1
    )


def test_deduplicate_strings_preserves_first_seen_order() -> None:
    """
    Verify string de-duplication preserves first-seen order.

    Warning order should remain deterministic.
    """

    # Confirm duplicates are removed in first-seen order.
    assert deduplicate_strings(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_apply_qc_filter_to_adata_rejects_misaligned_gene_decisions() -> None:
    """
    Verify filtering rejects gene decision indexes from a different object.

    This protects raw.X or stale decision-table mismatches from silently removing
    the wrong genes.
    """

    # Build AnnData.
    adata = make_stage_adata()

    # Build aligned cell decisions.
    cell_decisions = pd.DataFrame(
        {
            "keep": [True, True, True, True],
            "fail_any_qc": [False, False, False, False],
            "failed_rules": ["", "", "", ""],
        },
        index=adata.obs_names,
    )

    # Build misaligned gene decisions.
    gene_decisions = pd.DataFrame(
        {
            "keep": [True, True, True],
            "fail_any_qc": [False, False, False],
            "failed_rules": ["", "", ""],
        },
        index=["wrong_1", "wrong_2", "wrong_3"],
    )

    # Build decision result.
    decision_result = QCDecisionResult(
        cell_decisions=cell_decisions,
        gene_decisions=gene_decisions,
        summary={},
    )

    # Confirm filtering fails clearly.
    with pytest.raises(QCStageError, match="does not match"):
        apply_qc_filter_to_adata(
            adata=adata,
            decision_result=decision_result,
            config=make_stage_config(mode="filter"),
        )


def test_run_qc_workflow_propagates_empty_filter_error() -> None:
    """
    Verify workflow propagates decision-layer empty filter errors.

    This confirms the stage does not hide unsafe all-cell/all-gene filtering.
    """

    # Build config that filters every cell.
    config = make_stage_config(
        mode="filter",
        basic={
            "min_genes_per_cell": 999,
            "max_genes_per_cell": None,
            "min_counts_per_cell": None,
            "max_counts_per_cell": None,
            "min_cells_per_gene": None,
            "max_mito_percent": None,
            "max_ribo_percent": None,
            "max_hemoglobin_percent": None,
        },
        fail_on_empty_result=True,
    )

    # Confirm all-cell filtering fails.
    with pytest.raises(Exception, match="remove all cells"):
        run_qc_workflow(
            adata=make_stage_adata(),
            config=config,
            write_artifacts=False,
        )


def test_run_qc_workflow_contains_expected_threshold_rule() -> None:
    """
    Verify workflow threshold and decision outputs include expected rule names.

    This gives a small integration check across metrics, thresholds, and
    decisions.
    """

    # Run workflow without artifact writing.
    result = run_qc_workflow(
        adata=make_stage_adata(),
        config=make_stage_config(),
        write_artifacts=False,
    )

    # Confirm expected threshold rule exists.
    assert {threshold.rule_name for threshold in result.threshold_result.thresholds} == {
        "fixed_min_genes_per_cell",
        "fixed_min_counts_per_cell",
        "fixed_max_mito_percent",
        "fixed_min_cells_per_gene",
    }

    # Confirm corresponding decision columns exist.
    assert "fixed_min_counts_per_cell" in result.decision_result.cell_decisions.columns
    assert "fixed_min_cells_per_gene" in result.decision_result.gene_decisions.columns
