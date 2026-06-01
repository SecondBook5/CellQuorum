"""Tests for CellQuorum QC feature-family annotation utilities."""

from __future__ import annotations

# Import AnnData for constructing feature-annotation test objects.
import anndata as ad

# Import NumPy for deterministic test matrices.
import numpy as np

# Import pandas for direct feature-mask table tests.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import QC feature-pattern configuration.
from cellquorum.qc.config import QCFeaturePatternConfig

# Import feature annotation utilities under test.
from cellquorum.qc.features import (
    CUSTOM_EXCLUDE_COLUMN,
    HEMOGLOBIN_COLUMN,
    MITO_COLUMN,
    RIBO_COLUMN,
    QCFeatureAnnotationError,
    QCFeatureMaskSummary,
    annotate_qc_feature_masks,
    build_feature_masks,
    summarize_feature_masks,
)


def make_feature_test_adata() -> ad.AnnData:
    """
    Build a small AnnData object with biologically meaningful feature names.

    The feature names are chosen to test mitochondrial, ribosomal, hemoglobin,
    custom-exclude, and negative-control matching behavior.

    Returns:
        Small AnnData object for feature annotation tests.
    """

    # Build a small count-like matrix.
    matrix = np.ones((3, 10), dtype=float)

    # Define feature names covering QC feature families and negative controls.
    feature_names = [
        "MT-ND1",
        "mt-Nd1",
        "RPS3",
        "RPLP0",
        "HBA1",
        "HBB",
        "HBM",
        "HBX",
        "ACTB",
        "MALAT1",
    ]

    # Return the test AnnData object.
    return ad.AnnData(X=matrix, var=pd.DataFrame(index=feature_names))


def test_build_feature_masks_uses_human_defaults() -> None:
    """
    Verify default human-style QC feature-family matching.

    By default, CellQuorum should identify human mitochondrial genes with MT-,
    ribosomal genes with RPS/RPL, and hemoglobin genes with the configured
    hemoglobin regex.
    """

    # Build the test AnnData object.
    adata = make_feature_test_adata()

    # Build feature masks with default configuration.
    masks = build_feature_masks(adata)

    # Confirm the mask table preserves variable order.
    assert list(masks.index) == list(adata.var_names)

    # Confirm all expected mask columns are present.
    assert list(masks.columns) == [
        MITO_COLUMN,
        RIBO_COLUMN,
        HEMOGLOBIN_COLUMN,
        CUSTOM_EXCLUDE_COLUMN,
    ]

    # Confirm human mitochondrial prefix matching.
    assert bool(masks.loc["MT-ND1", MITO_COLUMN]) is True

    # Confirm mouse-style mitochondrial names are not matched by default.
    assert bool(masks.loc["mt-Nd1", MITO_COLUMN]) is False

    # Confirm ribosomal prefix matching.
    assert bool(masks.loc["RPS3", RIBO_COLUMN]) is True
    assert bool(masks.loc["RPLP0", RIBO_COLUMN]) is True

    # Confirm hemoglobin regex matching.
    assert bool(masks.loc["HBA1", HEMOGLOBIN_COLUMN]) is True
    assert bool(masks.loc["HBB", HEMOGLOBIN_COLUMN]) is True
    assert bool(masks.loc["HBM", HEMOGLOBIN_COLUMN]) is True

    # Confirm a non-matching HB-like gene is not overcalled as hemoglobin.
    assert bool(masks.loc["HBX", HEMOGLOBIN_COLUMN]) is False

    # Confirm housekeeping genes are not assigned to QC feature families.
    assert bool(masks.loc["ACTB", MITO_COLUMN]) is False
    assert bool(masks.loc["ACTB", RIBO_COLUMN]) is False
    assert bool(masks.loc["ACTB", HEMOGLOBIN_COLUMN]) is False


def test_build_feature_masks_supports_mouse_mitochondrial_prefix() -> None:
    """
    Verify mitochondrial prefix configuration supports mouse-style names.

    Mouse mitochondrial genes often use lowercase mt-. The feature config should
    support this without changing the core matching logic.
    """

    # Build the test AnnData object.
    adata = make_feature_test_adata()

    # Build a mouse-style mitochondrial feature configuration.
    config = QCFeaturePatternConfig(mitochondrial_prefixes=["mt-"])

    # Build feature masks with mouse-style mitochondrial matching.
    masks = build_feature_masks(adata, config)

    # Confirm lowercase mt- is matched.
    assert bool(masks.loc["mt-Nd1", MITO_COLUMN]) is True

    # Confirm uppercase MT- is not matched when only mt- is configured.
    assert bool(masks.loc["MT-ND1", MITO_COLUMN]) is False


def test_build_feature_masks_supports_multiple_mitochondrial_prefixes() -> None:
    """
    Verify mitochondrial matching can support multiple references at once.

    Some workflows may combine species or references, so multiple mitochondrial
    prefixes should be allowed.
    """

    # Build the test AnnData object.
    adata = make_feature_test_adata()

    # Build a feature config with both human and mouse mitochondrial prefixes.
    config = QCFeaturePatternConfig(mitochondrial_prefixes=["MT-", "mt-"])

    # Build feature masks.
    masks = build_feature_masks(adata, config)

    # Confirm both mitochondrial naming styles are matched.
    assert bool(masks.loc["MT-ND1", MITO_COLUMN]) is True
    assert bool(masks.loc["mt-Nd1", MITO_COLUMN]) is True


def test_build_feature_masks_supports_custom_exclude_prefixes() -> None:
    """
    Verify custom-exclude prefixes are matched.

    Custom exclusions are useful for project-specific QC review features such as
    MALAT1, ERCC spike-ins, or other technical feature groups.
    """

    # Build the test AnnData object.
    adata = make_feature_test_adata()

    # Build a feature config with a custom MALAT1 prefix.
    config = QCFeaturePatternConfig(custom_exclude_prefixes=["MALAT"])

    # Build feature masks.
    masks = build_feature_masks(adata, config)

    # Confirm MALAT1 is flagged as custom-excluded.
    assert bool(masks.loc["MALAT1", CUSTOM_EXCLUDE_COLUMN]) is True

    # Confirm unrelated genes are not custom-excluded.
    assert bool(masks.loc["ACTB", CUSTOM_EXCLUDE_COLUMN]) is False


def test_build_feature_masks_allows_empty_pattern_lists() -> None:
    """
    Verify that empty feature-pattern lists produce all-False masks.

    Users should be able to disable specific feature-family matching by providing
    empty pattern lists.
    """

    # Build the test AnnData object.
    adata = make_feature_test_adata()

    # Build a config with all matching patterns disabled.
    config = QCFeaturePatternConfig(
        mitochondrial_prefixes=[],
        ribosomal_prefixes=[],
        hemoglobin_regexes=[],
        custom_exclude_prefixes=[],
    )

    # Build feature masks.
    masks = build_feature_masks(adata, config)

    # Confirm all feature-family counts are zero.
    assert int(masks[MITO_COLUMN].sum()) == 0
    assert int(masks[RIBO_COLUMN].sum()) == 0
    assert int(masks[HEMOGLOBIN_COLUMN].sum()) == 0
    assert int(masks[CUSTOM_EXCLUDE_COLUMN].sum()) == 0


def test_build_feature_masks_preserves_duplicate_var_names() -> None:
    """
    Verify feature masks can be built when variable names are duplicated.

    Duplicate-name handling is configured separately. Feature-mask construction
    should still return one mask row per variable position.
    """

    # Build a matrix with duplicate variable names.
    adata = ad.AnnData(
        X=np.ones((2, 3), dtype=float),
        var=pd.DataFrame(index=["MT-ND1", "MT-ND1", "ACTB"]),
    )

    # Build feature masks.
    masks = build_feature_masks(adata)

    # Confirm one mask row exists per variable position.
    assert masks.shape == (3, 4)

    # Confirm both duplicated mitochondrial rows are flagged.
    assert masks[MITO_COLUMN].tolist() == [True, True, False]


def test_build_feature_masks_rejects_non_anndata_input() -> None:
    """
    Verify feature-mask construction rejects non-AnnData inputs.

    The feature annotation layer assumes AnnData variable names.
    """

    # Confirm non-AnnData input fails clearly.
    with pytest.raises(QCFeatureAnnotationError, match="AnnData object"):
        build_feature_masks({"not": "anndata"})  # type: ignore[arg-type]


def test_build_feature_masks_rejects_non_feature_config() -> None:
    """
    Verify feature-mask construction rejects invalid config objects.

    Passing a dictionary instead of a QCFeaturePatternConfig should fail clearly.
    """

    # Build the test AnnData object.
    adata = make_feature_test_adata()

    # Confirm invalid config input fails clearly.
    with pytest.raises(QCFeatureAnnotationError, match="QCFeaturePatternConfig"):
        build_feature_masks(adata, {"mitochondrial_prefixes": ["MT-"]})  # type: ignore[arg-type]


def test_build_feature_masks_rejects_zero_variable_anndata() -> None:
    """
    Verify feature-mask construction rejects AnnData objects with zero variables.

    Feature-family annotation is meaningless without variables.
    """

    # Build an AnnData object with zero variables.
    adata = ad.AnnData(X=np.ones((2, 0), dtype=float))

    # Confirm zero-variable inputs fail clearly.
    with pytest.raises(QCFeatureAnnotationError, match="zero variables"):
        build_feature_masks(adata)


def test_build_feature_masks_rejects_invalid_regex() -> None:
    """
    Verify invalid hemoglobin regex patterns fail clearly.

    Regex syntax errors should be caught before metrics use malformed masks.
    """

    # Build the test AnnData object.
    adata = make_feature_test_adata()

    # Build a config with an invalid regex pattern.
    config = QCFeaturePatternConfig(hemoglobin_regexes=["["])

    # Confirm invalid regex syntax fails clearly.
    with pytest.raises(QCFeatureAnnotationError, match="Invalid QC feature regex"):
        build_feature_masks(adata, config)


def test_annotate_qc_feature_masks_mutates_input_by_default() -> None:
    """
    Verify AnnData.var annotation mutates the input object by default.

    This behavior is useful inside stages where the QC stage owns the working
    AnnData object.
    """

    # Build the test AnnData object.
    adata = make_feature_test_adata()

    # Annotate the AnnData object in place.
    returned = annotate_qc_feature_masks(adata)

    # Confirm the same object was returned.
    assert returned is adata

    # Confirm feature-mask columns were written to AnnData.var.
    assert MITO_COLUMN in adata.var.columns
    assert RIBO_COLUMN in adata.var.columns
    assert HEMOGLOBIN_COLUMN in adata.var.columns
    assert CUSTOM_EXCLUDE_COLUMN in adata.var.columns

    # Confirm a known mitochondrial gene was annotated.
    assert bool(adata.var.loc["MT-ND1", MITO_COLUMN]) is True


def test_annotate_qc_feature_masks_can_return_copy() -> None:
    """
    Verify AnnData.var annotation can operate on a copy.

    Copy mode is useful for APIs and tests that should not mutate caller-owned
    AnnData objects.
    """

    # Build the test AnnData object.
    adata = make_feature_test_adata()

    # Annotate a copy.
    annotated = annotate_qc_feature_masks(adata, copy=True)

    # Confirm a new object was returned.
    assert annotated is not adata

    # Confirm the original AnnData object was not annotated.
    assert MITO_COLUMN not in adata.var.columns

    # Confirm the copied AnnData object was annotated.
    assert MITO_COLUMN in annotated.var.columns


def test_summarize_feature_masks_returns_expected_counts() -> None:
    """
    Verify feature-mask summaries report expected feature-family counts.

    Summary counts will later be used in stage metrics, reports, and provenance.
    """

    # Build the test AnnData object.
    adata = make_feature_test_adata()

    # Build feature masks with default configuration.
    masks = build_feature_masks(adata)

    # Summarize the feature masks.
    summary = summarize_feature_masks(masks)

    # Confirm a structured summary was returned.
    assert isinstance(summary, QCFeatureMaskSummary)

    # Confirm the total variable count.
    assert summary.n_vars == 10

    # Confirm the mitochondrial feature count.
    assert summary.n_mito == 1

    # Confirm the ribosomal feature count.
    assert summary.n_ribo == 2

    # Confirm the hemoglobin feature count.
    assert summary.n_hemoglobin == 3

    # Confirm no custom-excluded features by default.
    assert summary.n_custom_exclude == 0


def test_summarize_feature_masks_serializes_to_dict() -> None:
    """
    Verify feature-mask summaries serialize to dictionaries.

    The summary should be easy to write into provenance and stage-level metrics.
    """

    # Build the test AnnData object.
    adata = make_feature_test_adata()

    # Build and summarize feature masks.
    summary = summarize_feature_masks(build_feature_masks(adata))

    # Convert the summary to a dictionary.
    payload = summary.to_dict()

    # Confirm the serialized payload is correct.
    assert payload == {
        "n_vars": 10,
        "n_mito": 1,
        "n_ribo": 2,
        "n_hemoglobin": 3,
        "n_custom_exclude": 0,
    }


def test_summarize_feature_masks_rejects_non_dataframe() -> None:
    """
    Verify feature-mask summarization rejects non-DataFrame inputs.

    This prevents confusing attribute errors when callers pass the wrong object.
    """

    # Confirm non-DataFrame inputs fail clearly.
    with pytest.raises(QCFeatureAnnotationError, match="pandas DataFrame"):
        summarize_feature_masks({"not": "dataframe"})  # type: ignore[arg-type]


def test_summarize_feature_masks_rejects_missing_columns() -> None:
    """
    Verify feature-mask summarization rejects incomplete mask tables.

    Summary logic depends on all required feature-mask columns being present.
    """

    # Build an incomplete feature-mask table.
    masks = pd.DataFrame({MITO_COLUMN: [True, False]})

    # Confirm missing required columns fail clearly.
    with pytest.raises(QCFeatureAnnotationError, match="missing required column"):
        summarize_feature_masks(masks)
