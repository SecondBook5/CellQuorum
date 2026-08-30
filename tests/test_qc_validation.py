"""Tests for CellQuorum QC AnnData validation utilities."""

from __future__ import annotations

# Import AnnData for constructing small QC test objects.
import anndata as ad

# Import NumPy for deterministic dense matrices.
import numpy as np

# Import pandas for AnnData.obs metadata construction.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import scipy sparse matrices for sparse validation tests.
import scipy.sparse as sp

# Import QC configuration objects used by validation.
from cellquorum.stages.qc.config import QCConfig, QCDuplicateNameConfig

# Import QC validation utilities under test.
from cellquorum.stages.qc.validation import (
    QCInputValidationError,
    QCInputValidationSummary,
    get_qc_matrix,
    require_obs_columns,
    summarize_adata_shape,
    validate_duplicate_name_policy,
    validate_mad_groupby_columns,
    validate_qc_input_adata,
    validate_qc_matrix,
)


def make_test_adata() -> ad.AnnData:
    """
    Build a tiny valid AnnData object for QC validation tests.

    The matrix is intentionally small and count-like. It includes observation
    metadata so group-wise MAD validation can be tested without involving larger
    pipeline machinery.

    Returns:
        Small valid AnnData object.
    """

    # Build a deterministic count-like dense matrix.
    matrix = np.array(
        [
            [1, 0, 3],
            [0, 2, 1],
            [4, 0, 0],
        ],
        dtype=float,
    )

    # Build observation metadata with sample and batch fields.
    obs = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_1", "sample_2"],
            "batch": ["batch_1", "batch_1", "batch_2"],
        },
        index=["cell_1", "cell_2", "cell_3"],
    )

    # Build variable metadata.
    var = pd.DataFrame(index=["MT-ND1", "RPS3", "ACTB"])

    # Return the AnnData object.
    return ad.AnnData(X=matrix, obs=obs, var=var)


def test_validate_qc_input_adata_accepts_valid_anndata() -> None:
    """
    Verify that a valid AnnData object passes QC input validation.

    A valid QC input must be AnnData, non-empty, numeric, finite, non-negative,
    and compatible with the requested QC configuration.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Validate the QC input.
    summary = validate_qc_input_adata(adata)

    # Confirm a structured validation summary was returned.
    assert isinstance(summary, QCInputValidationSummary)

    # Confirm the AnnData dimensions were recorded.
    assert summary.n_obs == 3
    assert summary.n_vars == 3

    # Confirm the selected matrix dimensions were recorded.
    assert summary.matrix_n_obs == 3
    assert summary.matrix_n_vars == 3

    # Confirm AnnData.X was used by default.
    assert summary.matrix_source == "X"

    # Confirm AnnData.raw is absent.
    assert summary.has_raw is False

    # Confirm observation names are unique.
    assert summary.obs_names_unique is True

    # Confirm variable names are unique.
    assert summary.var_names_unique is True

    # Confirm no groupby columns were requested by default.
    assert summary.requested_groupby == ()

    # Confirm no validation warnings were emitted.
    assert summary.warnings == ()


def test_validate_qc_input_adata_summary_serializes_to_dict() -> None:
    """
    Verify that QC validation summaries serialize to dictionaries.

    The validation summary should be usable in provenance and stage notes without
    requiring custom conversion logic.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Validate the QC input.
    summary = validate_qc_input_adata(adata)

    # Convert the summary to a dictionary.
    payload = summary.to_dict()

    # Confirm the dictionary stores the observation count.
    assert payload["n_obs"] == 3

    # Confirm the dictionary stores the variable count.
    assert payload["n_vars"] == 3

    # Confirm requested_groupby is JSON-friendly.
    assert payload["requested_groupby"] == []

    # Confirm warnings are JSON-friendly.
    assert payload["warnings"] == []


def test_validate_qc_input_adata_rejects_non_anndata() -> None:
    """
    Verify that QC validation rejects non-AnnData inputs.

    QC metric calculation should not receive arbitrary objects because the rest
    of the QC engine assumes AnnData semantics.
    """

    # Confirm non-AnnData input fails clearly.
    with pytest.raises(QCInputValidationError, match="must be an AnnData object"):
        validate_qc_input_adata({"not": "anndata"})


def test_validate_qc_input_adata_rejects_non_qc_config() -> None:
    """
    Verify that QC validation rejects non-QCConfig configuration objects.

    This catches accidental use of dictionaries or unrelated config models before
    downstream logic tries to access nested QC attributes.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Confirm invalid config input fails clearly.
    with pytest.raises(QCInputValidationError, match="config to be a QCConfig"):
        validate_qc_input_adata(adata, config={"not": "qcconfig"})  # type: ignore[arg-type]


def test_validate_qc_input_adata_rejects_zero_observations() -> None:
    """
    Verify that AnnData objects with zero observations fail validation.

    QC cannot calculate cell-level metrics if there are no observations.
    """

    # Build an AnnData object with zero observations.
    adata = ad.AnnData(X=np.ones((0, 3)))

    # Confirm empty observation inputs fail clearly.
    with pytest.raises(QCInputValidationError, match="at least one observation"):
        validate_qc_input_adata(adata)


def test_validate_qc_input_adata_rejects_zero_variables() -> None:
    """
    Verify that AnnData objects with zero variables fail validation.

    QC cannot calculate gene-level metrics if there are no variables.
    """

    # Build an AnnData object with zero variables.
    adata = ad.AnnData(X=np.ones((3, 0)))

    # Confirm empty variable inputs fail clearly.
    with pytest.raises(QCInputValidationError, match="at least one variable"):
        validate_qc_input_adata(adata)


def test_get_qc_matrix_returns_x_by_default() -> None:
    """
    Verify that AnnData.X is used when no layer or raw matrix is requested.

    This is the default QC matrix source for CellQuorum.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Build the default QC configuration.
    config = QCConfig()

    # Resolve the QC matrix.
    matrix, source = get_qc_matrix(adata, config)

    # Confirm AnnData.X was returned.
    assert matrix is adata.X

    # Confirm the matrix source label is X.
    assert source == "X"


def test_get_qc_matrix_returns_named_layer() -> None:
    """
    Verify that a configured AnnData layer can be used for QC metrics.

    This supports workflows where raw counts are stored in a layer such as
    `counts` while AnnData.X may later contain transformed values.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Store a count matrix layer.
    adata.layers["counts"] = adata.X.copy()

    # Build a QC configuration that requests the layer.
    config = QCConfig(metrics={"layer": "counts"})

    # Resolve the QC matrix.
    matrix, source = get_qc_matrix(adata, config)

    # Confirm the named layer was returned.
    assert matrix is adata.layers["counts"]

    # Confirm the matrix source label identifies the layer.
    assert source == "layers[counts]"


def test_get_qc_matrix_rejects_missing_layer() -> None:
    """
    Verify that missing configured AnnData layers fail validation.

    A misspelled layer name should fail before metric calculation.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Build a QC configuration that requests a missing layer.
    config = QCConfig(metrics={"layer": "counts"})

    # Confirm missing layer selection fails clearly.
    with pytest.raises(QCInputValidationError, match="layer 'counts'"):
        get_qc_matrix(adata, config)


def test_get_qc_matrix_returns_raw_x_when_requested() -> None:
    """
    Verify that AnnData.raw.X can be used for QC metrics when requested.

    Some workflows keep raw counts in AnnData.raw and transformed values in X.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Store the current object as AnnData.raw.
    adata.raw = adata

    # Build a QC configuration that requests AnnData.raw.X.
    config = QCConfig(metrics={"use_raw": True})

    # Resolve the QC matrix.
    matrix, source = get_qc_matrix(adata, config)

    # Confirm AnnData.raw.X was returned.
    assert matrix is adata.raw.X

    # Confirm the matrix source label identifies raw.X.
    assert source == "raw.X"


def test_get_qc_matrix_rejects_missing_raw() -> None:
    """
    Verify that requesting AnnData.raw fails when raw is missing.

    The validator should not silently fall back to AnnData.X when a user
    explicitly requests raw.X.
    """

    # Build a valid test AnnData object without raw.
    adata = make_test_adata()

    # Build a QC configuration that requests raw.X.
    config = QCConfig(metrics={"use_raw": True})

    # Confirm missing raw data fails clearly.
    with pytest.raises(QCInputValidationError, match="AnnData.raw is missing"):
        get_qc_matrix(adata, config)


def test_get_qc_matrix_rejects_simultaneous_layer_and_raw_requests() -> None:
    """
    Verify that layer and raw selection cannot both be requested.

    Choosing both matrix sources would make QC metric calculation ambiguous.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Store a count matrix layer.
    adata.layers["counts"] = adata.X.copy()

    # Store AnnData.raw.
    adata.raw = adata

    # Build an ambiguous matrix-source configuration.
    config = QCConfig(metrics={"use_raw": True, "layer": "counts"})

    # Confirm ambiguous matrix selection fails clearly.
    with pytest.raises(QCInputValidationError, match="cannot request both"):
        get_qc_matrix(adata, config)


def test_validate_qc_matrix_accepts_dense_numeric_matrix() -> None:
    """
    Verify that valid dense numeric matrices pass matrix validation.

    Dense matrices are common in small tests and some transformed workflows.
    """

    # Build a valid dense matrix.
    matrix = np.array([[1, 0], [2, 3]], dtype=float)

    # Validate the matrix.
    shape = validate_qc_matrix(matrix, expected_n_obs=2, matrix_source="X")

    # Confirm the matrix shape was returned.
    assert shape == (2, 2)


def test_validate_qc_matrix_accepts_sparse_numeric_matrix() -> None:
    """
    Verify that valid sparse numeric matrices pass matrix validation.

    Sparse count matrices are the normal representation for real scRNA-seq data.
    """

    # Build a valid sparse matrix.
    matrix = sp.csr_matrix(np.array([[1, 0], [2, 3]], dtype=float))

    # Validate the matrix.
    shape = validate_qc_matrix(matrix, expected_n_obs=2, matrix_source="X")

    # Confirm the matrix shape was returned.
    assert shape == (2, 2)


def test_validate_qc_matrix_rejects_object_without_shape() -> None:
    """
    Verify that QC matrices must expose shape.

    Matrix-like inputs without a shape cannot support dimension validation.
    """

    # Confirm shape-less objects fail clearly.
    with pytest.raises(QCInputValidationError, match="does not expose a shape"):
        validate_qc_matrix(object(), expected_n_obs=2, matrix_source="X")


def test_validate_qc_matrix_rejects_wrong_number_of_dimensions() -> None:
    """
    Verify that QC matrices must be two-dimensional.

    AnnData count matrices should be observations by variables.
    """

    # Build a one-dimensional matrix.
    matrix = np.array([1, 2, 3], dtype=float)

    # Confirm one-dimensional input fails clearly.
    with pytest.raises(QCInputValidationError, match="two-dimensional"):
        validate_qc_matrix(matrix, expected_n_obs=3, matrix_source="X")


def test_validate_qc_matrix_rejects_observation_mismatch() -> None:
    """
    Verify that QC matrix observation dimension must match AnnData.obs.

    Layer or raw matrices with the wrong observation count would misalign QC
    metrics with cells.
    """

    # Build a valid dense matrix with two observations.
    matrix = np.array([[1, 0], [2, 3]], dtype=float)

    # Confirm mismatched observation counts fail clearly.
    with pytest.raises(QCInputValidationError, match="but AnnData has 3 observations"):
        validate_qc_matrix(matrix, expected_n_obs=3, matrix_source="X")


def test_validate_qc_matrix_rejects_zero_variables() -> None:
    """
    Verify that selected QC matrices must contain at least one variable.

    Gene-level QC and cell-level count summaries require at least one feature.
    """

    # Build a matrix with zero variables.
    matrix = np.ones((2, 0), dtype=float)

    # Confirm zero-variable matrices fail clearly.
    with pytest.raises(QCInputValidationError, match="at least one variable"):
        validate_qc_matrix(matrix, expected_n_obs=2, matrix_source="X")


def test_validate_qc_matrix_rejects_non_numeric_dense_matrix() -> None:
    """
    Verify that dense QC matrices must be numeric.

    String/object matrices cannot support count-like QC metric calculation.
    """

    # Build a non-numeric dense matrix.
    matrix = np.array([["a", "b"], ["c", "d"]], dtype=object)

    # Confirm non-numeric matrices fail clearly.
    with pytest.raises(QCInputValidationError, match="must be numeric"):
        validate_qc_matrix(matrix, expected_n_obs=2, matrix_source="X")


def test_validate_qc_matrix_rejects_nan_dense_values() -> None:
    """
    Verify that dense QC matrices cannot contain NaN values.

    NaN values would propagate into QC metrics and filtering decisions.
    """

    # Build a dense matrix containing NaN.
    matrix = np.array([[1.0, np.nan], [2.0, 3.0]])

    # Confirm NaN values fail clearly.
    with pytest.raises(QCInputValidationError, match="NaN or infinite"):
        validate_qc_matrix(matrix, expected_n_obs=2, matrix_source="X")


def test_validate_qc_matrix_rejects_infinite_dense_values() -> None:
    """
    Verify that dense QC matrices cannot contain infinite values.

    Infinite values would make summaries and thresholds meaningless.
    """

    # Build a dense matrix containing infinity.
    matrix = np.array([[1.0, np.inf], [2.0, 3.0]])

    # Confirm infinite values fail clearly.
    with pytest.raises(QCInputValidationError, match="NaN or infinite"):
        validate_qc_matrix(matrix, expected_n_obs=2, matrix_source="X")


def test_validate_qc_matrix_rejects_negative_dense_values() -> None:
    """
    Verify that dense QC matrices cannot contain negative values.

    QC is intended for count-like non-negative matrices.
    """

    # Build a dense matrix containing a negative value.
    matrix = np.array([[1.0, -1.0], [2.0, 3.0]])

    # Confirm negative values fail clearly.
    with pytest.raises(QCInputValidationError, match="negative values"):
        validate_qc_matrix(matrix, expected_n_obs=2, matrix_source="X")


def test_validate_qc_matrix_rejects_nan_sparse_values() -> None:
    """
    Verify that sparse QC matrices cannot contain NaN stored values.

    Sparse matrices should still be checked through their stored data array.
    """

    # Build a sparse matrix containing NaN.
    matrix = sp.csr_matrix(np.array([[1.0, np.nan], [2.0, 3.0]]))

    # Confirm sparse NaN values fail clearly.
    with pytest.raises(QCInputValidationError, match="NaN or infinite"):
        validate_qc_matrix(matrix, expected_n_obs=2, matrix_source="X")


def test_validate_qc_matrix_rejects_negative_sparse_values() -> None:
    """
    Verify that sparse QC matrices cannot contain negative stored values.

    Negative values are invalid for count-like QC inputs.
    """

    # Build a sparse matrix containing a negative value.
    matrix = sp.csr_matrix(np.array([[1.0, -1.0], [2.0, 3.0]]))

    # Confirm sparse negative values fail clearly.
    with pytest.raises(QCInputValidationError, match="negative values"):
        validate_qc_matrix(matrix, expected_n_obs=2, matrix_source="X")


def test_validate_mad_groupby_columns_accepts_existing_columns() -> None:
    """
    Verify that requested MAD groupby columns can exist in AnnData.obs.

    Group-wise MAD thresholding should be allowed when all requested metadata
    columns are present.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Build a QC configuration requesting existing groupby columns.
    config = QCConfig(mad={"groupby": ["sample_id", "batch"]})

    # Confirm validation passes without raising.
    validate_mad_groupby_columns(adata, config)


def test_validate_mad_groupby_columns_rejects_missing_columns() -> None:
    """
    Verify that missing MAD groupby columns fail validation.

    Group-wise thresholding cannot proceed if requested metadata is absent.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Build a QC configuration requesting a missing groupby column.
    config = QCConfig(mad={"groupby": ["donor_id"]})

    # Confirm missing groupby columns fail clearly.
    with pytest.raises(QCInputValidationError, match="missing AnnData.obs column"):
        validate_mad_groupby_columns(adata, config)


def test_validate_mad_groupby_columns_skips_when_mad_disabled() -> None:
    """
    Verify that groupby validation is skipped when MAD thresholding is disabled.

    A fixed-threshold-only run should not fail because MAD-specific metadata is
    absent.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Build a fixed-only QC configuration with missing groupby metadata.
    config = QCConfig(
        threshold_strategy="fixed",
        mad={
            "enabled": False,
            "groupby": ["donor_id"],
        },
    )

    # Confirm validation does not raise when MAD is disabled.
    validate_mad_groupby_columns(adata, config)


def test_validate_qc_input_adata_records_requested_groupby_columns() -> None:
    """
    Verify that validation summaries record requested groupby columns.

    This supports later provenance and report output for group-wise QC settings.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Build a QC configuration requesting group-wise MAD thresholds.
    config = QCConfig(mad={"groupby": ["sample_id"]})

    # Validate the QC input.
    summary = validate_qc_input_adata(adata, config)

    # Confirm requested groupby columns were recorded.
    assert summary.requested_groupby == ("sample_id",)


def test_require_obs_columns_accepts_existing_columns() -> None:
    """
    Verify that required AnnData.obs columns pass when present.

    This helper will be reused by QC and later modules that need metadata fields.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Confirm existing columns pass without raising.
    require_obs_columns(adata, ["sample_id", "batch"])


def test_require_obs_columns_rejects_single_string_argument() -> None:
    """
    Verify that require_obs_columns rejects a single string.

    A string is iterable, so accepting it would accidentally validate one
    character at a time.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Confirm a single string fails clearly.
    with pytest.raises(QCInputValidationError, match="not a string"):
        require_obs_columns(adata, "sample_id")  # type: ignore[arg-type]


def test_require_obs_columns_rejects_missing_columns() -> None:
    """
    Verify that require_obs_columns reports missing metadata.

    Missing metadata should fail before methods attempt to use absent columns.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Confirm missing columns fail clearly.
    with pytest.raises(QCInputValidationError, match="missing required column"):
        require_obs_columns(adata, ["donor_id"])


def test_summarize_adata_shape_returns_shape_dictionary() -> None:
    """
    Verify that AnnData shape summaries are returned as simple dictionaries.

    Shape summaries are useful for provenance, smoke tests, and stage metrics.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Summarize the AnnData shape.
    summary = summarize_adata_shape(adata)

    # Confirm the shape dictionary is correct.
    assert summary == {
        "n_obs": 3,
        "n_vars": 3,
    }


def test_summarize_adata_shape_rejects_non_anndata() -> None:
    """
    Verify that shape summarization rejects non-AnnData input.

    This keeps helper behavior consistent with the main QC validation entry
    point.
    """

    # Confirm non-AnnData input fails clearly.
    with pytest.raises(QCInputValidationError, match="expected an AnnData object"):
        summarize_adata_shape(np.ones((2, 2)))  # type: ignore[arg-type]


def test_validate_duplicate_name_policy_returns_no_warnings_when_unique() -> None:
    """
    Verify that duplicate-name validation is silent when names are unique.

    Unique AnnData indices should not create warnings or errors.
    """

    # Build a duplicate-name policy config.
    policy = QCDuplicateNameConfig()

    # Validate a unique observation index state.
    warnings = validate_duplicate_name_policy(
        names_are_unique=True,
        policy=policy,
        axis_name="obs_names",
    )

    # Confirm no warnings were emitted.
    assert warnings == []


def test_validate_duplicate_name_policy_warns_for_warn_policy() -> None:
    """
    Verify that duplicate-name validation warns under warn policy.

    Warning-only policy should preserve execution while surfacing ambiguity.
    """

    # Build a duplicate-name policy config with obs warning behavior.
    policy = QCDuplicateNameConfig(obs_names="warn")

    # Validate duplicate observation names.
    warnings = validate_duplicate_name_policy(
        names_are_unique=False,
        policy=policy,
        axis_name="obs_names",
    )

    # Confirm a warning was returned.
    assert warnings == ["AnnData.obs_names contains duplicate values."]


def test_validate_duplicate_name_policy_warns_for_make_unique_policy() -> None:
    """
    Verify that duplicate-name validation warns under make_unique policy.

    The validation layer should not mutate names, but it should tell the stage
    that names need to be made unique before metric calculation.
    """

    # Build a duplicate-name policy config with var make_unique behavior.
    policy = QCDuplicateNameConfig(var_names="make_unique")

    # Validate duplicate variable names.
    warnings = validate_duplicate_name_policy(
        names_are_unique=False,
        policy=policy,
        axis_name="var_names",
    )

    # Confirm a make_unique warning was returned.
    assert "make names unique" in warnings[0]


def test_validate_duplicate_name_policy_ignores_for_ignore_policy() -> None:
    """
    Verify that duplicate-name validation is silent under ignore policy.

    Ignore mode is not recommended, but it is explicit and should not warn.
    """

    # Build a duplicate-name policy config with ignore behavior.
    policy = QCDuplicateNameConfig(obs_names="ignore")

    # Validate duplicate observation names.
    warnings = validate_duplicate_name_policy(
        names_are_unique=False,
        policy=policy,
        axis_name="obs_names",
    )

    # Confirm no warnings were emitted.
    assert warnings == []


def test_validate_duplicate_name_policy_errors_for_error_policy() -> None:
    """
    Verify that duplicate-name validation errors under error policy.

    Users should be able to make duplicate AnnData indices fatal.
    """

    # Build a duplicate-name policy config with error behavior.
    policy = QCDuplicateNameConfig(var_names="error")

    # Confirm duplicate variable names fail under error policy.
    with pytest.raises(QCInputValidationError, match="policy is 'error'"):
        validate_duplicate_name_policy(
            names_are_unique=False,
            policy=policy,
            axis_name="var_names",
        )


def test_validate_duplicate_name_policy_rejects_invalid_axis_name() -> None:
    """
    Verify that duplicate-name validation rejects unsupported axis names.

    The helper should only accept obs_names or var_names to avoid silent misuse.
    """

    # Build a duplicate-name policy config.
    policy = QCDuplicateNameConfig()

    # Confirm invalid axis names fail clearly.
    with pytest.raises(QCInputValidationError, match="axis must be"):
        validate_duplicate_name_policy(
            names_are_unique=False,
            policy=policy,
            axis_name="bad_axis",
        )


def test_validate_qc_input_adata_reports_duplicate_name_warnings() -> None:
    """
    Verify that main QC validation reports duplicate-name warnings.

    The validation entry point should surface duplicate AnnData index issues in
    the returned summary when policies are warning-like.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Introduce duplicate observation names.
    adata.obs_names = ["cell_1", "cell_1", "cell_3"]

    # Introduce duplicate variable names.
    adata.var_names = ["gene_1", "gene_1", "gene_3"]

    # Build a duplicate-name policy that warns on both axes.
    config = QCConfig(
        duplicate_names={
            "obs_names": "warn",
            "var_names": "make_unique",
        }
    )

    # Validate the QC input.
    summary = validate_qc_input_adata(adata, config)

    # Confirm observation names were recognized as non-unique.
    assert summary.obs_names_unique is False

    # Confirm variable names were recognized as non-unique.
    assert summary.var_names_unique is False

    # Confirm both warnings were captured.
    assert len(summary.warnings) == 2

    # Confirm the observation warning is present.
    assert any("obs_names" in warning for warning in summary.warnings)

    # Confirm the variable warning is present.
    assert any("var_names" in warning for warning in summary.warnings)


def test_validate_qc_input_adata_errors_on_duplicate_names_when_configured() -> None:
    """
    Verify that main QC validation can make duplicate names fatal.

    Error policy should stop QC before metric calculation when duplicate indices
    are not allowed.
    """

    # Build a valid test AnnData object.
    adata = make_test_adata()

    # Introduce duplicate variable names.
    adata.var_names = ["gene_1", "gene_1", "gene_3"]

    # Build a duplicate-name policy that forbids duplicate variable names.
    config = QCConfig(duplicate_names={"var_names": "error"})

    # Confirm duplicate variable names fail validation.
    with pytest.raises(QCInputValidationError, match="policy is 'error'"):
        validate_qc_input_adata(adata, config)
