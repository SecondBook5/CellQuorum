"""Unit tests for AnalysisMethod helper methods."""

from cellquorum.methods.base import AnalysisMethod, MethodSkip


class _Dummy(AnalysisMethod):
    """Synthetic test method for testing helpers."""

    name = "dummy"
    stage_category = "test"
    backend = "python"

    def input_contract(self, config):
        """Test stub."""
        pass

    def _run(self, adata, config, context):
        """Test stub."""
        pass


def test_skip_helper_shapes_reason_and_details():
    """Test that _skip() shapes reason and details correctly."""
    skip = _Dummy()._skip("Rscript unavailable", r_package="edgeR")
    assert isinstance(skip, MethodSkip)
    assert skip.reason == "dummy skipped: Rscript unavailable"
    assert skip.details == {"method": "dummy", "r_package": "edgeR"}
