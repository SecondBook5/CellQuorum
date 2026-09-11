"""SampleQC boundary validation and preservation of assessment-only semantics."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.qc.config import QCConfig, QCSampleQCConfig
from cellquorum.stages.qc.sampleqc import fit_sampleqc


def metrics():
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "total_counts": rng.uniform(100, 10000, 60),
            "n_genes_by_counts": rng.uniform(10, 90, 60),
            "pct_counts_mito": rng.uniform(1, 20, 60),
            "sample_id": ["NA"] * 30 + ["001"] * 30,
        },
        index=[f"barcode_{i}" for i in range(60)],
    )


class Backend:
    def __init__(self, corrupt=None):
        self.corrupt = corrupt
        self.calls = 0

    def run_script(self, script, args, **kwargs):
        self.calls += 1
        source = pd.read_csv(args[0])
        self.source = source
        result = pd.DataFrame(
            {
                "cell_id": source.cell_id,
                "distance": 0.1,
                "component": 1,
                "outlier": 0,
            }
        )
        result.loc[0, ["distance", "outlier"]] = [100, 1]
        if self.corrupt:
            result = self.corrupt(result)
        result.iloc[::-1].to_csv(args[1], index=False)
        Path(args[1] + ".version").write_text("0.6.6")
        return SimpleNamespace(returncode=0, stderr="")


def test_explicit_components_and_strict_config():
    assert not QCConfig().sampleqc.enabled
    with pytest.raises(ValidationError, match="n_components"):
        QCSampleQCConfig(enabled=True)
    for kwargs in (
        {"n_components": True},
        {"alpha": np.nan},
        {"alpha": 1},
        {"sample_key": " "},
        {"min_cells_per_sample": 0},
    ):
        with pytest.raises(ValidationError):
            QCSampleQCConfig(**kwargs)


def test_alignment_and_unscored_cells_are_explicit():
    data = metrics()
    data.loc["barcode_4", "total_counts"] = 0
    original = data.copy(deep=True)
    backend = Backend()
    result = fit_sampleqc(data, QCSampleQCConfig(n_components=2), backend)
    pd.testing.assert_frame_equal(data, original)
    assert result.cells.index.equals(data.index)
    assert result.cells.loc["barcode_0", "sampleqc_outlier"]
    assert result.cells.loc["barcode_4", "sampleqc_status"] == "unusable_metrics"
    assert np.isnan(result.cells.loc["barcode_4", "sampleqc_pvalue"])
    assert result.provenance["n_scored"] == 59
    assert result.provenance["assessment_only"]
    assert np.isfinite(backend.source.iloc[:, 2:]).all().all()


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda x: x.iloc[1:],
        lambda x: pd.concat([x, x.iloc[:1]]),
        lambda x: x.assign(distance=np.nan),
        lambda x: x.assign(outlier=0),
        lambda x: x.assign(component=3),
    ],
)
def test_rejects_invalid_backend_results(corrupt):
    with pytest.raises(CellQuorumDataError, match="Invalid SampleQC output"):
        fit_sampleqc(metrics(), QCSampleQCConfig(n_components=2), Backend(corrupt))


def test_missing_samples_and_degenerate_features_fail_before_execution():
    for data in (
        metrics().assign(sample_id=None),
        metrics().iloc[:5],
        metrics().assign(pct_counts_mito=50, total_counts=100),
        metrics().assign(pct_counts_mito=0),
    ):
        backend = Backend()
        with pytest.raises(CellQuorumDataError):
            fit_sampleqc(data, QCSampleQCConfig(n_components=2), backend)
        assert backend.calls == 0


def test_backend_failure_is_not_silently_skipped():
    backend = SimpleNamespace(
        run_script=lambda *a, **kw: SimpleNamespace(returncode=1, stderr="SampleQC unavailable")
    )
    with pytest.raises(CellQuorumDataError, match="SampleQC unavailable"):
        fit_sampleqc(metrics(), QCSampleQCConfig(n_components=2), backend)


def test_stage_persists_assessments_without_removing_outliers(tmp_path, monkeypatch):
    import anndata as ad

    from cellquorum.core.context import PipelineContext, PipelinePaths
    from cellquorum.stages.qc.sampleqc import SampleQCResult
    from cellquorum.stages.qc.stage import QCStage

    data = ad.AnnData(
        X=np.array([[5, 10, 2], [4, 20, 3], [3, 25, 1], [6, 10, 4]]),
        obs=pd.DataFrame({"sample_id": ["a", "a", "b", "b"]}, index=list("abcd")),
        var=pd.DataFrame(index=["MT-ND1", "ACTB", "GAPDH"]),
    )
    config = QCConfig(
        sampleqc={"enabled": True, "n_components": 2},
        mito_mixture={"enabled": False},
        graded={"enabled": False},
        metrics={"percent_top": [2]},
        floors={
            "min_genes_per_cell": None,
            "min_counts_per_cell": None,
            "min_cells_per_gene": None,
        },
        outputs={"write_figures": False, "write_h5ad": True},
    )

    def fit(table, settings, backend):
        assert "sample_id" in table
        return SampleQCResult(
            pd.DataFrame(
                {"sampleqc_outlier": True, "sampleqc_status": "scored"}, index=table.index
            ),
            {"status": "fitted", "assessment_only": True, "version": "test"},
        )

    monkeypatch.setattr("cellquorum.stages.qc.sampleqc.fit_sampleqc", fit)
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    context = PipelineContext(config=SimpleNamespace(), paths=paths, adata=data)
    result = QCStage(config=config).run(context)
    assert result.adata.n_obs == 4
    assert result.adata.obs.sampleqc_outlier.all()
    assert result.metrics["sampleqc"]["assessment_only"]
    assert result.adata.uns["cellquorum"]["sampleqc"]["status"] == "fitted"
    tables = list(tmp_path.rglob("cell_metrics.csv"))
    assert tables and pd.read_csv(tables[0]).sampleqc_outlier.all()
    context.adata = result.adata
    config.sampleqc.enabled = False
    rerun = QCStage(config=config).run(context)
    assert "sampleqc_outlier" not in rerun.adata.obs
    assert "sampleqc" not in rerun.adata.uns["cellquorum"]


@pytest.mark.r
@pytest.mark.parametrize("n_samples", [1, 2, 3])
def test_upstream_sampleqc_smoke(n_samples):
    import shutil
    import subprocess

    from cellquorum.backends.rscript import RscriptBackend

    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is unavailable")
    available = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            'quit(status=if(requireNamespace("SampleQC", quietly=TRUE)) 0 else 1)',
        ],
        capture_output=True,
        timeout=60,
    )
    if available.returncode:
        pytest.skip("Upstream SampleQC is not installed")
    rng = np.random.default_rng(11)
    n = 200 * n_samples
    rare = np.tile(np.arange(200) >= 180, n_samples)
    log_counts = rng.normal(8 - rare * 2, 0.25, n)
    counts = np.exp(log_counts)
    data = pd.DataFrame(
        {
            "sample_id": np.repeat([f"library_{i}" for i in range(n_samples)], 200),
            "total_counts": counts,
            "n_genes_by_counts": np.exp(log_counts * 0.6 + rng.normal(0, 0.1, n)),
            "pct_counts_mito": 100 / (1 + np.exp(-rng.normal(-3 + rare, 0.25, n))),
        }
    )
    result = fit_sampleqc(data, QCSampleQCConfig(n_components=2), RscriptBackend())
    assert len(result.cells) == n
    assert result.cells.sampleqc_status.eq("scored").all()
    assert result.cells.sampleqc_pvalue.between(0, 1).all()
    assert result.provenance["assessment_only"]
    assert result.provenance["version"]
