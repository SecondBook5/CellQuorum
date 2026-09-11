"""Assess joint QC profiles with the upstream SampleQC R package."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from cellquorum.backends.rscript import RscriptBackend
from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.qc.config import QCSampleQCConfig


@dataclass(frozen=True)
class SampleQCResult:
    """Store aligned assessments and the provenance of the joint model."""

    cells: pd.DataFrame
    provenance: dict


_R_SCRIPT = r"""
args <- commandArgs(trailingOnly=TRUE)
if (!requireNamespace("SampleQC", quietly=TRUE)) {
    stop("Install the upstream R package SampleQC from github.com/wmacnair/SampleQC")
}
suppressPackageStartupMessages(library(SampleQC))
set.seed(as.integer(args[5]))
x <- read.csv(args[1], stringsAsFactors=FALSE, check.names=FALSE)
qc <- make_qc_dt(x)
samples <- sort(unique(qc$sample_id))
parts <- lapply(samples, function(s) as.data.frame(qc[qc$sample_id == s, ]))
qc_names <- c("log_counts", "log_feats", "logit_mito")
disc <- c("N_cat", "mito_cat")
cont <- c("log_N", "med_mito", "med_counts")
cd <- S4Vectors::DataFrame(
    sample_id=samples, group_id=rep("SG1", length(samples)),
    cell_id=I(lapply(parts, function(d) d$cell_id)),
    qc_metrics=I(lapply(parts, function(d) d[, qc_names, drop=FALSE])),
    annot_disc=I(lapply(parts, function(d) unique(d[, disc, drop=FALSE]))),
    annot_cont=I(lapply(parts, function(d) unique(d[, cont, drop=FALSE])))
)
placeholder <- Matrix::Matrix(0, length(samples), length(samples), sparse=TRUE)
obj <- SingleCellExperiment::SingleCellExperiment(
    assays=list(mmd=placeholder, mmd_adj=placeholder), colData=cd,
    metadata=list(qc_names=qc_names, D=length(qc_names), n_groups=1L,
                  group_list="SG1", mmd_params=list(computed=FALSE),
                  annots=list(disc=disc, cont=cont))
)
obj <- fit_sampleqc(obj, K_all=as.integer(args[3]), alpha=as.numeric(args[4]),
                    n_cores=1, method="robust", bp_seed=as.integer(args[5]))
rows <- data.table::rbindlist(SummarizedExperiment::colData(obj)$outlier)
distances <- as.matrix(rows[, grep("^maha_", names(rows)), with=FALSE])
result <- data.frame(cell_id=rows$cell_id,
                     distance=apply(distances, 1, min),
                     component=max.col(-distances, ties.method="first"),
                     outlier=as.integer(rows$outlier))
write.csv(result, args[2], row.names=FALSE)
writeLines(as.character(utils::packageVersion("SampleQC")), paste0(args[2], ".version"))
"""


def fit_sampleqc(
    metrics: pd.DataFrame,
    config: QCSampleQCConfig,
    backend: RscriptBackend,
) -> SampleQCResult:
    """Fit one joint sample group; return outlier assessments without removing cells."""
    required = ["total_counts", "n_genes_by_counts", "pct_counts_mito", config.sample_key]
    missing = set(required) - set(metrics.columns)
    if missing:
        raise CellQuorumDataError(f"SampleQC requires metric columns: {sorted(missing)}")
    if not metrics.index.is_unique:
        raise CellQuorumDataError("SampleQC requires unique cell identifiers.")
    if config.n_components is None:
        raise CellQuorumDataError("SampleQC requires an explicit n_components.")
    samples = metrics[config.sample_key]
    if samples.isna().any() or samples.astype(str).str.strip().eq("").any():
        raise CellQuorumDataError("SampleQC sample identifiers cannot be missing or blank.")
    values = metrics[required[:3]].apply(pd.to_numeric, errors="coerce")
    counts, genes, mito = (values[col] for col in required[:3])
    usable = np.isfinite(values).all(axis=1) & (counts >= 1) & (genes >= 1) & mito.between(0, 100)
    cells = pd.DataFrame(index=metrics.index)
    cells["sampleqc_status"] = "unusable_metrics"
    cells["sampleqc_distance"] = np.nan
    cells["sampleqc_pvalue"] = np.nan
    cells["sampleqc_component"] = -1
    cells["sampleqc_outlier"] = False
    sample_codes = pd.Series(pd.factorize(samples, sort=False)[0], index=metrics.index)
    sizes = usable.groupby(sample_codes).transform("sum")
    cells.loc[usable, "sampleqc_status"] = "insufficient_sample_cells"
    usable &= sizes >= config.min_cells_per_sample
    provenance = {
        "method": "SampleQC",
        "status": "not_fitted",
        "assessment_only": True,
        "sample_key": config.sample_key,
        "n_components": config.n_components,
        "alpha": config.alpha,
        "random_state": config.random_state,
        "min_cells_per_sample": config.min_cells_per_sample,
        "sample_grouping": "one_joint_group",
        "n_scored": 0,
        "features": ["log10_counts", "log10_genes", "logit_mito_pseudocount_1"],
    }
    if not usable.any():
        raise CellQuorumDataError("SampleQC has no samples with sufficient usable cells.")
    if usable.sum() < config.n_components * 4:
        raise CellQuorumDataError("SampleQC has too few usable cells for n_components.")
    if mito[usable].nunique() < 2:
        raise CellQuorumDataError("SampleQC requires informative mitochondrial measurements.")
    positions = np.flatnonzero(usable)
    proportion = (counts[usable] * mito[usable] / 100 + 1) / (counts[usable] + 2)
    table = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in positions],
            "sample_id": [f"sample_{i}" for i in sample_codes[usable]],
            "log_counts": np.log10(counts[usable].to_numpy()),
            "log_feats": np.log10(genes[usable].to_numpy()),
            "logit_mito": np.log(proportion.to_numpy() / (1 - proportion.to_numpy())),
        }
    )
    if (
        np.linalg.matrix_rank(table.iloc[:, 2:].to_numpy() - table.iloc[:, 2:].mean().to_numpy())
        < 3
    ):
        raise CellQuorumDataError("SampleQC requires three non-degenerate QC features.")
    with tempfile.TemporaryDirectory(prefix="cellquorum-sampleqc-") as directory:
        root = Path(directory)
        source, target, script = root / "metrics.csv", root / "result.csv", root / "fit.R"
        table.to_csv(source, index=False)
        script.write_text(_R_SCRIPT)
        try:
            run = backend.run_script(
                script,
                [
                    str(source),
                    str(target),
                    str(config.n_components),
                    str(config.alpha),
                    str(config.random_state),
                ],
                timeout=config.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CellQuorumDataError(f"SampleQC could not execute: {exc}") from exc
        if run.returncode:
            raise CellQuorumDataError(f"SampleQC failed: {run.stderr[-4000:]}")
        try:
            result = pd.read_csv(target, dtype={"cell_id": str}).set_index("cell_id")
            version = Path(str(target) + ".version").read_text().strip()
            if not result.index.is_unique or set(result.index) != set(table.cell_id):
                raise ValueError("returned cell identifiers do not match the input")
            result = result.loc[table.cell_id]
            distance = result.distance.to_numpy(dtype=float)
            component = result.component.to_numpy(dtype=float)
            outlier = result.outlier.to_numpy(dtype=float)
            if (
                not np.isfinite(distance).all()
                or (distance < 0).any()
                or not np.isfinite(component).all()
                or (component != np.floor(component)).any()
                or (component < 1).any()
                or (component > config.n_components).any()
                or not np.isin(outlier, [0, 1]).all()
                or not version
            ):
                raise ValueError("invalid model output")
            if not np.array_equal(outlier.astype(bool), distance > chi2.isf(config.alpha, 3)):
                raise ValueError("outlier calls disagree with model distances")
        except (OSError, ValueError, KeyError, AttributeError) as exc:
            raise CellQuorumDataError(f"Invalid SampleQC output: {exc}") from exc
    cells.loc[usable, "sampleqc_distance"] = distance
    cells.loc[usable, "sampleqc_pvalue"] = chi2.sf(distance, 3)
    cells.loc[usable, "sampleqc_component"] = component.astype(int)
    cells.loc[usable, "sampleqc_outlier"] = outlier.astype(bool)
    cells.loc[usable, "sampleqc_status"] = "scored"
    provenance.update(
        status="fitted",
        version=version,
        n_scored=int(usable.sum()),
        n_unscored=int((~usable).sum()),
        n_outliers=int(outlier.sum()),
        backend_log=run.stderr[-8000:],
    )
    return SampleQCResult(cells=cells, provenance=provenance)
