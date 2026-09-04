# tests/test_de_end_to_end.py
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.backends.registry import build_default_backend_registry
from cellquorum.config.design import DesignConfig, validate_design_against_obs
from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.core.exceptions import CellQuorumConfigError
from cellquorum.stages.comparative.differential_expression.stage import DifferentialExpressionStage


def _edger_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status=ifelse(requireNamespace('edgeR', quietly=TRUE),0,1))",
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _adata():
    rng = np.random.default_rng(2)
    donors = ["d1", "d2", "d3"]
    blocks, obs_rows = [], []
    for donor in donors:
        for cond in ["Normal", "LE"]:
            for _ in range(8):
                base = rng.poisson(5, size=15).astype(float)
                if cond == "LE":
                    base[2] += 40
                blocks.append(base)
                obs_rows.append({"patient_id": donor, "condition": cond})
    a = ad.AnnData(X=sp.csr_matrix(np.vstack(blocks)), obs=pd.DataFrame(obs_rows))
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"G{i}" for i in range(15)]
    set_layer_tag(a, "counts", kind="counts")
    return a


class _Cfg:
    # Minimal config object exposing the stage sub-block the dispatcher reads.
    differential_expression = {
        "enabled": True,
        "method": "pseudobulk_edger",
        "layer": "counts",
        "condition_col": "condition",
        "donor_col": "patient_id",
        "case": "LE",
        "control": "Normal",
        "covariates": [],
        "paired": True,
    }
    cohort = None


def test_design_guardrail_accepts_paired_cohort():
    # The existing design guard should accept a complete 3-donor paired design.
    a = _adata()
    res = validate_design_against_obs(
        a.obs,
        design=DesignConfig(
            donor_col="patient_id",
            condition_col="condition",
            case="LE",
            control="Normal",
            paired=True,
        ),
    )
    assert len(res.complete_pair_donors) == 3


def _adata_with_confounded_covariate():
    """A paired design whose 'batch' covariate is perfectly aliased with condition.

    Every Normal sample is batch 'b0' and every LE sample is batch 'b1', so the
    fixed-effects model ~ batch + condition is rank-deficient: the condition
    effect cannot be separated from batch.
    """
    a = _adata()
    a.obs["batch"] = np.where(a.obs["condition"].to_numpy() == "LE", "b1", "b0")
    return a


class _CfgConfoundedCovariate:
    # Same design as _Cfg but adds a covariate confounded with condition.
    differential_expression = {
        "enabled": True,
        "method": "pseudobulk_edger",
        "layer": "counts",
        "condition_col": "condition",
        "donor_col": "patient_id",
        "case": "LE",
        "control": "Normal",
        "covariates": ["batch"],
        "paired": True,
    }
    cohort = None


def test_stage_halts_on_covariate_confounded_with_condition(tmp_path):
    # A covariate aliased with the tested condition makes the multi-factor design
    # non-estimable. The stage must halt loudly BEFORE reaching edgeR (so this
    # holds with or without R installed), never silently hand a rank-deficient
    # design to the fit.
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    ctx = PipelineContext(
        config=_CfgConfoundedCovariate(),
        paths=paths,
        adata=_adata_with_confounded_covariate(),
        backend_registry=build_default_backend_registry(),
    )
    with pytest.raises(CellQuorumConfigError, match="estimable|Confounded|rank"):
        DifferentialExpressionStage().run(ctx)


def test_stage_runs_through_context(tmp_path):
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    # CONTROLLER NOTE 2 FIX: construct the context WITH a real backend registry,
    # exactly as the production pipeline does (core/pipeline.py uses
    # build_default_backend_registry()).
    ctx = PipelineContext(
        config=_Cfg(),
        paths=paths,
        adata=_adata(),
        backend_registry=build_default_backend_registry(),
    )
    result = DifferentialExpressionStage().run(ctx)
    # Either a real DE table (edgeR present) or a recorded skip — never a crash.
    if _edger_available():
        de_paths = [a.path for a in result.artifacts if a.name == "de_results"]
        assert de_paths and Path(de_paths[0]).is_file()
        de = pd.read_csv(de_paths[0])
        assert de.loc[de["gene"] == "G2"].iloc[0]["FDR"] < 0.05
    else:
        assert result.metrics.get("skipped") is True
