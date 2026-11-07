#!/usr/bin/env python3
# ruff: noqa: E501  (the cell text is prose and code samples, wrapped for the notebook, not for the editor)
"""Write docs/tutorials/api_walkthrough.ipynb, the tutorial notebook on the GSE140819 data.

The notebook is generated, then executed (``jupyter nbconvert --execute``) so the committed copy carries real
outputs. Edit the cells here, not in the .ipynb. Usage:

    python scripts/build_api_walkthrough_notebook.py
    jupyter nbconvert --to notebook --execute --inplace docs/tutorials/api_walkthrough.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbformat.NotebookNode:
    return new_code_cell(text.strip("\n"))


cells = [
    md("""
# CellQuorum Python API walkthrough

The command line (`cellquorum run --config ...`) is the usual way to run an analysis. This notebook shows
the same engine from Python: load a configuration as an object, run one stage at a time and look at what
each stage reports, then run the whole pipeline and read its outputs. Nothing here is a second
implementation. The notebook calls (`cq.pp`, `cq.tl`) run the same registered stages the command line
runs, so what you see here is what a config-driven run produces.

The data are the four-donor subset of GSE140819 (Slyper et al. 2020, *Nat Med*) described in the
GSE140819 data: each donor has a whole-cell and a nuclei
library, so the design is paired and the donor is the unit of replication.
"""),
    code("""
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import cellquorum as cq
from cellquorum.config.loader import load_config
from cellquorum.version import __version__

warnings.filterwarnings("ignore", category=FutureWarning)
# The repository root, whether this runs from docs/tutorials or from a Jupyter started at the root.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs").is_dir())
print("CellQuorum", __version__)
"""),
    md("""
## 1. The data

`scripts/prepare_gse140819_tutorial.py` downloads the GEO archive (1.2 GB, once) and writes the prepared
object. The next cell runs it if the file is not there yet.
"""),
    code("""
from cellquorum.datasets import _gse140819 as gse

PREPARED = gse.dataset_dir() / "tutorial.h5ad"
if not PREPARED.exists():
    import runpy

    runpy.run_path(str(ROOT / "scripts" / "prepare_gse140819_tutorial.py"), run_name="__main__")

adata = ad.read_h5ad(PREPARED)
print(adata)
adata.obs.groupby(["donor", "condition"], observed=True).size().unstack(fill_value=0)
"""),
    md("""
Raw counts are in `adata.layers["counts"]`. The columns the configuration names are `sample` (the library),
`donor`, `condition` (`cells` or `nuclei`) and `cell_type` (the authors' labels, used here as given).

To keep this notebook quick we work with a random 4,000-cell subset. The full pipeline run in section 5 uses
the same subset; the tutorial page shows the 15,000-cell run.
"""),
    code("""
rng = np.random.default_rng(0)
subset = adata[np.sort(rng.choice(adata.n_obs, 4000, replace=False))].copy()
subset.obs.groupby(["donor", "condition"], observed=True).size().unstack(fill_value=0)
"""),
    md("""
## 2. The configuration is an object

`load_config` reads and validates the YAML. Validation is strict: an unknown setting, a misspelt key or an
impossible combination is an error at load time, before any stage runs.
"""),
    code("""
cfg = load_config(ROOT / "configs" / "gse140819_tutorial.yaml")
print("design:   ", cfg.design.model_dump())
print("cohort:   ", {k: v for k, v in cfg.cohort.model_dump().items() if v})
print("nuclei:   ", cfg.qc_evidence.graded.nuclei_key, cfg.qc_evidence.graded.nuclei_values)
"""),
    md("""
Two examples of what strict validation catches. A `methods:` entry must name its method with `method:`;
without it the stage would quietly run its default instead.
"""),
    code("""
from pydantic import ValidationError

try:
    load_config(
        {**cfg.model_dump(), "differential_abundance": {"methods": [{"name": "proportion_ttest"}]}}
    )
except (ValidationError, Exception) as error:
    print(type(error).__name__)
    print(str(error)[:400])
"""),
    md("""
And the nuclear-integrity setting. Naming the nuclei **and** switching the axis off for the whole run would
be the same decision written twice, so it is refused.
"""),
    code("""
from cellquorum.stages.qc.config import QCGradedConfig

try:
    QCGradedConfig(nuclear_axis_applicable=False, nuclei_key="condition", nuclei_values=["nuclei"])
except ValidationError as error:
    print(str(error).splitlines()[-1][:300])
"""),
    md("""
## 3. One stage at a time

Each notebook call takes the AnnData and the configuration, runs one stage and returns a
`NotebookStageOutput`: the updated AnnData plus the stage's metrics, notes and warnings. We start with QC.

The QC stage scores every cell on several independent kinds of evidence (capture complexity, metabolic
stress, nuclear integrity) and calls it `core`, `borderline` or `quarantine`; it removes nothing. The
nuclear-integrity axis (MALAT1) means damage in a whole cell but is the expected signal in a nucleus, so the
configuration names the nuclei and the axis is switched off for them only.
"""),
    code("""
out_dir = ROOT / "runs" / "api_walkthrough"  # outside docs/, which the site build scans
qc = cq.pp.qc(subset, config=cfg, output_dir=out_dir / "qc")
data = qc.adata
print("warnings:", len(qc.warnings), "| notes:", len(qc.notes))
pd.crosstab(data.obs["condition"], data.obs["qc_state_initial"])
"""),
    code("""
# The nuclear axis: scored for whole cells, not applicable for nuclei.
pd.crosstab(data.obs["condition"], data.obs["qc_ev_malat1_fraction_availability"])
"""),
    md("""
Because a nucleus has one evidence family fewer, its **evidence coverage** is lower, and the QC design says
so on purpose: a verdict reached on fewer families is reported with less certainty.
"""),
    code("""
data.obs.groupby("condition", observed=True)["qc_evidence_coverage"].agg(["mean", "min", "max"])
"""),
    md("""
Next normalisation, feature selection, PCA, Harmony integration over the donor, and clustering. Each call
reads its settings from the configuration, so nothing is repeated here. The configuration names the
normalisation recipe (`cellquorum_log1p_cp10k_v1`, log1p of counts per 10,000); the default recipe needs a
separate environment, which a fresh install does not have.
"""),
    code("""
data = cq.pp.normalize(data, config=cfg, output_dir=out_dir / "norm").adata
data = cq.pp.select_features(data, config=cfg, output_dir=out_dir / "hvg").adata
data = cq.tl.reduce_dimensions(data, config=cfg, output_dir=out_dir / "pca").adata
integrated = cq.tl.integrate(data, config=cfg, output_dir=out_dir / "integration")
data = integrated.adata
print("integration batch key:", integrated.metrics.get("batch_key"))
print("obsm:", list(data.obsm))
"""),
    code("""
clustered = cq.tl.cluster(data, config=cfg, output_dir=out_dir / "clustering")
data = clustered.adata
data.obs["leiden"].value_counts().sort_index().head(10)
"""),
    md("""
Cluster identity against the authors' labels, as a sanity check that the integrated space separates cell
types and not preparations:
"""),
    code("""
pd.crosstab(data.obs["cell_type"], data.obs["leiden"]).iloc[:, :9]
"""),
    md("""
## 4. Reading the stage output

`NotebookStageOutput` also carries the underlying `StageResult`, with the artifacts the stage wrote.
"""),
    code("""
print(sorted(qc.metrics)[:8])
print([a.name if hasattr(a, "name") else str(a) for a in qc.result.artifacts][:6])
"""),
    md("""
## 5. The whole pipeline in one call

`cq.run_pipeline` takes a path, a validated configuration object, or a plain dictionary. Here it gets the
tutorial configuration with the input pointed at our subset. Every stage is named in that configuration, so
exactly the stages we want run and nothing else.
"""),
    code("""
subset_path = out_dir / "subset.h5ad"
out_dir.mkdir(exist_ok=True)
subset.write_h5ad(subset_path)

config = cfg.model_dump(mode="json")
config["input"]["h5ad"] = str(subset_path)
config["compute"]["backend"] = "cpu"

result = cq.run_pipeline(config=config, output_dir=out_dir / "run", quiet=True)
records = result.execution_result.stage_execution_records
ran = [r.stage_name for r in records if r.status == "success"]
print("ran:", ", ".join(ran))
"""),
    code("""
run = out_dir / "run"
pd.read_csv(run / "results" / "composition_by_condition.csv").round(2).head(12)
"""),
    code("""
da = pd.read_csv(run / "results" / "da_proportion_ttest.csv")
columns = ["cell_type", "control_mean_pct", "case_mean_pct", "effect_pp", "pvalue", "fdr"]
da[[*columns, "design_floor_p"]].round(3)
"""),
    md("""
With four donors the smallest p-value a paired exact test can reach is 0.125, so no cell type can reach
FDR significance here, and the stage reports that floor (`design_floor_p`) next to every result instead of
leaving the reader to wonder. The differences are descriptive; the tutorial page discusses what to do about
it (more donors).
"""),
    code("""
from IPython.display import Image

Image(filename=str(run / "figures" / "composition" / "composition_by_condition.png"), width=360)
"""),
    md("""
## 6. Reusing a primitive

The statistical building blocks are importable on their own. `aggregate_pseudobulk` turns cells into one
row per donor and condition, which is what the donor-level tests use as the unit.
"""),
    code("""
pb = cq.utils.aggregate_pseudobulk(
    data, layer="counts", donor_col="donor", condition_col="condition"
)
print(type(pb).__name__)
print({k: getattr(v, "shape", None) for k, v in vars(pb).items() if hasattr(v, "shape")})
"""),
    md("""
## Where next

- The same run from the command line: `cellquorum run --config configs/gse140819_tutorial.yaml`.
- [Configuration guide](../guide/configuration.md) and [Python API reference](../reference/api.md).
"""),
]


def main() -> None:
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = ROOT / "docs" / "tutorials" / "api_walkthrough.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
