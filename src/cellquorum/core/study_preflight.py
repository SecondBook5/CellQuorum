"""Study-specific prerequisites that can be decided before fitting models."""

import glob
import importlib.util
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.config.design import validate_design_against_obs
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.anndata_access import obs_frame
from cellquorum.core.context import PipelineContext
from cellquorum.core.exceptions import CellQuorumConfigError
from cellquorum.core.stage import StageArtifact


def study_blockers(config: CellQuorumConfig, methods: dict[str, list[str]]) -> list[str]:
    errors = []
    r_packages = set()
    imports = set()
    if "pseudobulk_edger" in methods.get("differential_expression", []):
        r_packages.add("edgeR")
    if "gsea_nes_heatmap" in methods.get("enrichment_viz", []):
        r_packages.update(("ComplexHeatmap", "circlize"))
    communication = methods.get("cell_cell_communication", [])
    ccc = config.cell_cell_communication
    if {"chord_viz", "multinichenet_viz"} & set(methods.get("ccc_viz", [])):
        imports.add("pycirclize")
    if "liana" in communication:
        imports.add("liana")
    if "tensor_c2c" in communication:
        imports.add("cell2cell")
        if "liana" not in communication:
            errors.append("Tensor-cell2cell requires per-library LIANA ranks. Select LIANA first.")
    if set(communication) & {"nichenet", "multinichenet"}:
        priors = ["nichenet_lr_network", "nichenet_ligand_target_matrix"]
        if "nichenet" in communication:
            priors.append("nichenet_weighted_networks")
            if not ccc.nichenet_receiver or not ccc.nichenet_sender:
                errors.append("Declare NicheNet sender and receiver populations.")
        r_packages.add("nichenetr")
        if "multinichenet" in communication:
            r_packages.add("multinichenetr")
        for key in priors:
            value = getattr(ccc, key)
            if not value or not Path(value).is_file():
                errors.append(f"Supply a readable cell_cell_communication.{key} ({value}).")
    if "grn" in methods:
        g = config.grn
        if not shutil.which(g.launcher):
            errors.append(f"Install the isolated pySCENIC environment launcher {g.launcher}.")
        else:
            try:
                names = [g.env_name] if isinstance(g.env_name, str) else g.env_name
                for name in names:
                    probe = subprocess.run(
                        [
                            g.launcher,
                            "run",
                            "-n",
                            name,
                            "python",
                            "-c",
                            "import pyscenic, arboreto, ctxcore",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=40,
                        check=False,
                    )
                    if probe.returncode == 0:
                        break
                else:
                    errors.append(
                        f"Prepare pySCENIC environment {names} using envs/pyscenic_env.yml."
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"Could not verify the isolated pySCENIC environment: {exc}")
        for key in ("tfs_path", "motifs_path"):
            value = getattr(g, key)
            if not value or not Path(value).is_file():
                errors.append(f"Supply a readable grn.{key} for {g.organism} ({value}).")
        if not g.rankings_glob or not any(Path(f).is_file() for f in glob.glob(g.rankings_glob)):
            errors.append(f"Supply matching cisTarget ranking databases ({g.rankings_glob}).")
    if "reference_mapping" in methods:
        atlas = config.reference_mapping.atlas_h5ad
        if not atlas or not Path(atlas).is_file():
            errors.append(f"Supply reference_mapping.atlas_h5ad ({atlas}).")
    if "integration_benchmark" in methods and config.integration_benchmark.mode == "full":
        imports.add("scib_metrics")
    if "reference_mapping" in methods:
        imports.add("scvi")
    for stage, pkg in [("multicellular_programs", "DIALOGUE"), ("ambient_correction", "SoupX")]:
        if stage in methods:
            r_packages.add(pkg)
    if "cytotrace" in methods.get("trajectory", []):
        from cellquorum.stages.trajectory._cytotrace import worker_issue

        issue = worker_issue()
        if issue:
            errors.append(issue)
    for method, module in [
        ("velocity", "scvelo"),
        ("cellrank", "cellrank"),
        ("palantir", "palantir"),
    ]:
        if method in methods.get("trajectory", []):
            imports.add(module)
    if (
        "velocity" in methods.get("trajectory", [])
        and config.trajectory.velocity.mode == "stochastic"
    ):
        try:
            scvelo_version = metadata.version("scvelo")
            numpy_version = np.lib.NumpyVersion(metadata.version("numpy"))
        except metadata.PackageNotFoundError:
            pass  # Missing packages are reported by the module checks below.
        else:
            if scvelo_version == "0.3.4" and numpy_version >= "2.4.0":
                errors.append(
                    "Stochastic scVelo 0.3.4 requires NumPy below 2.4 for its least-squares "
                    "fit. Install cellquorum[lineage] in the analysis environment."
                )
    for entry in config.enrichment.methods if "enrichment" in methods else []:
        gmt = entry.get("gmt_path") or config.enrichment.gmt_path
        if gmt and not Path(gmt).is_file():
            errors.append(f"Supply the gene set file {gmt} for {entry.get('method')}.")
    if (
        "velocity" in methods.get("trajectory", [])
        and config.trajectory.velocity.generation.generate_missing
    ):
        for tool in ("samtools", "velocyto"):
            if not shutil.which(tool):
                errors.append(
                    f"Install {tool} to generate missing splice counts from the BAM files."
                )
    if "aucell" in methods.get("state_scoring", []) or "activity" in methods.get("enrichment", []):
        imports.add("decoupler")
    for name in sorted(imports):
        if importlib.util.find_spec(name) is None:
            errors.append(f"Install {name} in the CellQuorum Python environment.")
    if r_packages:
        executable = shutil.which(config.r.rscript_path)
        if not config.r.enabled or not executable:
            errors.append(f"Enable R and supply Rscript for {', '.join(sorted(r_packages))}.")
        else:
            expr = (
                'p=commandArgs(TRUE); cat(p[!sapply(p, requireNamespace, quietly=TRUE)], sep="\\n")'
            )
            try:
                proc = subprocess.run(
                    [executable, "--vanilla", "-e", expr, *sorted(r_packages)],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    check=False,
                )
                if proc.returncode:
                    errors.append("R package check failed: " + proc.stderr.strip()[-300:])
                else:
                    for package in proc.stdout.strip().splitlines():
                        errors.append(f"Install the required R package {package}.")
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"R package check failed: {exc}")
    path = config.input.h5ad
    if path and not path.is_file():
        errors.append(f"Input file does not exist: {path}")
    elif path and config.design.case and config.design.control:
        from cellquorum.cli.study import _metadata

        obs, _, _ = _metadata(path)
        design = config.design
        try:
            validate_design_against_obs(obs, design=design)
            if "multinichenet" in communication and design.paired and ccc.cell_type_col in obs:
                sample = ccc.sample_col or config.cohort.sample_key
                groups = ccc.mnn_cell_types or sorted(obs[ccc.cell_type_col].dropna().unique())
                meta = obs[[sample, design.donor_col, design.condition_col]].drop_duplicates()
                if meta[sample].duplicated().any():
                    errors.append("Each MultiNicheNet library must have one donor and condition.")
                else:
                    counts = pd.crosstab(obs[sample], obs[ccc.cell_type_col]).reindex(
                        columns=groups, fill_value=0
                    )
                    eligible = meta[
                        meta[sample].isin(
                            counts.index[counts.ge(ccc.nichenet_min_cells).all(axis=1)]
                        )
                    ]
                    pairs = pd.crosstab(
                        eligible[design.donor_col], eligible[design.condition_col]
                    ).reindex(columns=[design.control, design.case], fill_value=0)
                    n_pairs = int(pairs.eq(1).all(axis=1).sum())
                    if n_pairs < 3:
                        errors.append(
                            f"MultiNicheNet has {n_pairs} complete eligible pairs "
                            "before QC. Supply at least three or revise mnn_cell_types."
                        )
            batch = config.cohort.batch_key
            if batch:
                arms = [design.control, design.case]
                table = pd.crosstab(obs[batch], obs[design.condition_col]).reindex(
                    columns=arms, fill_value=0
                )
                if not table.gt(0).all(axis=1).any():
                    errors.append(
                        f"Condition {design.condition_col} and batch {batch} are fully confounded."
                    )
        except (KeyError, ValueError, CellQuorumConfigError) as exc:
            errors.append(f"Correct the study design: {exc}")
    return errors


def record_qc_support(
    context: PipelineContext, stage: str, adata: ad.AnnData
) -> StageArtifact | None:
    """Record eligible biological samples after QC for the declared comparison."""
    from cellquorum.stages.qc.eligibility import de_eligible_cells

    config = context.config
    if (
        not getattr(getattr(config, "run", None), "study_questions", ())
        or not config.design.case
        or not config.design.control
    ):
        return None
    obs = obs_frame(adata)
    keys = [config.design.donor_col, config.design.condition_col]
    if any(key not in obs for key in keys):
        return None
    table = obs[keys].copy()
    mask = de_eligible_cells(obs)
    table["eligible_cells"] = 1 if mask is None else mask.astype(int)
    counts = (
        table.groupby(keys, observed=True)
        .agg(cells=("eligible_cells", "size"), eligible_cells=("eligible_cells", "sum"))
        .reset_index()
    )
    if config.input.h5ad and config.input.h5ad.is_file():
        from cellquorum.cli.study import _metadata

        before, _, _ = _metadata(config.input.h5ad)
        if all(key in before for key in keys):
            inputs = before.groupby(keys, observed=True).size().rename("input_cells").reset_index()
            counts = inputs.merge(counts, on=keys, how="left", validate="one_to_one")
            for column in ("cells", "eligible_cells"):
                counts[column] = counts[column].fillna(0).astype(int)
            counts["excluded_cells"] = counts.input_cells - counts.eligible_cells
    path = context.paths.provenance / f"{stage}_donor_support.csv"
    counts.to_csv(path, index=False)
    return StageArtifact(
        name=f"{stage}_donor_support",
        path=path,
        kind="csv",
        description="Cells retained for the comparison by donor and condition.",
    )
