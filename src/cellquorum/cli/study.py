"""Set up a first CPU analysis from an AnnData file and the researcher's study design."""

from __future__ import annotations

import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import typer
import yaml
from anndata.io import read_elem

from cellquorum.config.design import DesignConfig, validate_design_against_obs
from cellquorum.config.models import CellQuorumConfig, StageSelectionConfig
from cellquorum.core.labels import as_label_strings
from cellquorum.stages.qc.config import QCFloorConfig


def _choose(prompt: str, options: list[str], *suggestions: str) -> str:
    """Only offer a default when it exists among the available choices."""
    if not options:
        raise ValueError(f"No choices available for '{prompt}'. Check your input metadata.")
    default = next((name for name in suggestions if name in options), None)
    if len(options) == 1:
        default = options[0]
    choices = ", ".join(name for name in options[:25] if name)
    if len(options) > 25:
        choices += f", …; {len(options)} choices, enter the full value"
    while True:
        answer = typer.prompt(f"{prompt} ({choices})", default=default, show_default=bool(default))
        if answer in options:
            return answer
        typer.echo("Choose one of the displayed values.")


def _metadata(path: Path) -> tuple[pd.DataFrame, int, list[str]]:
    with h5py.File(path) as handle:
        layers = list(handle.get("layers", {}))
        if isinstance(handle["obs"], h5py.Group) and "_index" in handle["var"].attrs:
            obs = read_elem(handle["obs"])
            var = handle["var"]
            gene_names = read_elem(var[var.attrs["_index"]])
            return obs, len(gene_names), layers

    # Older files store categorical codes in obs and their labels in uns. AnnData's
    # backwards-compatible reader rejoins them before we show any values to the user.
    data = ad.read_h5ad(path, backed="r")
    try:
        return data.obs, data.n_vars, layers
    finally:
        data.file.close()


def _counts(path: Path, layers: list[str]) -> str | None:
    with h5py.File(path) as handle:
        sources = (["X"] if "X" in handle else []) + [f"layers/{name}" for name in layers]
        source = _choose("Where are the original raw counts?", sources, "layers/counts", "X")
        matrix = handle[source]
        values = matrix["data"] if isinstance(matrix, h5py.Group) else matrix
        rows = max(1, 1_000_000 // max(1, int(np.prod(values.shape[1:]))))
        for start in range(0, values.shape[0], rows):
            block = values[start : start + rows]
            if block.dtype.kind not in "iuf" or not (
                np.isfinite(block).all() and (block >= 0).all() and (block == np.floor(block)).all()
            ):
                raise ValueError(
                    f"{source} contains values inconsistent with raw counts. Choose the original "
                    "nonnegative integer count matrix in X or a named layer."
                )
    return None if source == "X" else source.removeprefix("layers/")


def _abundance(config: dict[str, Any], obs: pd.DataFrame, *, infer_clusters: bool = False) -> None:
    columns = list(obs.columns)
    if infer_clusters:
        cell_type = (
            _choose(
                "Population labels, or Enter to compare inferred Leiden clusters",
                ["", *columns],
                "cell_type",
                "celltype_alt",
                "celltype",
                "",
            )
            or "leiden"
        )
    else:
        cell_type = _choose(
            "Column with existing cell-type labels", columns, "cell_type", "celltype"
        )
    if cell_type in columns:
        columns.remove(cell_type)
    donor = _choose(
        "Column identifying the biological replicate",
        columns,
        "donor",
        "donor_id",
        "patient_id",
        "replicate",
    )
    columns.remove(donor)
    condition = _choose(
        "Column identifying the condition", columns, "condition", "treatment", "stim", "label"
    )
    design_obs = (
        obs[[donor, condition] + ([cell_type] if cell_type in obs else [])]
        .dropna()
        .apply(as_label_strings)
    )
    levels = sorted(design_obs[condition].unique().tolist())
    if len(levels) < 2:
        raise ValueError("The labeled observations need at least two conditions for a comparison.")
    control = _choose("Reference condition", levels, "control", "ctrl")
    alternatives = [level for level in levels if level != control]
    case = _choose("Condition to compare against the reference", alternatives)
    eligible = design_obs[design_obs[condition].isin([case, control])]
    if eligible[donor].is_unique:
        raise ValueError(
            "Every observation has a different replicate ID. Choose the patient "
            "or animal ID shared by cells from the same biological replicate."
        )
    case_donors = set(eligible.loc[eligible[condition] == case, donor])
    control_donors = set(eligible.loc[eligible[condition] == control, donor])
    shared = case_donors & control_donors
    typer.echo(
        f"Before QC, {len(case_donors)} replicates in {case}, {len(control_donors)} in "
        f"{control}, {len(shared)} shared. {len(eligible):,} labeled observations in both arms."
    )
    paired = _choose("Study design", ["paired", "independent"]) == "paired"
    if not paired and shared:
        raise ValueError(
            "Replicate IDs occur in both arms. Check the IDs and use paired for "
            "repeated measurements of the same biological replicates."
        )
    design = DesignConfig(
        donor_col=donor, condition_col=condition, case=case, control=control, paired=paired
    )
    if paired:
        if len(shared) < 2:
            raise ValueError(
                "A paired comparison needs at least two complete biological replicates."
            )
        eligible = eligible[eligible[donor].isin(shared)]
        typer.echo(f"The paired test uses the {len(shared)} complete replicates before QC.")
    validate_design_against_obs(eligible, design=design)
    if cell_type in eligible and eligible[cell_type].nunique() < 2:
        raise ValueError("At least two cell-type labels are needed to compare their proportions.")
    config["design"] = design.model_dump(exclude_none=True)
    config["cohort"].update(
        donor_key=donor, condition_key=condition, condition_levels=[control, case]
    )
    config["composition"] = {"cell_type_key": cell_type}
    config["differential_abundance"] = {
        "methods": [{"method": "proportion_ttest"}],
        "cell_type_col": cell_type,
    }


def _free_layer(name: str, layers: list[str]) -> str:
    while name in layers:
        name = f"{name}_new"
    return name


def init_study(directory: Path, path: Path) -> None:
    """Inspect an input, confirm its assay and design, and write a runnable CPU configuration."""
    directory = directory.expanduser().resolve()
    config_path = directory / "config.yaml"
    if config_path.exists() or (directory / "run").exists():
        raise ValueError(
            f"{directory} already contains a config or run. Choose a new study folder."
        )
    path = path.resolve()
    if path.suffix.lower() == ".h5ad":
        _configure_study(directory, path)
        return

    from cellquorum.io.study_inputs import import_libraries

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".cellquorum-") as tmp:
        staging = Path(tmp)
        prepared = staging / "input.h5ad"
        manifest = import_libraries(path, prepared)
        for heading in manifest:
            key = str(heading)
            if key == "path" or key.endswith("_path"):
                manifest[key] = manifest[key].map(
                    lambda value: os.path.relpath(value, directory), na_action="ignore"
                )
        manifest.to_csv(staging / "samples.csv", index=False)
        typer.echo(
            f"Imported {len(manifest)} libraries. Available files: {', '.join(manifest.columns)}."
        )
        _configure_study(directory, prepared, staging=staging)


def _configure_study(directory: Path, path: Path, *, staging: Path | None = None) -> None:
    config_path = directory / "config.yaml"
    obs, genes, layers = _metadata(path)
    if len(obs) == 0 or genes == 0:
        raise ValueError("The input needs observations and genes.")
    typer.echo(f"Opened {path.name} with {len(obs):,} observations and {genes:,} genes.")
    typer.echo(
        "Discovery considers all study questions. Questions selects specific analyses. "
        "Explore runs QC, clusters and UMAP. Abundance "
        "compares cell types across biological replicates."
    )
    analysis = _choose("Analysis", ["discovery", "questions", "explore", "abundance"], "discovery")
    counts = _counts(path, layers)
    organism = _choose("Organism", ["human", "mouse"])
    assay = _choose("Preparation", ["cells", "nuclei", "mixed"])
    graded: dict[str, Any] = {"archetype_audit": False}
    if assay == "nuclei":
        graded["nuclear_axis_applicable"] = False
    elif assay == "mixed":
        key = _choose("Column identifying the preparation", list(obs.columns), "assay", "protocol")
        values = sorted(obs[key].dropna().astype(str).unique().tolist())
        graded.update(
            nuclei_key=key,
            nuclei_values=[_choose("Value identifying nuclei", values, "nuclei", "nucleus")],
        )
    columns = list(obs.columns)
    typer.echo("Library IDs support QC by library. Press Enter if no library ID is available.")
    sample = _choose("Library ID column", ["", *columns], "sample_id", "") if columns else ""
    enabled = {"qc_evidence"}
    config: dict[str, Any] = {
        "project": {
            "name": directory.name,
            "organism": organism,
            "species_id": 9606 if organism == "human" else 10090,
        },
        "input": {
            "h5ad": "input.h5ad" if staging else os.path.relpath(path, directory),
            "counts_layer": counts,
        },
        "paths": {
            "relative_to_config": True,
            "output_dir": "run",
            **({"manifest": "samples.csv"} if staging else {}),
        },
        "compute": {"backend": "cpu", "prefer_gpu": False},
        "r": {"enabled": False},
        "cohort": {"sample_key": sample} if sample else {},
        "qc_evidence": {
            "metrics": {"layer": counts},
            "graded": graded,
            "ambient": {"enabled": False},
            "doublets": {"enabled": False},
        },
        "report": {"enabled": True, "html": True, "markdown": True},
    }
    if organism == "mouse":
        config["qc_evidence"]["features"] = {
            "mitochondrial_prefixes": ["mt-", "MT-"],
            "ribosomal_prefixes": ["Rps", "Rpl", "RPS", "RPL"],
            "hemoglobin_regexes": ["^Hb[ab]-", "^HB[AB]"],
        }
    floor = QCFloorConfig().min_genes_per_cell
    if floor is not None and genes < floor:
        typer.echo(
            f"This file has {genes} genes. The usual QC floor of {floor} detected genes "
            "per cell would discard every observation."
        )
        while True:
            minimum = typer.prompt(
                f"Minimum detected genes per cell for this assay (1–{genes})", type=int
            )
            if 1 <= minimum <= genes:
                break
            typer.echo(f"Enter a whole number from 1 to {genes}.")
        config["qc_evidence"]["floors"] = {"min_genes_per_cell": minimum}
    if analysis == "abundance":
        _abundance(config, obs)
        enabled.update(["composition", "differential_abundance"])
        table = directory / "run/results/da_proportion_ttest.csv"
        expected = f"Donor-level abundance results: {table}"
    else:
        enabled.update(
            ["preprocessing", "feature_selection", "dimensionality", "clustering", "embeddings"]
        )
        normalized = _free_layer("cellquorum_normalized", layers)
        preserved = counts or _free_layer("counts", layers)
        config["preprocessing"] = {
            "normalization": {
                "input_layer": counts,
                "output_layer": normalized,
                "preserve_counts_layer": preserved,
                "recipe": "cellquorum_log1p_cp10k_v1",
            }
        }
        config["feature_selection"] = {"counts_layer": preserved, "lognorm_layer": normalized}
        config["dimensionality"] = {"input_layer": normalized}
        config["clustering"] = {"use_rep": "X_pca"}
        config["embeddings"] = {
            "use_rep": "X_pca",
            "embeddings": ["umap"],
            "color_by": ["leiden"],
            "methods": [{"method": m} for m in ("umap", "paga", "categorical_embedding")],
        }
        expected = "The report links to QC figures and a UMAP colored by Leiden cluster."
    if analysis in {"discovery", "questions"}:
        from cellquorum.cli.study_questions import configure_questions

        manifest = pd.read_csv(staging / "samples.csv") if staging else None
        configure_questions(config, enabled, obs, layers, analysis, manifest)
        for field, keys in {
            "grn": ("tfs_path", "motifs_path", "rankings_glob"),
            "cell_cell_communication": (
                "nichenet_lr_network",
                "nichenet_weighted_networks",
                "nichenet_ligand_target_matrix",
            ),
            "reference_mapping": ("atlas_h5ad",),
        }.items():
            for key in keys:
                value = config.get(field, {}).get(key)
                if value:
                    config[field][key] = os.path.relpath(
                        Path(value).expanduser().resolve(), directory
                    )
        for method in config.get("enrichment", {}).get("methods", []):
            if method.get("gmt_path"):
                method["gmt_path"] = os.path.relpath(
                    Path(method["gmt_path"]).expanduser().resolve(), directory
                )
        expected = "The report groups the requested analyses and links their evidence."
    config["stages"] = {key: key in enabled for key in StageSelectionConfig.model_fields}
    CellQuorumConfig.model_validate(config)
    contents = yaml.safe_dump(config, sort_keys=False)
    directory.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        if staging:
            for name in ("input.h5ad", "samples.csv"):
                os.link(staging / name, directory / name)
                created.append(directory / name)
        with config_path.open("x", encoding="utf-8") as handle:
            created.append(config_path)
            handle.write(contents)
    except BaseException:
        for target in reversed(created):
            target.unlink(missing_ok=True)
        raise
    typer.echo(f"Wrote {config_path}")
    for command in ("plan", "run"):
        typer.echo(shlex.join(["cellquorum", command, "-c", str(config_path)]))
    typer.echo(f"{expected}\nOpen {directory / 'run/reports/report.html'} after the run.")
