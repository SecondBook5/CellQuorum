#!/usr/bin/env python
"""Smoke test script for real h5ad files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import click


@click.command()
@click.option(
    "--input-h5ad",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input h5ad file",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for smoke test",
)
@click.option(
    "--n-cells",
    type=int,
    default=200,
    help="Number of cells to subset",
)
@click.option(
    "--n-genes",
    type=int,
    default=500,
    help="Number of genes to subset",
)
@click.option(
    "--counts-layer",
    type=str,
    default=None,
    help="Layer to use as counts (if provided, sets X to this layer)",
)
@click.option(
    "--recipe",
    type=str,
    default="cellquorum_log1p_cp10k_v1",
    help="Normalization recipe to use",
)
@click.option(
    "--overwrite-output",
    is_flag=True,
    help="Overwrite existing output directory",
)
def main(
    input_h5ad: Path,
    output_dir: Path,
    n_cells: int,
    n_genes: int,
    counts_layer: str | None,
    recipe: str,
    overwrite_output: bool,
) -> None:
    """
    Run CellQuorum smoke test on a real h5ad file.

    This script:
    1. Loads and subsets the input h5ad
    2. Optionally sets X to a specified counts layer
    3. Writes the subset as _subset_input.h5ad
    4. Runs CellQuorum with QC and preprocessing
    5. Validates the outputs
    6. Prints a JSON summary
    """
    # Import CellQuorum after click setup to avoid import overhead for --help.
    from cellquorum import run_pipeline
    from cellquorum.config.models import (
        CellQuorumConfig,
        ComputeConfig,
        InputConfig,
        PreprocessingConfig,
        ProjectConfig,
        QCConfig,
        RConfig,
    )
    from cellquorum.preprocessing.config import NormalizationConfig

    # Create output directory.
    output_dir.mkdir(parents=True, exist_ok=overwrite_output)

    # Load input h5ad.
    click.echo(f"Loading {input_h5ad}...")
    adata = ad.read_h5ad(input_h5ad)
    click.echo(f"Original shape: {adata.shape}")

    # Make var_names unique to avoid downstream issues.
    adata.var_names_make_unique()

    # Subset deterministically.
    n_cells_actual = min(n_cells, adata.n_obs)
    n_genes_actual = min(n_genes, adata.n_vars)
    subset = adata[:n_cells_actual, :n_genes_actual].copy()
    click.echo(f"Subset shape: {subset.shape}")

    # Optionally set X to counts layer.
    if counts_layer is not None:
        if counts_layer not in subset.layers:
            click.echo(
                f"Error: Layer '{counts_layer}' not found. "
                f"Available layers: {list(subset.layers.keys())}",
                err=True,
            )
            sys.exit(1)

        subset.X = subset.layers[counts_layer].copy()
        click.echo(f"Set X to layer '{counts_layer}'")

    # Write subset input.
    subset_input_path = output_dir / "_subset_input.h5ad"
    subset.write_h5ad(subset_input_path)
    click.echo(f"Wrote subset: {subset_input_path}")

    # Build CellQuorum config.
    config = CellQuorumConfig(
        project=ProjectConfig(name="smoke_test"),
        input=InputConfig(h5ad=subset_input_path),
        compute=ComputeConfig(backend="cpu", prefer_gpu=False, fallback_to_cpu=True),
        r=RConfig(enabled=False),
        qc=QCConfig(
            mode="flag_no_drop",
            threshold_strategy="fixed",
            metrics={"percent_top": [20]},
            basic={
                "min_genes_per_cell": 1,
                "min_cells_per_gene": 1,
                "max_mito_percent": 100.0,
            },
            mad={"enabled": False},
            outputs={
                "write_h5ad": False,
                "write_figures": False,
            },
        ),
        preprocessing=PreprocessingConfig(
            enabled=True,
            normalization=NormalizationConfig(
                enabled=True,
                recipe=recipe,
                input_layer=counts_layer,
                output_layer="cellquorum_normalized",
                preserve_counts_layer="counts",
                overwrite=True,
            ),
        ),
    )

    # Run pipeline.
    run_dir = output_dir / "run"
    click.echo("\nRunning CellQuorum pipeline...")
    result = run_pipeline(config, output_dir=run_dir, execute=True, load_input=True)

    # Validate execution.
    execution_result = result.execution_result
    if execution_result is None:
        click.echo("Error: Pipeline did not execute", err=True)
        sys.exit(1)

    succeeded = execution_result.succeeded_stage_names()
    failed = execution_result.failed_stage_names()

    if "qc" not in succeeded:
        click.echo("Error: QC stage did not succeed", err=True)
        sys.exit(1)

    if "preprocessing" not in succeeded:
        click.echo("Error: Preprocessing stage did not succeed", err=True)
        sys.exit(1)

    if failed:
        click.echo(f"Error: Stages failed: {failed}", err=True)
        sys.exit(1)

    # Validate outputs.
    final_adata = result.context.adata

    if "counts" not in final_adata.layers:
        click.echo("Error: 'counts' layer not in final AnnData", err=True)
        sys.exit(1)

    if "cellquorum_normalized" not in final_adata.layers:
        click.echo("Error: 'cellquorum_normalized' layer not in final AnnData", err=True)
        sys.exit(1)

    # Validate artifacts.
    preprocessing_summary = run_dir / "results" / "preprocessing" / "preprocessing_summary.json"
    if not preprocessing_summary.exists():
        click.echo(f"Error: {preprocessing_summary} not found", err=True)
        sys.exit(1)

    stage_records = run_dir / "provenance" / "stage_execution_records.json"
    if not stage_records.exists():
        click.echo(f"Error: {stage_records} not found", err=True)
        sys.exit(1)

    # Build summary.
    summary = {
        "status": "success",
        "input_h5ad": str(input_h5ad),
        "input_shape": list(adata.shape),
        "subset_shape": list(subset.shape),
        "subset_input_path": str(subset_input_path),
        "run_dir": str(run_dir),
        "successful_stages": succeeded,
        "skipped_stages": execution_result.skipped_stage_names(),
        "failed_stages": failed,
        "final_shape": list(final_adata.shape),
        "final_layers": list(final_adata.layers.keys()),
        "has_qc_annotations": "cellquorum_qc_keep" in final_adata.obs,
        "has_normalized_layer": "cellquorum_normalized" in final_adata.layers,
        "has_counts_layer": "counts" in final_adata.layers,
        "preprocessing_summary_exists": preprocessing_summary.exists(),
        "stage_records_exists": stage_records.exists(),
        "recipe": recipe,
    }

    # Print JSON summary.
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
