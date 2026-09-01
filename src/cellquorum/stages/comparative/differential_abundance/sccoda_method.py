"""scCODA Bayesian compositional differential-abundance method."""

from __future__ import annotations

import subprocess
from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.backends.sccoda_backend import SCCODA_HELPER
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.comparative.differential_abundance.aggregation import (
    aggregate_celltype_counts,
)


class SccodaMethod(AnalysisMethod):
    """scCODA Bayesian compositional DA test via sccoda_env subprocess.

    scCODA tests for cell-type compositional differences between conditions using
    a Bayesian model over the simplex (spec §DA). Aggregates cells to sample ×
    cell-type counts, fits a compositional model with an automatic or explicit
    reference cell type, and identifies credible abundance changes via hierarchical
    modeling with spike-and-slab priors.
    """

    name = "sccoda"
    stage_category = "differential_abundance"
    backend = "sccoda"

    def input_contract(self, config: dict) -> DataContract:
        """Require the design obs columns (no layer needed for DA)."""
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col", "cell_type")
        return DataContract(
            required_obs=[condition_col, donor_col, cell_type_col],
        )

    def requires_obs(self, config: dict) -> list[str]:
        """Return the design obs columns that must exist for DA to run."""

        # Read the design columns from config.
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col", "cell_type")

        # Require all design columns to exist.
        return [condition_col, donor_col, cell_type_col]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        """Aggregate cell-type counts, fit scCODA, and return the DA table."""

        # Resolve config fields (all schema-driven; no hardcoded study assumptions).
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col", "cell_type")
        case = config.get("case")
        control = config.get("control")
        reference_celltype = config.get("reference_celltype")
        seed = int(config.get("seed", 0))
        num_iterations = int(config.get("num_iterations", 20000))
        timeout = int(config.get("timeout_seconds", 3600))

        # A comparison needs both case and control labels.
        if not case or not control:
            return self._skip("case/control labels not set in config")

        # Resolve the sccoda backend from the context registry.
        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("sccoda")
            except Exception:
                backend = None
        if backend is None:
            return self._skip("sccoda backend unavailable")

        # Check backend availability (micromamba + sccoda_env import).
        status = backend.status()
        if not status.available:
            missing_str = ", ".join(status.missing) if status.missing else "unknown"
            return self._skip(f"sccoda backend unavailable ({missing_str})", missing=status.missing)

        # Aggregate to sample × cell-type counts.
        cc = aggregate_celltype_counts(
            adata,
            donor_col=donor_col,
            condition_col=condition_col,
            cell_type_col=cell_type_col,
        )

        # Write aggregated inputs to scratch.
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        counts_meta_csv = scratch / "da_counts_meta.csv"

        # counts_meta.csv: samples × cell types + condition column.
        # Join counts with the condition column from sample_meta.
        counts_with_meta = cc.counts.copy()
        counts_with_meta[condition_col] = cc.sample_meta[condition_col]
        counts_with_meta.to_csv(counts_meta_csv, index=True)

        # Prepare the output path in the run results directory.
        results_dir = Path(context.paths.results)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_csv = results_dir / "da_sccoda.csv"

        # Build helper args.
        args = [
            str(counts_meta_csv),
            str(out_csv),
            condition_col,
            case,
            control,
            str(seed),
            str(num_iterations),
        ]

        # Add reference_celltype if set.
        if reference_celltype:
            args.append(reference_celltype)

        # Invoke the sccoda helper; non-zero exit -> recorded skip (never crash).
        try:
            proc = backend.run_helper(SCCODA_HELPER, args, timeout=timeout)
        except FileNotFoundError as exc:
            return self._skip("helper execution failed", error=str(exc)[:500])
        except subprocess.TimeoutExpired as exc:
            # A configured timeout must skip this method, not crash the stage
            # and abort the sibling methods still queued after it.
            return self._skip(f"helper execution timed out after {timeout}s", error=str(exc)[:500])
        if proc.returncode != 0:
            return self._skip("sccoda helper failed", stderr=proc.stderr.strip()[:500])

        # Read output CSV to compute metrics + back the composition figure (skip-not-crash).
        df = None
        n_credible = None
        try:
            df = pd.read_csv(out_csv)
            n_credible = int(df["credible_effect"].sum())
        except Exception:
            pass  # CSV should exist but don't crash if reading fails

        # Auto-emit the compositional DA figure (gated, skip-not-crash).
        figure_artifacts = self._composition_artifacts(
            df,
            cc,
            condition_col=condition_col,
            donor_col=donor_col,
            case=case,
            control=control,
            config=config,
            context=context,
        )

        # Return the DA table as an artifact plus provenance metrics.
        return StageResult(
            adata=adata,
            artifacts=[
                StageArtifact(
                    name="da_results",
                    path=out_csv,
                    kind="csv",
                    description=(
                        f"scCODA DA ({case} vs {control}), "
                        f"reference={reference_celltype or 'auto'}, "
                        f"iterations={num_iterations}."
                    ),
                ),
                *figure_artifacts,
            ],
            notes=[
                f"scCODA DA: {case} vs {control}, "
                f"reference={reference_celltype or 'auto'}, "
                f"iterations={num_iterations}."
            ],
            metrics={
                "case": case,
                "control": control,
                "reference_celltype": reference_celltype,
                "seed": seed,
                "num_iterations": num_iterations,
                "n_samples": int(cc.counts.shape[0]),
                "n_celltypes": int(cc.counts.shape[1]),
                "n_credible": n_credible,
            },
            backend="sccoda",
        )

    def _composition_artifacts(
        self,
        effects: pd.DataFrame | None,
        cc: object,
        *,
        condition_col: str,
        donor_col: str,
        case: str,
        control: str,
        config: dict,
        context: object,
    ) -> list[StageArtifact]:
        """Build the two-panel scCODA composition figure (gated, default on).

        Pairs a per-condition proportion dumbbell with each cell type's posterior
        inclusion probability, driven by the scCODA effects table and the
        aggregated sample × cell-type counts. Study-agnostic: condition
        labels/colors come from config, no biology hardcoded. Emits nothing when
        disabled, when the effects table is unreadable/empty, or when the counts
        yield no proportions. Never raises — a plotting failure skips the figure.
        """

        if not config.get("write_da_figure", True):
            return []
        if effects is None or effects.empty:
            return []

        # Local imports keep matplotlib off the pure-method import path.
        from cellquorum.stages.comparative.differential_abundance.aggregation import (
            build_composition_proportions,
        )
        from cellquorum.stages.comparative.differential_abundance.da_figures import (
            plot_sccoda_composition,
        )
        from cellquorum.visualization.figstyle import save_figure

        try:
            proportions = build_composition_proportions(
                cc.counts,
                cc.sample_meta[condition_col],
                cc.sample_meta[donor_col],
                case=case,
                control=control,
            )
            if proportions.empty:
                return []
            fig = plot_sccoda_composition(effects, proportions, case=case, control=control)
        except Exception:
            return []

        figures_dir = Path(getattr(context.paths, "figures", context.paths.results))
        figures_dir = figures_dir / "differential_abundance"
        artifacts: list[StageArtifact] = []
        try:
            for path in save_figure(fig, figures_dir, "da_sccoda_composition"):
                artifacts.append(
                    StageArtifact(
                        name="da_sccoda_composition",
                        path=path,
                        kind="figure",
                        description=(
                            f"scCODA compositional DA ({case} vs {control}): per-condition "
                            "proportion dumbbell + posterior inclusion probability."
                        ),
                    )
                )
        except Exception:
            return []
        return artifacts


__all__ = ["SccodaMethod"]
