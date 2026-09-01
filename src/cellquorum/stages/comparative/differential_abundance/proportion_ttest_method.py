"""Proportion t-test differential-abundance method (pure Python)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.comparative.differential_abundance.aggregation import (
    aggregate_celltype_counts,
    build_cell_distribution_summary,
)


class ProportionTTestMethod(AnalysisMethod):
    """Paired/unpaired proportion t-test for differential abundance.

    This method generalizes the project owner's manuscript-validated per-cell-type
    proportion test into a pure-Python differential-abundance method. It computes
    cell-type proportions per sample, applies arcsin-sqrt transformation, and performs
    paired (default) or unpaired t-tests with bootstrap confidence intervals.

    This is the trusted default anchor method — pure Python (scipy + statsmodels),
    no R dependencies, essentially always runs.
    """

    name = "proportion_ttest"
    stage_category = "differential_abundance"
    backend = "python"

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
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col", "cell_type")
        return [condition_col, donor_col, cell_type_col]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        """Compute per-cell-type proportion tests and return the DA table."""

        # Resolve config fields (all schema-driven; no hardcoded study assumptions).
        condition_col = config.get("condition_col", "condition")
        donor_col = config.get("donor_col", "patient_id")
        cell_type_col = config.get("cell_type_col", "cell_type")
        case = config.get("case")
        control = config.get("control")
        paired = bool(config.get("paired", True))  # default True (anchor is paired)
        n_bootstrap = int(config.get("n_bootstrap", 10000))
        seed = int(config.get("seed", 42))
        fdr_method = config.get("fdr_method", "fdr_bh")

        # Guard: case/control unset → skip
        if not case or not control:
            return self._skip("case/control labels not set in config")

        # Aggregate to sample × cell-type counts
        cc = aggregate_celltype_counts(
            adata,
            donor_col=donor_col,
            condition_col=condition_col,
            cell_type_col=cell_type_col,
        )

        # Compute proportions: counts / total_per_sample
        total_per_sample = cc.counts.sum(axis=1)
        proportion_df = cc.counts.div(total_per_sample, axis=0)

        # Attach donor and condition from sample_meta (align on shared index)
        # cc.sample_meta has columns literally named by condition_col and donor_col
        sample_df = proportion_df.copy()
        sample_df[donor_col] = cc.sample_meta[donor_col]
        sample_df[condition_col] = cc.sample_meta[condition_col]

        # Restrict to {case, control} samples
        mask = sample_df[condition_col].isin([case, control])
        sample_df = sample_df[mask].copy()

        # Count samples per arm
        case_samples = sample_df[sample_df[condition_col] == case]
        control_samples = sample_df[sample_df[condition_col] == control]

        # Guard: <2 samples per arm → skip
        if len(case_samples) < 2 or len(control_samples) < 2:
            return self._skip(
                "need ≥2 samples per arm",
                n_case=len(case_samples),
                n_control=len(control_samples),
            )

        # Initialize artifact writer for result table (used in both paired/unpaired branches)
        writer = StageArtifactWriter.from_context(context)

        # Paired branch
        if paired:
            # Identify donors present in BOTH case and control
            case_donors = set(case_samples[donor_col].unique())
            control_donors = set(control_samples[donor_col].unique())
            paired_donors = case_donors & control_donors

            # Guard: <2 paired donors → skip
            if len(paired_donors) < 2:
                return self._skip(
                    "<2 donors present in both arms for paired test",
                    n_paired=len(paired_donors),
                )

            # Restrict to paired donors
            paired_mask = sample_df[donor_col].isin(paired_donors)
            sample_df = sample_df[paired_mask].copy()

            # Build results for each cell type
            results = []
            cell_types = [col for col in cc.counts.columns]

            for ct in cell_types:
                # Get proportions for this cell type by donor×condition
                ct_data = sample_df[[ct, donor_col, condition_col]].copy()
                ct_data.columns = ["proportion", "donor", "condition"]

                # Pivot to donor × condition
                pivot = ct_data.pivot(index="donor", columns="condition", values="proportion")

                # Ensure we have both case and control columns
                if case not in pivot.columns or control not in pivot.columns:
                    # This shouldn't happen given our guards, but be defensive
                    continue

                # Align to paired donors (drop any NaN rows)
                pivot = pivot.dropna()
                if len(pivot) < 2:
                    # Not enough paired donors for this cell type
                    continue

                case_props = pivot[case].values
                control_props = pivot[control].values

                # Transform: arcsin(sqrt(proportion))
                arcsin_case = np.arcsin(np.sqrt(case_props))
                arcsin_control = np.arcsin(np.sqrt(control_props))

                # Delta in percentage points: 100*(case - control)
                delta_pp = 100 * (case_props - control_props)

                # Bootstrap CI
                ci_low, ci_high = _bootstrap_ci(delta_pp, seed=seed, n_bootstrap=n_bootstrap)

                # Paired t-test
                try:
                    test_result = stats.ttest_rel(arcsin_case, arcsin_control)
                    statistic = test_result.statistic
                    pvalue = test_result.pvalue
                except Exception:
                    # Handle zero-variance case gracefully
                    statistic = np.nan
                    pvalue = np.nan

                # Mean percentages
                control_mean_pct = 100 * control_props.mean()
                case_mean_pct = 100 * case_props.mean()
                effect_pp = case_mean_pct - control_mean_pct

                results.append(
                    {
                        "cell_type": ct,
                        "n_case": len(paired_donors),
                        "n_control": len(paired_donors),
                        "control_mean_pct": control_mean_pct,
                        "case_mean_pct": case_mean_pct,
                        "effect_pp": effect_pp,
                        "bootstrap_ci_low_pp": ci_low,
                        "bootstrap_ci_high_pp": ci_high,
                        "statistic": statistic,
                        "pvalue": pvalue,
                        "paired": True,
                    }
                )

            # Convert to DataFrame
            results_df = pd.DataFrame(results)

            # Compute FDR (mask out NaN pvalues for multipletests)
            pvalues = results_df["pvalue"].values
            finite_mask = np.isfinite(pvalues)
            fdr_values = np.full_like(pvalues, np.nan)
            if finite_mask.sum() > 0:
                fdr_values[finite_mask] = multipletests(pvalues[finite_mask], method=fdr_method)[1]
            results_df["fdr"] = fdr_values

            # Count significant results (fdr < 0.05, ignoring NaN)
            n_significant = int((results_df["fdr"] < 0.05).sum())

            # Return result
            summary_artifacts = self._distribution_summary_artifacts(
                cc.counts,
                cc.sample_meta[condition_col],
                case=case,
                control=control,
                results_df=results_df,
                config=config,
                writer=writer,
            )
            return StageResult(
                adata=adata,
                artifacts=[
                    writer.table(
                        results_df,
                        "da_proportion_ttest.csv",
                        name="da_results",
                        description=f"Proportion t-test DA (paired, {case} vs {control}).",
                        index=False,
                    ),
                    *summary_artifacts,
                ],
                notes=[f"Proportion t-test DA (paired): {case} vs {control}."],
                metrics={
                    "case": case,
                    "control": control,
                    "paired": True,
                    "n_celltypes": len(cell_types),
                    "n_donors_paired": len(paired_donors),
                    "n_significant": n_significant,
                    "seed": seed,
                    "n_bootstrap": n_bootstrap,
                },
                backend="python",
            )

        else:
            # Unpaired branch
            # Gather case and control proportions independently
            results = []
            cell_types = [col for col in cc.counts.columns]

            for ct in cell_types:
                case_props = case_samples[ct].values
                control_props = control_samples[ct].values

                # Transform: arcsin(sqrt(proportion))
                arcsin_case = np.arcsin(np.sqrt(case_props))
                arcsin_control = np.arcsin(np.sqrt(control_props))

                # Delta in percentage points: 100*(mean_case - mean_control)
                case_mean = case_props.mean()
                control_mean = control_props.mean()
                effect_pp = 100 * (case_mean - control_mean)

                # Bootstrap CI on the difference (unpaired: resample both arms independently)
                ci_low, ci_high = _bootstrap_ci_unpaired(
                    case_props, control_props, seed=seed, n_bootstrap=n_bootstrap
                )

                # Unpaired t-test (Welch)
                try:
                    test_result = stats.ttest_ind(arcsin_case, arcsin_control, equal_var=False)
                    statistic = test_result.statistic
                    pvalue = test_result.pvalue
                except Exception:
                    statistic = np.nan
                    pvalue = np.nan

                # Mean percentages
                control_mean_pct = 100 * control_mean
                case_mean_pct = 100 * case_mean

                results.append(
                    {
                        "cell_type": ct,
                        "n_case": len(case_samples),
                        "n_control": len(control_samples),
                        "control_mean_pct": control_mean_pct,
                        "case_mean_pct": case_mean_pct,
                        "effect_pp": effect_pp,
                        "bootstrap_ci_low_pp": ci_low,
                        "bootstrap_ci_high_pp": ci_high,
                        "statistic": statistic,
                        "pvalue": pvalue,
                        "paired": False,
                    }
                )

            # Convert to DataFrame
            results_df = pd.DataFrame(results)

            # Compute FDR
            pvalues = results_df["pvalue"].values
            finite_mask = np.isfinite(pvalues)
            fdr_values = np.full_like(pvalues, np.nan)
            if finite_mask.sum() > 0:
                fdr_values[finite_mask] = multipletests(pvalues[finite_mask], method=fdr_method)[1]
            results_df["fdr"] = fdr_values

            # Count significant results
            n_significant = int((results_df["fdr"] < 0.05).sum())

            # Return result
            summary_artifacts = self._distribution_summary_artifacts(
                cc.counts,
                cc.sample_meta[condition_col],
                case=case,
                control=control,
                results_df=results_df,
                config=config,
                writer=writer,
            )
            return StageResult(
                adata=adata,
                artifacts=[
                    writer.table(
                        results_df,
                        "da_proportion_ttest.csv",
                        name="da_results",
                        description=f"Proportion t-test DA (unpaired, {case} vs {control}).",
                        index=False,
                    ),
                    *summary_artifacts,
                ],
                notes=[f"Proportion t-test DA (unpaired): {case} vs {control}."],
                metrics={
                    "case": case,
                    "control": control,
                    "paired": False,
                    "n_celltypes": len(cell_types),
                    "n_case": len(case_samples),
                    "n_control": len(control_samples),
                    "n_significant": n_significant,
                    "seed": seed,
                    "n_bootstrap": n_bootstrap,
                },
                backend="python",
            )

    def _distribution_summary_artifacts(
        self,
        counts: pd.DataFrame,
        conditions: pd.Series,
        *,
        case: str,
        control: str,
        results_df: pd.DataFrame,
        config: dict,
        writer: StageArtifactWriter,
    ) -> list:
        """Build the Cell Distribution Summary artifact (gated, default on).

        Emits a pooled per-cell-type composition table (absolute counts +
        within-condition relative %, with case-arm p/FDR) alongside the DA
        result. Uses the full case/control sample set for the descriptive
        counts, independent of any paired-donor restriction the test applies.
        """

        if not config.get("write_distribution_summary", True):
            return []

        summary = build_cell_distribution_summary(
            counts,
            conditions,
            case=case,
            control=control,
            test_results=results_df,
        )
        return [
            writer.table(
                summary,
                "cell_distribution_summary.csv",
                name="cell_distribution_summary",
                description=(
                    f"Pooled cell-type composition ({case} vs {control}): absolute "
                    "counts + within-condition relative %, with case-arm p/FDR."
                ),
                index=False,
            )
        ]


def _bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int) -> tuple[float, float]:
    """
    Bootstrap 95% confidence interval on the mean of values.

    Args:
        values: 1D array of values (e.g., paired differences).
        seed: Random seed for reproducibility.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Tuple of (lower_bound, upper_bound) at 2.5% and 97.5% quantiles.
    """
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(boot_means, [0.025, 0.975]))


def _bootstrap_ci_unpaired(
    case_props: np.ndarray,
    control_props: np.ndarray,
    seed: int,
    n_bootstrap: int,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI on the difference of means (unpaired).

    Args:
        case_props: 1D array of case proportions.
        control_props: 1D array of control proportions.
        seed: Random seed.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Tuple of (lower_bound, upper_bound) in percentage points.
    """
    rng = np.random.default_rng(seed)
    n_case = len(case_props)
    n_control = len(control_props)

    boot_diffs = []
    for _ in range(n_bootstrap):
        boot_case = rng.choice(case_props, size=n_case, replace=True).mean()
        boot_control = rng.choice(control_props, size=n_control, replace=True).mean()
        boot_diffs.append(100 * (boot_case - boot_control))

    boot_diffs = np.array(boot_diffs)
    return tuple(np.quantile(boot_diffs, [0.025, 0.975]))


__all__ = ["ProportionTTestMethod"]
