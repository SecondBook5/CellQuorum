"""In-env scCODA helper: Bayesian compositional differential abundance.

Runs INSIDE the isolated sccoda_env environment (invoked by ``SccodaBackend``), so
it may import sccoda freely. Data crosses the process boundary as files:

INPUT:
    <counts_meta.csv>: samples × cell-types integer counts PLUS one extra column
                      named by <condition_col> arg (values = case/control labels).

ARGS:
    <counts_meta_csv> <out_csv> <condition_col> <case> <control> <seed>
    <num_iterations> [reference_celltype]
    [--covariates COL,COL] [--diagnostics-json PATH]

OUTPUT:
    <out_csv>: rows = cell types (optionally two sets if reference_celltype given),
               columns = cell_type, log2_fold_change, inclusion_probability,
               credible_effect, reference
    <diagnostics-json>: per-fit sampler statistics, so the caller can tell a
               genuine null apart from a fit that never converged

Exit code 0 on success; non-zero with a message on stderr otherwise (the caller
inspects the return code and raises a domain-specific error).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING

# CRITICAL: these must be set BEFORE importing tensorflow/sccoda, because each is
# read once when the library initialises.
#
# TF_DETERMINISTIC_OPS selects deterministic kernels. The two thread counts pin the
# BLAS/OpenMP pools that decide floating-point reduction ORDER: left to their own
# devices they are sized from the visible CPU count, so the same seed on the same
# input can sum a reduction in a different order depending on how busy the machine
# was when the process started, and an HMC chain amplifies that last-bit difference
# into a different chain. See the comment in ``_fit_sccoda`` for what that cost.
#
# Assigned, not ``setdefault``. ``setdefault`` yields to whatever the calling
# environment happens to export, which means a machine with OMP_NUM_THREADS already
# set to its core count would silently skip the pin and produce a fit that does not
# reproduce -- the one failure mode this is here to prevent. A published posterior
# must not depend on the shell that launched it, so the override is refused.
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

if TYPE_CHECKING:
    import pandas as pd


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and fit scCODA model(s)."""

    parser = argparse.ArgumentParser(description="scCODA in-env fit helper.")
    parser.add_argument("counts_meta_csv", help="Input CSV: samples × cell types + condition col")
    parser.add_argument("out_csv", help="Output CSV: DA results")
    parser.add_argument("condition_col", help="Name of condition column in input CSV")
    parser.add_argument("case", help="Case label")
    parser.add_argument("control", help="Control label")
    parser.add_argument("seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("num_iterations", type=int, help="Number of HMC iterations")
    parser.add_argument(
        "reference_celltype",
        nargs="?",
        default=None,
        help="Optional explicit reference cell type (if set, runs two fits)",
    )
    parser.add_argument(
        "--covariates",
        default="",
        help=(
            "Comma-separated extra columns of the input CSV to add to the model "
            "formula, e.g. the donor column to make a paired design"
        ),
    )
    parser.add_argument(
        "--diagnostics-json",
        default=None,
        help="Optional path to write per-fit sampler diagnostics as JSON",
    )

    args = parser.parse_args(argv)

    try:
        return int(_fit_sccoda(args))
    except Exception as exc:
        print(f"sccoda helper failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _fit_sccoda(args: argparse.Namespace) -> int:
    """Run scCODA fit(s) and write results."""

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sccoda.util import cell_composition_data as ccd

    # Determinism across separate processes. scCODA's ``sample_hmc`` calls
    # ``tfp.mcmc.sample_chain`` without a ``seed=`` argument and exposes no seed
    # parameter, so ``tf.random.set_seed`` alone does NOT pin the stateful HMC
    # RNG — identical-seed runs otherwise diverge (differing acceptance rates and
    # inclusion probabilities), flipping borderline cell types across the FDR
    # threshold. Op-level determinism pins the stateful ops so the chain is
    # byte-identical across runs. Must be enabled before any TF op is built.
    tf.config.experimental.enable_op_determinism()

    # Op determinism is necessary and NOT sufficient. It fixes which algorithm each
    # op uses; it does not fix how many threads the op gets, and a parallel sum is
    # only associative in exact arithmetic. Left at the default 0 both pools are
    # sized from the visible CPU count at process start, so two runs of this script
    # with the SAME seed and a byte-identical input CSV were observed to return
    # different posteriors -- inclusion probabilities of 0.729 vs 0.875 and 0.647 vs
    # 0.813 on a 3-cell-type cohort. ``set_fdr`` thresholds exactly those numbers,
    # so the boolean ``credible_effect`` flipped: the same data called a cell type
    # credibly changed or not depending on machine load. Single-threaded execution
    # fixes the reduction order, and these models are small enough that the wall
    # clock barely notices. Must precede any op construction.
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # Set seeds for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Read input CSV
    df = pd.read_csv(args.counts_meta_csv, index_col=0)

    # Resolve the extra model terms, keeping only columns that are actually here.
    # A silently-dropped covariate would change the model without changing the
    # reported formula, so the resolved list is echoed back in the diagnostics.
    requested = [c.strip() for c in str(args.covariates or "").split(",") if c.strip()]
    covariate_cols = [c for c in requested if c in df.columns and c != args.condition_col]
    missing = [c for c in requested if c not in df.columns]

    # Separate cell type columns from the metadata columns
    meta_cols = [args.condition_col, *covariate_cols]
    cell_cols = [c for c in df.columns if c not in meta_cols]

    # Build scCODA data object
    data = ccd.from_pandas(df, covariate_columns=meta_cols)

    # Set condition as categorical with control as base
    data.obs[args.condition_col] = pd.Categorical(
        data.obs[args.condition_col], categories=[args.control, args.case]
    )

    # Covariates are forced to categorical. Donor and batch identifiers are very
    # often integers ("1", "2", "3"), and patsy reads a numeric column as a
    # continuous predictor -- which would silently fit a linear trend across donor
    # number instead of a per-donor intercept, absorbing none of the donor effect
    # the paired design exists to remove.
    for col in covariate_cols:
        data.obs[col] = pd.Categorical(data.obs[col].astype(str))

    # Build the patsy formula. Every term is wrapped in Q() because cell-type and
    # metadata column names in real datasets contain spaces, slashes and hyphens,
    # none of which are legal patsy identifiers. Covariates additionally get C() so
    # the categorical treatment is stated in the formula rather than inferred from
    # the column's dtype.
    terms = [f'Q("{args.condition_col}")']
    terms += [f'C(Q("{col}"))' for col in covariate_cols]
    formula = " + ".join(terms)

    results = []
    diagnostics = []

    # Always run auto-reference fit
    frame, diag = _run_fit(data, args, "automatic", "auto", formula)
    results.append(frame)
    diagnostics.append(diag)

    # If reference_celltype is set and valid, run explicit-reference fit
    if args.reference_celltype and args.reference_celltype in cell_cols:
        # Reset seeds for second fit
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
        frame, diag = _run_fit(
            data, args, args.reference_celltype, args.reference_celltype, formula
        )
        results.append(frame)
        diagnostics.append(diag)

    # Concatenate all results
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(args.out_csv, index=False)

    if args.diagnostics_json:
        payload = {
            "formula": formula,
            "covariates": covariate_cols,
            "covariates_missing": missing,
            "fits": diagnostics,
        }
        with open(args.diagnostics_json, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    return 0


def _run_fit(
    data: object,
    args: argparse.Namespace,
    reference_cell_type: str,
    reference_label: str,
    formula: str,
) -> tuple[pd.DataFrame, dict]:
    """Run a single scCODA fit and extract results plus sampler diagnostics."""

    import numpy as np
    import pandas as pd
    from sccoda.util import comp_ana as mod

    # Build and fit model
    model = mod.CompositionalAnalysis(
        data, formula=formula, reference_cell_type=reference_cell_type
    )

    res = model.sample_hmc(
        num_results=args.num_iterations, num_burnin=max(500, args.num_iterations // 4)
    )

    # Set FDR threshold
    res.set_fdr(est_fdr=0.1)

    # Extract effect_df
    effect_df = res.effect_df

    # Keep only the rows for the condition term. With extra covariates in the
    # formula the effect table carries one row per (design column, cell type), so
    # without this filter a paired design would emit the donor contrasts as
    # though they were condition effects.
    #
    # Matched on the patsy term prefix, not by substring. A substring test would
    # also match a covariate whose name merely contains the condition column
    # ("condition" inside "condition_batch"), silently mixing a nuisance contrast
    # into the reported effects.
    term_prefix = f'Q("{args.condition_col}")'
    n_cell_types = len({idx[1] for idx in effect_df.index})
    rows = []
    for idx in effect_df.index:
        covariate, cell_type = idx
        if not str(covariate).startswith(term_prefix):
            continue
        row = effect_df.loc[idx]
        rows.append(
            {
                "cell_type": cell_type,
                "log2_fold_change": row["log2-fold change"],
                "inclusion_probability": row["Inclusion probability"],
                "credible_effect": row["Final Parameter"] != 0,
                "reference": reference_label,
            }
        )

    # The filter has to keep exactly one row per cell type. Keeping too few means
    # the condition term was not found and results are being dropped; too many
    # means a nuisance term leaked in and is about to be reported as a condition
    # effect. Both are wrong answers rather than crashes, so fail loudly with the
    # labels that were actually present.
    if len(rows) != n_cell_types:
        raise RuntimeError(
            f"condition term {term_prefix!r} matched {len(rows)} of an expected "
            f"{n_cell_types} effect rows; effect table covariates were "
            f"{sorted({str(idx[0]) for idx in effect_df.index})}"
        )

    frame = pd.DataFrame(rows)

    # Sampler diagnostics. These exist so a caller can distinguish the two very
    # different situations that both produce an empty result table: a healthy
    # chain that found nothing (a real null), and a chain that never explored the
    # posterior. scCODA prints the acceptance rate to stdout and then discards
    # it, so the number is captured here where it is still available.
    stats = getattr(res, "sampling_stats", {}) or {}
    specs = getattr(res, "model_specs", {}) or {}
    inclusion = (
        frame["inclusion_probability"].to_numpy(dtype=float) if not frame.empty else np.array([])
    )
    diagnostics = {
        "reference": reference_label,
        "acceptance_rate": _as_float(stats.get("acc_rate")),
        "duration_seconds": _as_float(stats.get("duration")),
        "chain_length": _as_float(stats.get("chain_length")),
        "num_burnin": _as_float(stats.get("num_burnin")),
        # The inclusion probability that set_fdr had to demand to hold the false
        # discovery rate; at 1.0 no threshold could satisfy it and nothing is
        # callable however the effects look.
        "fdr_threshold_probability": _as_float(specs.get("threshold_prob")),
        "n_tested": int(inclusion.size),
        "n_credible": int(frame["credible_effect"].sum()) if not frame.empty else 0,
        "max_inclusion_probability": _as_float(inclusion.max()) if inclusion.size else None,
        # Distance of the posterior from the spike-and-slab prior. Under a genuine
        # null every inclusion probability sits near 0.5 and this is near zero,
        # which looks identical to a chain that learned nothing -- the acceptance
        # rate above is what separates those two readings.
        "mean_prior_distance": (
            _as_float(np.abs(inclusion - 0.5).mean()) if inclusion.size else None
        ),
    }

    return frame, diagnostics


def _as_float(value: object) -> float | None:
    """Coerce a diagnostic to a JSON-serialisable float, or None if it will not go."""

    if value is None:
        return None
    try:
        import math

        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


if __name__ == "__main__":
    raise SystemExit(main())
