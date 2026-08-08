"""In-env scCODA helper: Bayesian compositional differential abundance.

Runs INSIDE the isolated sccoda_env environment (invoked by ``SccodaBackend``), so
it may import sccoda freely. Data crosses the process boundary as files:

INPUT:
    <counts_meta.csv>: samples × cell-types integer counts PLUS one extra column
                      named by <condition_col> arg (values = case/control labels).

ARGS:
    <counts_meta_csv> <out_csv> <condition_col> <case> <control> <seed>
    <num_iterations> [reference_celltype]

OUTPUT:
    <out_csv>: rows = cell types (optionally two sets if reference_celltype given),
               columns = cell_type, log2_fold_change, inclusion_probability,
               credible_effect, reference

Exit code 0 on success; non-zero with a message on stderr otherwise (the caller
inspects the return code and raises a domain-specific error).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

# CRITICAL: Set TF_DETERMINISTIC_OPS BEFORE importing tensorflow/sccoda
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

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

    # Set seeds for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Read input CSV
    df = pd.read_csv(args.counts_meta_csv, index_col=0)

    # Separate cell type columns from condition column
    cell_cols = [c for c in df.columns if c != args.condition_col]

    # Build scCODA data object
    data = ccd.from_pandas(df, covariate_columns=[args.condition_col])

    # Set condition as categorical with control as base
    data.obs[args.condition_col] = pd.Categorical(
        data.obs[args.condition_col], categories=[args.control, args.case]
    )

    results = []

    # Always run auto-reference fit
    results.append(_run_fit(data, args, "automatic", "auto"))

    # If reference_celltype is set and valid, run explicit-reference fit
    if args.reference_celltype and args.reference_celltype in cell_cols:
        # Reset seeds for second fit
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
        results.append(_run_fit(data, args, args.reference_celltype, args.reference_celltype))

    # Concatenate all results
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(args.out_csv, index=False)

    return 0


def _run_fit(
    data: object, args: argparse.Namespace, reference_cell_type: str, reference_label: str
) -> pd.DataFrame:
    """Run a single scCODA fit and extract results."""

    import pandas as pd
    from sccoda.util import comp_ana as mod

    # Build and fit model
    model = mod.CompositionalAnalysis(
        data, formula=args.condition_col, reference_cell_type=reference_cell_type
    )

    res = model.sample_hmc(
        num_results=args.num_iterations, num_burnin=max(500, args.num_iterations // 4)
    )

    # Set FDR threshold
    res.set_fdr(est_fdr=0.1)

    # Extract effect_df
    effect_df = res.effect_df

    # Build result rows
    rows = []
    for idx in effect_df.index:
        covariate, cell_type = idx
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

    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
