"""In-env partipy helper: archetypal analysis on a cell embedding.

Runs INSIDE the isolated partipy environment (invoked by ``PartipyBackend``), so it may import
partipy freely. CellQuorum itself never imports it — partipy is GPL-3 and CellQuorum is BSD-3,
so the process boundary is a licensing boundary as much as a dependency one.

Data crosses as files:

    in:   <embedding.npy>     (cells x components, float)
    args: --n-archetypes-min N --n-archetypes-max M [--seed S] [--bootstrap B]
          [--permutations P]
    out:  <weights.npy>       (cells x archetypes, each row summing to 1)
          <meta.json>         ({"n_archetypes": int, "t_ratio": float,
                                "t_ratio_pvalue": float|null,
                                "selection": [{"n": int, "rss": float, "varexpl": float}, ...],
                                "bootstrap_variance": [float, ...]|null})

Exit code 0 on success; non-zero with a message on stderr otherwise (the caller inspects the
return code and raises a domain-specific error).
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_fit(args: argparse.Namespace) -> int:
    import numpy as np

    embedding = np.load(args.embedding)
    if embedding.ndim != 2 or embedding.shape[0] < 10:
        print(f"embedding must be 2D with >=10 cells, got {embedding.shape}", file=sys.stderr)
        return 2

    import anndata as ad
    import pandas as pd
    import partipy as pt

    adata = ad.AnnData(
        X=np.zeros((embedding.shape[0], 1), dtype="float32"),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(embedding.shape[0])]),
    )
    # The embedding must exist before set_obsm selects it: set_obsm names which obsm key and
    # how many of its dimensions partipy should fit on, it does not create the key.
    adata.obsm["X_input"] = np.ascontiguousarray(embedding, dtype="float64")
    pt.set_obsm(adata, obsm_key="X_input", n_dimensions=int(embedding.shape[1]))

    # How many distinct extreme phenotypes the data supports. partipy reports an information
    # criterion across candidate counts, so the choice is a minimum rather than an eyeballed
    # elbow — IC already trades variance explained against the cost of another vertex.
    lo = max(2, int(args.n_archetypes_min))
    hi = int(min(args.n_archetypes_max, embedding.shape[0] - 1))
    candidates = list(range(lo, max(lo, hi) + 1))
    # One restart per candidate during the sweep, and single-threaded. The sweep is
    # ``len(candidates)`` fits and dominates the cost; the default of 5 restarts turns an
    # 8-candidate sweep into 40 fits. Nested joblib inside a micromamba subprocess also
    # deadlocked at cohort scale, so parallelism is opt-in rather than default.
    pt.compute_selection_metrics(
        adata,
        n_archetypes_list=candidates,
        n_restarts=int(args.n_restarts),
        n_jobs=int(args.n_jobs),
        seed=int(args.seed),
    )
    metrics = pt.summarize_aa_metrics(adata)

    chosen = _select_n_archetypes(metrics, lo)

    # compute_selection_metrics leaves one stored result per candidate count, so every call
    # after it must name which result it means or partipy refuses with "multiple AA results".
    pinned = {"n_archetypes": chosen}
    pt.compute_archetypes(
        adata,
        n_archetypes=chosen,
        n_restarts=int(args.n_restarts),
        n_jobs=int(args.n_jobs),
        seed=int(args.seed),
    )
    pt.compute_archetype_weights(adata, result_filters=pinned)

    # These are distance-kernel memberships and do NOT sum to one per cell, so they are
    # normalised here into a share of each cell's weight. The caller thresholds that share to
    # decide which cells genuinely sit at a vertex, which only means something on a
    # per-cell total of one.
    weights = np.asarray(pt.get_aa_cell_weights(adata, n_archetypes=chosen), dtype="float64")
    totals = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
    np.save(args.out_weights, weights)

    meta: dict[str, object] = {
        "n_archetypes": int(chosen),
        "selection": _serialize_metrics(metrics),
        "t_ratio": None,
        "t_ratio_pvalue": None,
        "bootstrap_variance": None,
    }

    # The t-ratio and its permutation p-value say whether a polytope describes the data better
    # than noise does. A polytope can always be fitted, so without this "we found archetypes"
    # is not a finding.
    try:
        meta["t_ratio"] = float(
            pt.compute_t_ratio(adata, result_filters=pinned, return_result=True)
        )
        meta["t_ratio_pvalue"] = _as_pvalue(
            pt.t_ratio_significance(
                adata,
                n_iter=int(args.permutations),
                seed=int(args.seed),
                result_filters=pinned,
            )
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics are optional, the fit is not
        print(f"t-ratio unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)

    # Bootstrap variance says whether each vertex is stably located. An unstable vertex must
    # not be reported to a user as a population.
    if int(args.bootstrap) > 0:
        try:
            pt.compute_bootstrap_variance(
                adata, n_bootstrap=int(args.bootstrap), seed=int(args.seed)
            )
            meta["bootstrap_variance"] = _flatten(pt.get_aa_bootstrap(adata, n_archetypes=chosen))
        except Exception as exc:  # noqa: BLE001
            print(f"bootstrap unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)

    with open(args.out_meta, "w") as handle:
        json.dump(meta, handle)
    return 0


def _select_n_archetypes(metrics: object, floor: int) -> int:
    """Archetype count minimising partipy's information criterion.

    IC already balances variance explained against the cost of an extra vertex, so a minimum
    is the principled choice. Falls back to ``floor`` if the column is absent in a future
    partipy, because a wrong count is recoverable and a crash mid-run is not.
    """
    try:
        frame = metrics.reset_index()
        count_column = next(name for name in ("n_archetypes", "k") if name in frame.columns)
        if "IC" in frame.columns and frame["IC"].notna().any():
            return int(frame.loc[frame["IC"].idxmin(), count_column])
        return int(frame[count_column].min())
    except Exception:  # noqa: BLE001 - selection is heuristic; the floor is always valid
        return floor


def _serialize_metrics(metrics: object) -> list[dict[str, float]]:
    """Selection metrics as plain records, whatever shape partipy returned."""
    try:
        frame = metrics.reset_index()
        return [
            {str(key): _as_float(value) for key, value in record.items()}
            for record in frame.to_dict(orient="records")
        ]
    except Exception:  # noqa: BLE001
        return []


def _as_float(value: object) -> float:
    """Best-effort float, NaN when not convertible."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _as_pvalue(significance: object) -> float | None:
    """Extract a p-value from whatever partipy's significance call returned."""
    if isinstance(significance, int | float):
        return float(significance)
    if isinstance(significance, dict):
        for key in ("t_ratio_p_value", "pvalue", "p_value", "p"):
            if key in significance:
                return _as_float(significance[key])
    return None


def _flatten(values: object) -> list[float]:
    """Flatten an array or frame of per-archetype variances."""
    import numpy as np

    array = np.asarray(getattr(values, "to_numpy", lambda: values)())
    return [float(value) for value in array.ravel()]


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested mode."""
    parser = argparse.ArgumentParser(description="partipy in-env archetype helper.")
    sub = parser.add_subparsers(dest="mode", required=True)

    fit = sub.add_parser("fit", help="Fit archetypes to an embedding.")
    fit.add_argument("embedding")
    fit.add_argument("out_weights")
    fit.add_argument("out_meta")
    fit.add_argument("--n-archetypes-min", dest="n_archetypes_min", type=int, default=3)
    fit.add_argument("--n-archetypes-max", dest="n_archetypes_max", type=int, default=10)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--bootstrap", type=int, default=0)
    fit.add_argument("--permutations", type=int, default=100)
    fit.add_argument("--n-restarts", dest="n_restarts", type=int, default=1)
    fit.add_argument("--n-jobs", dest="n_jobs", type=int, default=1)
    fit.set_defaults(func=_cmd_fit)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # surface a clean message to the caller's stderr
        print(
            f"partipy helper failed ({args.mode}): {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
