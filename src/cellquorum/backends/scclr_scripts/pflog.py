"""In-env scclr helper: PFlog1pPF normalization and sparse PCA.

Runs INSIDE the isolated scclr environment (invoked by ``ScclrBackend``), so it
may import scclr freely. Data crosses the process boundary as files:

normalize mode:
    in:  <counts.npz>         (scipy sparse CSR, cells x genes, raw counts)
    args: --target <auto|mean|median|FLOAT>
    out: <pflog.npz>          (sparse PFlog values, cells x genes)
         <meta.json>          ({"row_center": [...], "k": float, "alpha": float|null})

pca mode:
    in:  <pflog.npz> <center.npy>   (sparse PFlog + per-cell row_center)
    args: --n-components N [--seed S]
    out: <pca.npz>            (scores, components, explained_variance,
                               explained_variance_ratio, singular_values,
                               plus scalars in <pca_meta.json>)

Exit code 0 on success; non-zero with a message on stderr otherwise (the caller
inspects the return code and raises a domain-specific error).
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_normalize(args: argparse.Namespace) -> int:
    import numpy as np
    import scclr
    import scipy.sparse as sp

    X = sp.load_npz(args.counts)

    # Resolve target: numeric string -> fixed K, else the named strategy.
    target: object = args.target
    try:
        target = float(args.target)
    except (TypeError, ValueError):
        target = args.target

    sclr = scclr.normalize(X, target=target)

    sparse = sclr.sparse.tocsr()
    sp.save_npz(args.out_matrix, sparse)
    meta = {
        "row_center": np.asarray(sclr.row_center, dtype=float).tolist(),
        "k": None if sclr.k is None else float(sclr.k),
        "alpha": None if sclr.alpha is None else float(sclr.alpha),
        "shape": [int(sparse.shape[0]), int(sparse.shape[1])],
    }
    with open(args.out_meta, "w") as handle:
        json.dump(meta, handle)
    return 0


def _cmd_pca(args: argparse.Namespace) -> int:
    import numpy as np
    import scclr
    import scipy.sparse as sp

    sparse = sp.load_npz(args.matrix)
    row_center = np.load(args.center)

    # Rebuild the ShiftedCLR container so scclr runs the implicit-centered path.
    sclr = scclr.ShiftedCLR(sparse=sparse, row_center=row_center)
    res = scclr.pca(sclr, n_components=args.n_components, seed=args.seed)

    np.savez(
        args.out,
        scores=np.asarray(res.scores, dtype=float),
        components=np.asarray(res.components, dtype=float),
        explained_variance=np.asarray(res.explained_variance, dtype=float),
        explained_variance_ratio=np.asarray(res.explained_variance_ratio, dtype=float),
        singular_values=np.asarray(res.singular_values, dtype=float),
    )
    meta = {
        "n_components": int(res.n_components),
        "n_samples": int(res.n_samples),
        "n_features": int(res.n_features),
    }
    with open(args.out_meta, "w") as handle:
        json.dump(meta, handle)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested mode."""

    parser = argparse.ArgumentParser(description="scclr in-env PFlog helper.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_norm = sub.add_parser("normalize", help="PFlog1pPF normalization.")
    p_norm.add_argument("counts")
    p_norm.add_argument("out_matrix")
    p_norm.add_argument("out_meta")
    p_norm.add_argument("--target", default="auto")
    p_norm.set_defaults(func=_cmd_normalize)

    p_pca = sub.add_parser("pca", help="Sparse implicit-centered PCA.")
    p_pca.add_argument("matrix")
    p_pca.add_argument("center")
    p_pca.add_argument("out")
    p_pca.add_argument("out_meta")
    p_pca.add_argument("--n-components", dest="n_components", type=int, default=50)
    p_pca.add_argument("--seed", type=int, default=0)
    p_pca.set_defaults(func=_cmd_pca)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # surface a clean message to the caller's stderr
        print(f"scclr helper failed ({args.mode}): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
