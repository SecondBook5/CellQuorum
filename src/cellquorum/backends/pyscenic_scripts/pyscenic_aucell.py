#!/usr/bin/env python
"""CellQuorum in-env pySCENIC AUCell — per-cell regulon activity matrix.

Runs the AUCell step: score each cell for each regulon's target-gene enrichment.
Converts loom + regulons (from the GRN step) to per-cell AUC matrix and exports
as parquet (index = CellID, columns = regulon names).

Graceful: missing loom or regulons -> writes an empty sentinel at --out + a
<stem>_SKIPPED.txt marker, exit 0 (isolated backend, no downstream effect).
A real aucell CLI failure fails loud (exit non-zero, <stem>_FAILED.txt written).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _write_skip(out_path: Path, reason: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")  # empty sentinel
    (out_path.parent / f"{out_path.stem}_SKIPPED.txt").write_text(
        f"aucell skipped (isolated backend, no downstream effect): {reason}\n"
    )
    print(f"[aucell] SKIPPED gracefully: {reason}")


def main() -> None:
    p = argparse.ArgumentParser(description="pySCENIC AUCell: per-cell regulon activity matrix")
    p.add_argument("--loom", required=True, help="expression loom written by scenic_grn (grn step)")
    p.add_argument(
        "--regulons", required=True, help="ctx regulons CSV (scenic_regulons_{edge}.csv)"
    )
    p.add_argument("--out", required=True, help="output auc_mtx parquet (cells x regulons)")
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    out_path = Path(args.out)

    # graceful bail-outs -------------------------------------------------------
    for name, path in [("loom", args.loom), ("regulons", args.regulons)]:
        if not Path(path).exists() or Path(path).stat().st_size == 0:
            _write_skip(out_path, f"{name} missing/empty ({path!r})")
            sys.exit(0)
    try:
        import loompy  # noqa: F401
        import pandas as pd
    except Exception as e:
        _write_skip(out_path, f"pyscenic env import failed: {type(e).__name__}: {e}")
        sys.exit(0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    auc_csv = out_path.with_suffix(".csv")

    # aucell CLI: loom + regulons -> per-cell AUC (cells x regulons) ------------
    cmd = [
        "pyscenic",
        "aucell",
        args.loom,
        args.regulons,
        "--output",
        str(auc_csv),
        "--num_workers",
        str(args.num_workers),
    ]
    print(f"[aucell] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        (out_path.parent / f"{out_path.stem}_FAILED.txt").write_text(
            f"pyscenic aucell failed (exit {e.returncode}): {' '.join(cmd)}\n"
        )
        print(f"[aucell] FAILED: exit {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode or 1)

    # aucell writes a CSV (cells x regulons). Convert to parquet for the figure step.
    auc = pd.read_csv(auc_csv, index_col=0)
    auc.to_parquet(out_path)
    print(f"[aucell] wrote {out_path}  ({auc.shape[0]} cells x {auc.shape[1]} regulons)")


if __name__ == "__main__":
    main()
