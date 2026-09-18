#!/usr/bin/env python
"""CellQuorum in-env pyucell scoring backend.

Runs pyucell's UCell algorithm (per-cell gene rank percentiles, no
recovery-curve AUC -- built to stay stable on small/sparse gene signatures
where decoupler's AUCell goes noisy) inside the isolated env that has it
installed. Gene-presence filtering happens in the calling method (main
CellQuorum env, has var_names already); this script just scores whatever
signatures it's handed.

Two failure regimes, matching the CellOracle in-env script's contract:
  - NOT CONFIGURED (pyucell absent, nothing scored): writes an empty
    ucell_scores.csv + SKIPPED_{tag}.txt marker and exits 0 -> harmless skip.
  - REAL FAILURE (pyucell present but compute_ucell_scores itself raises):
    writes FAILED_{tag}.txt and exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="pyucell per-cell signature scoring")
    p.add_argument("--h5ad", required=True)
    p.add_argument("--programs-json", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--tag", default="ucell")
    p.add_argument("--max-rank", type=int, default=1500)
    p.add_argument("--layer", default="")
    return p


def write_skip(out_csv: Path, tag: str, reason: str) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("cell\n", encoding="utf-8")
    (out_csv.parent / f"SKIPPED_{tag}.txt").write_text(
        f"pyucell skipped (isolated backend, no downstream effect): {reason}\n",
        encoding="utf-8",
    )
    print(f"[pyucell] SKIPPED gracefully: {reason}")


def write_failure(out_csv: Path, tag: str, reason: str) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    (out_csv.parent / f"FAILED_{tag}.txt").write_text(
        f"pyucell failed: {reason}\n", encoding="utf-8"
    )
    print(f"[pyucell] FAILED: {reason}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_csv = Path(args.out_csv)

    try:
        import anndata as ad
    except Exception as exc:  # noqa: BLE001
        write_skip(out_csv, args.tag, f"anndata unavailable: {exc}")
        return 0

    try:
        import pyucell
    except Exception as exc:  # noqa: BLE001
        write_skip(out_csv, args.tag, f"pyucell unavailable: {exc}")
        return 0

    programs_path = Path(args.programs_json)
    if not programs_path.is_file():
        write_skip(out_csv, args.tag, f"programs file not found: {programs_path}")
        return 0
    programs = json.loads(programs_path.read_text(encoding="utf-8"))
    if not programs:
        write_skip(out_csv, args.tag, "no programs given")
        return 0

    h5ad_path = Path(args.h5ad)
    if not h5ad_path.is_file():
        write_skip(out_csv, args.tag, f"h5ad not found: {h5ad_path}")
        return 0

    try:
        adata = ad.read_h5ad(h5ad_path)
        layer = args.layer or None
        pyucell.compute_ucell_scores(
            adata,
            signatures=programs,
            layer=layer,
            max_rank=int(args.max_rank),
            suffix="",
        )
    except Exception as exc:  # noqa: BLE001
        write_failure(out_csv, args.tag, str(exc))
        return 1

    score_cols = [name for name in programs if name in adata.obs.columns]
    if not score_cols:
        write_skip(out_csv, args.tag, "pyucell produced no signature columns")
        return 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scores = adata.obs[score_cols].copy()
    scores.insert(0, "cell", adata.obs_names.to_numpy())
    scores.to_csv(out_csv, index=False)
    print(f"[pyucell] wrote {len(score_cols)} signature score(s) -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
