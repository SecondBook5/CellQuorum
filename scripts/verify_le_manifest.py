"""Verify every manifest row points to a CellRanger outs dir with raw+filtered h5.

Usage:
    python scripts/verify_le_manifest.py <manifest.csv> <cellranger_root>
Exits non-zero and prints the offending rows if any library is missing a matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main(manifest_path: str, root: str) -> int:
    df = pd.read_csv(manifest_path, dtype=str)
    root_path = Path(root)
    problems: list[str] = []
    for row in df.itertuples(index=False):
        outs = root_path / row.cellranger_path / "outs"
        raw = outs / "raw_feature_bc_matrix.h5"
        filt = outs / "filtered_feature_bc_matrix.h5"
        missing = [p.name for p in (raw, filt) if not p.is_file()]
        if missing:
            problems.append(f"{row.sample_id}: {outs} missing {missing}")
    if problems:
        print("MANIFEST VERIFICATION FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"OK: all {len(df)} libraries have raw+filtered matrices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
