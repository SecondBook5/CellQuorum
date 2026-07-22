"""Print the label vocabulary a CellTypist model emits on a query subsample.

Usage:
    python scripts/inspect_celltypist_labels.py <one_query_filtered_h5> <model_name_or_path>
"""

from __future__ import annotations

import sys

import scanpy as sc


def main(query_h5: str, model: str) -> int:
    import celltypist

    a = sc.read_10x_h5(query_h5)
    a.var_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    pred = celltypist.annotate(a, model=model, majority_voting=False)
    labels = pred.predicted_labels
    col = labels.columns[0]
    print("CellTypist labels emitted:")
    for lab in sorted(map(str, labels[col].unique())):
        print("  -", lab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
