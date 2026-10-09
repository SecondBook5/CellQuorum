"""Native scTenifoldKnk runner. No CellQuorum imports in the isolated process."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Read a genes-by-cells matrix, run the actual package, and export its table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from scTenifold import scTenifoldKnk

    cfg = json.loads(Path(args.config).read_text())
    x = pd.read_csv(args.input, index_col=0)
    if not x.index.is_unique or not x.columns.is_unique:
        raise ValueError("Input gene and cell identifiers must be unique")
    if not np.isfinite(x.to_numpy()).all() or (x.to_numpy() < 0).any():
        raise ValueError("Input must contain finite nonnegative counts")
    absent = set(cfg["ko_genes"]) - set(x.index)
    if absent:
        raise ValueError(f"Knockout genes absent: {sorted(absent)}")
    model = scTenifoldKnk(
        x,
        ko_genes=cfg["ko_genes"],
        ko_method=cfg["ko_method"],
        qc_kws={
            "min_lib_size": 0,
            "remove_outlier_cells": False,
            "max_mito_ratio": 1.0,
            "min_percent": 0,
            "min_exp_avg": 0,
            "min_exp_sum": 0,
            "plot": False,
        },
        nc_kws={
            "n_nets": cfg["n_nets"],
            "n_samp_cells": min(cfg["n_samp_cells"], x.shape[1]),
            "n_comp": min(cfg["n_comp"], x.shape[1] - 1, x.shape[0] - 1),
            "q": cfg["network_quantile"],
            "backend": "serial",
            "n_jobs": 1,
            "random_state": cfg["seed"],
        },
        td_kws={
            "K": cfg["tensor_rank"],
            "max_iter": cfg["tensor_max_iter"],
            "init": "random",
            "random_state": cfg["seed"],
        },
        ma_kws={"d": min(cfg["manifold_dim"], x.shape[0] - 2)},
        ko_kws={"degree": cfg["propagation_degree"]} if cfg["ko_method"] == "propagation" else None,
    )
    result = model.build()
    required = {"Gene", "Distance", "p-value", "adjusted p-value"}
    if result.empty or not required.issubset(result.columns):
        raise RuntimeError("Native scTenifoldKnk returned an empty or unsupported result schema")
    if not set(cfg["ko_genes"]).issubset(set(result.Gene.astype(str))):
        raise RuntimeError("Knockout gene was removed inside the native package")
    result.to_csv(args.output, index=False)
    wt = model.tensor_dict["WT"]
    outgoing = []
    for gene in cfg["ko_genes"]:
        for target, coefficient in wt.loc[gene].items():
            if target != gene and abs(float(coefficient)) > 1e-12:
                outgoing.append(
                    {"source": gene, "target": str(target), "weight": float(coefficient)}
                )
    pd.DataFrame(outgoing, columns=["source", "target", "weight"]).to_csv(
        Path(args.output).with_suffix(".outgoing_edges.csv"), index=False
    )
    Path(args.output).with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "package": "scTenifoldpy",
                "version": version("scTenifoldpy"),
                "n_input_cells": x.shape[1],
                "n_input_genes": x.shape[0],
                "n_fitted_cells": model.QC_dict["WT"].shape[1],
                "n_fitted_genes": model.QC_dict["WT"].shape[0],
                "fitted_cell_ids": model.QC_dict["WT"].columns.astype(str).tolist(),
                "config": cfg,
                "n_knockout_outgoing_edges": len(outgoing),
                "interpretation": (
                    "Unsigned differential network regulation, not expression fold change"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
