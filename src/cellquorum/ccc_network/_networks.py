"""Biology-free network construction + LR adapters for ccc_network.

No anndata, no optional deps, no I/O. The canonical LR edge schema
(source, target, ligand, receptor, weight, sample[, condition]) is the contract
between the adapter, the network builders, and the two methods.
"""

from __future__ import annotations

import pandas as pd

# Canonical LR edge columns in stable order.
CANONICAL_COLUMNS = ["source", "target", "ligand", "receptor", "weight", "sample"]


def liana_to_canonical(liana_res: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map spec #1's ``uns['liana_res']`` frame to the canonical LR schema.

    ``magnitude_rank`` is a rank in [0, 1] where LOWER is stronger, so the
    canonical weight is ``1 - magnitude_rank`` (higher = stronger), matching the
    ``inverse_fun`` convention used by the Tensor method.

    Rows missing any required source column produce an empty frame and a note.

    Returns
    -------
    (canonical_df, notes)
    """
    notes: list[str] = []
    required = [
        "sample",
        "source",
        "target",
        "ligand_complex",
        "receptor_complex",
        "magnitude_rank",
    ]
    missing = [c for c in required if c not in liana_res.columns]
    if missing:
        notes.append(f"liana_to_canonical: source frame missing columns {missing}; no edges built.")
        return pd.DataFrame(columns=CANONICAL_COLUMNS), notes

    df = liana_res.loc[:, required].copy()
    # Drop rows with any null in a required field (skip-not-crash on dirty input).
    n_before = len(df)
    df = df.dropna(subset=required)
    if len(df) < n_before:
        notes.append(f"liana_to_canonical: dropped {n_before - len(df)} rows with missing values.")

    canon = pd.DataFrame(
        {
            "source": df["source"].astype(str).to_numpy(),
            "target": df["target"].astype(str).to_numpy(),
            "ligand": df["ligand_complex"].astype(str).to_numpy(),
            "receptor": df["receptor_complex"].astype(str).to_numpy(),
            "weight": 1.0 - df["magnitude_rank"].astype(float).to_numpy(),
            "sample": df["sample"].astype(str).to_numpy(),
        }
    )
    return canon.loc[:, CANONICAL_COLUMNS], notes


__all__ = ["CANONICAL_COLUMNS", "liana_to_canonical"]
