from __future__ import annotations

import numpy as np
import pandas as pd

from cellquorum.ccc_network._networks import liana_to_canonical
from cellquorum.ccc_network.config import CCCNetworkConfig


def _liana_res() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": ["s1", "s1", "s2"],
            "source": ["A", "B", "A"],
            "target": ["B", "A", "B"],
            "ligand_complex": ["L1", "L2", "L1"],
            "receptor_complex": ["R1", "R2", "R1"],
            "magnitude_rank": [0.1, 0.4, 0.2],
        }
    )


def test_config_defaults():
    cfg = CCCNetworkConfig()
    assert cfg.enabled is True
    assert cfg.source_key == "liana_res"
    assert cfg.gci_max_edges == 200_000
    assert cfg.ricci_alpha == 0.5
    assert cfg.pagerank_alpha == 0.01
    assert cfg.seed == 42


def test_liana_to_canonical_maps_and_inverts_weight():
    canon, notes = liana_to_canonical(_liana_res())
    assert list(canon.columns) == ["source", "target", "ligand", "receptor", "weight", "sample"]
    assert len(canon) == 3
    # weight = 1 - magnitude_rank (higher = stronger)
    row = canon[(canon["source"] == "A") & (canon["ligand"] == "L1") & (canon["sample"] == "s1")]
    assert np.isclose(row["weight"].iloc[0], 0.9)


def test_liana_to_canonical_drops_missing_required_columns():
    bad = _liana_res().drop(columns=["magnitude_rank"])
    canon, notes = liana_to_canonical(bad)
    assert canon.empty
    assert any("magnitude_rank" in n for n in notes)
