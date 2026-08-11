"""Tests for the GRN (pySCENIC) stage configuration model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.grn.config import GrnConfig


def test_grn_config_defaults() -> None:
    cfg = GrnConfig()
    assert cfg.enabled is True
    assert cfg.method == "pyscenic"
    assert cfg.layer == "counts"
    assert cfg.group_by is None
    assert cfg.organism == "human"
    assert cfg.tfs_path is None
    assert cfg.motifs_path is None
    assert cfg.rankings_glob is None
    assert cfg.num_workers == 8
    assert cfg.max_cells == 20000
    assert cfg.min_cells_total == 200
    assert cfg.top_n == 5
    assert cfg.seed == 0
    assert cfg.env_name == "pyscenic_env"
    assert cfg.launcher == "micromamba"
    assert cfg.timeout_seconds == 7200


def test_grn_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        GrnConfig(nonsense_key=123)


def test_grn_config_accepts_db_paths() -> None:
    cfg = GrnConfig(
        tfs_path="/db/allTFs_hg38.txt",
        motifs_path="/db/motifs.tbl",
        rankings_glob="/db/*.feather",
        group_by="cell_type",
    )
    assert cfg.tfs_path == "/db/allTFs_hg38.txt"
    assert cfg.rankings_glob == "/db/*.feather"
    assert cfg.group_by == "cell_type"
