"""input.layer_kinds: declare what the layers of an external, untagged object are."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.contracts.layer_tags import get_layer_tag, set_layer_tag
from cellquorum.core.pipeline import load_input_adata_from_config
from cellquorum.io.anndata import AnnDataLoadError


def _write(tmp_path: Path, *, tagged: str | None = None) -> Path:
    a = ad.AnnData(X=np.ones((6, 3), dtype="float32"))
    a.layers["counts"] = np.arange(18, dtype="float32").reshape(6, 3)
    a.obs["cell_type"] = ["A", "A", "A", "B", "B", "B"]
    if tagged:
        set_layer_tag(a, "counts", kind=tagged)
    path = tmp_path / "in.h5ad"
    a.write_h5ad(path)
    return path


def _config(path: Path, **input_extra):
    return CellQuorumConfig(input={"h5ad": str(path), **input_extra})


def test_by_default_no_layer_is_tagged(tmp_path):
    a = load_input_adata_from_config(_config(_write(tmp_path)))

    assert get_layer_tag(a, "counts") is None


def test_a_declared_layer_is_tagged_with_its_kind(tmp_path):
    config = _config(_write(tmp_path), layer_kinds={"counts": "counts"})

    a = load_input_adata_from_config(config)

    assert get_layer_tag(a, "counts") == {"kind": "counts", "recipe": None}


def test_a_declared_layer_is_tagged_after_a_subset_too(tmp_path):
    config = _config(
        _write(tmp_path),
        layer_kinds={"counts": "counts"},
        subset={"column": "cell_type", "values": ["A"]},
    )

    a = load_input_adata_from_config(config)

    assert a.n_obs == 3
    assert get_layer_tag(a, "counts")["kind"] == "counts"


def test_declaring_a_layer_that_is_not_in_the_file_is_an_error(tmp_path):
    config = _config(_write(tmp_path), layer_kinds={"nope": "counts"})

    with pytest.raises(AnnDataLoadError, match="nope"):
        load_input_adata_from_config(config)


def test_a_declaration_that_contradicts_an_existing_tag_is_an_error(tmp_path):
    config = _config(_write(tmp_path, tagged="lognorm"), layer_kinds={"counts": "counts"})

    with pytest.raises(AnnDataLoadError, match="lognorm"):
        load_input_adata_from_config(config)


def test_repeating_an_existing_tag_is_fine(tmp_path):
    config = _config(_write(tmp_path, tagged="counts"), layer_kinds={"counts": "counts"})

    a = load_input_adata_from_config(config)

    assert get_layer_tag(a, "counts")["kind"] == "counts"


def test_an_unknown_kind_is_rejected_when_the_config_is_read(tmp_path):
    with pytest.raises(ValueError, match="raw"):
        _config(_write(tmp_path), layer_kinds={"counts": "raw"})
