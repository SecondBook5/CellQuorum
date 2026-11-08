"""Case-minus-control density figure: the numbers, the refusals and the written files.

Real scipy KDE and real matplotlib rendering; the only synthetic thing is the cell cloud.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.integration.embeddings.density_difference import (
    DensityDifferenceError,
    density_difference,
)
from cellquorum.stages.integration.embeddings.density_difference_umap_method import (
    DensityDifferenceUmapMethod,
)

_GRID = 40


def _cloud(counts: dict[str, int], arms: dict[str, str], seed: int = 0):
    """Two blobs: the control specimens sit on the left one, the case specimens on the right."""
    rng = np.random.default_rng(seed)
    xy, specimen, condition = [], [], []
    for name, n in counts.items():
        centre = np.array([-3.0, 0.0]) if arms[name] == "control" else np.array([3.0, 0.0])
        xy.append(rng.normal(centre, 0.7, size=(n, 2)))
        specimen += [name] * n
        condition += [arms[name]] * n
    return np.vstack(xy), np.array(specimen), np.array(condition)


_ARMS = {"c1": "control", "c2": "control", "k1": "case", "k2": "case"}


def _difference(counts, **kwargs):
    xy, specimen, condition = _cloud(counts, _ARMS)
    return density_difference(
        xy, specimen, condition, case="case", control="control", grid_size=_GRID, **kwargs
    )


def test_case_region_is_positive_and_control_region_negative():
    result = _difference({"c1": 300, "c2": 300, "k1": 300, "k2": 300})
    x = np.linspace(*result.x_edges, _GRID)
    left, right = result.difference[:, x < 0], result.difference[:, x > 0]
    assert left.sum() < 0 < right.sum()
    assert result.case_specimens == ("k1", "k2")
    assert result.control_specimens == ("c1", "c2")


def test_each_specimen_weighs_the_same_whatever_its_cell_count():
    even = _difference({"c1": 300, "c2": 300, "k1": 300, "k2": 300})
    # One case specimen with ten times the cells: a pooled-cell density would let it dominate.
    lopsided = _difference({"c1": 300, "c2": 300, "k1": 3000, "k2": 300})
    np.testing.assert_allclose(even.limit, lopsided.limit, rtol=0.25)
    # The arms still carry equal total mass, so the two sides cancel exactly.
    assert abs(lopsided.difference.sum()) < 1e-9


def test_a_specimen_mixing_conditions_is_refused():
    xy, specimen, condition = _cloud({"c1": 100, "k1": 100}, _ARMS)
    condition[:20] = "case"  # c1 now holds both conditions
    with pytest.raises(DensityDifferenceError, match="mixes"):
        density_difference(xy, specimen, condition, case="case", control="control")


def test_a_specimen_too_small_is_dropped_and_reported():
    result = _difference({"c1": 300, "c2": 20, "k1": 300, "k2": 300})
    assert result.dropped == (("c2", 20),)
    assert result.control_specimens == ("c1",)


def test_an_arm_with_no_usable_specimen_is_refused():
    with pytest.raises(DensityDifferenceError, match="control"):
        _difference({"c1": 20, "c2": 20, "k1": 300, "k2": 300})


def _adata():
    counts = {"c1": 400, "c2": 400, "k1": 400, "k2": 400}
    xy, specimen, condition = _cloud(counts, _ARMS)
    a = ad.AnnData(X=np.zeros((len(xy), 2), dtype="float32"))
    a.obsm["X_umap"] = xy
    a.obs["sample"] = specimen
    a.obs["genotype"] = condition
    a.obs["tissue"] = ["gut" if i % 2 else "skin" for i in range(len(xy))]
    a.obs["state"] = np.where(xy[:, 0] < 0, "left_state", "right_state")
    return a


def _context(tmp_path: Path, design=None):
    paths = SimpleNamespace(figures=tmp_path / "figs", root=tmp_path)
    return SimpleNamespace(paths=paths, config=SimpleNamespace(design=design), random_seed=0)


def test_method_writes_one_file_per_stratum_with_labels(tmp_path):
    result = DensityDifferenceUmapMethod()._run(
        _adata(),
        {
            "sample_col": "sample",
            "condition_col": "genotype",
            "case": "case",
            "control": "control",
            "stratify_by": "tissue",
            "label_column": "state",
            "grid_size": 40,
            "figure_formats": ["png"],
            "dpi": 60,
        },
        _context(tmp_path),
    )
    assert not isinstance(result, MethodSkip)
    names = sorted(a.path.name for a in result.artifacts)
    assert names == ["density_difference_umap_gut.png", "density_difference_umap_skin.png"]
    assert all(
        a.path.parent.name == "embeddings" and a.path.stat().st_size > 0 for a in result.artifacts
    )
    assert result.metrics["strata"] == ["gut", "skin"]


def test_method_takes_case_control_and_specimen_from_the_design(tmp_path):
    design = SimpleNamespace(
        condition_col="genotype",
        case="case",
        control="control",
        donor_col="sample",
        factorial=None,
    )
    result = DensityDifferenceUmapMethod()._run(
        _adata(),
        {"grid_size": 40, "figure_formats": ["png"], "dpi": 60},
        _context(tmp_path, design),
    )
    assert not isinstance(result, MethodSkip)
    assert [a.path.name for a in result.artifacts] == ["density_difference_umap.png"]


def test_method_skips_with_a_reason_when_case_and_control_are_unknown(tmp_path):
    result = DensityDifferenceUmapMethod()._run(
        _adata(), {"sample_col": "sample"}, _context(tmp_path)
    )
    assert isinstance(result, MethodSkip)
    assert "case and control" in result.reason


def test_method_skips_when_a_named_column_is_missing(tmp_path):
    result = DensityDifferenceUmapMethod()._run(
        _adata(),
        {"sample_col": "nope", "condition_col": "genotype", "case": "case", "control": "control"},
        _context(tmp_path),
    )
    assert isinstance(result, MethodSkip)
    assert "nope" in result.reason
