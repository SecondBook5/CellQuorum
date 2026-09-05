"""An over-strict floor must fail at the floor, not five stages downstream.

``QCConfig.fail_on_empty_result`` defaulted to True and was read by nothing. The consequence,
found by running the generalization smoke test: a 50-gene matrix met the default 200-gene floor,
every barcode was removed, QC reported success, and the run continued until a reduction in a
later stage raised ``zero-size array to reduction operation minimum which has no identity``.

That is the least useful place to learn the floor was wrong, and it is the mistake a first-time
user makes most often — the 200-gene default assumes a filtered whole-transcriptome matrix, and
silently empties a targeted panel, a subsample, or an aggregated object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.qc.floors import QCFloorError, apply_floors, require_non_empty_qc_result


def _floors(*, n_cells: int, n_genes: int, min_genes: int | None, min_cells_per_gene: int | None):
    """Apply floors to a dense matrix where every cell detects every gene."""
    matrix = np.full((n_cells, n_genes), 3.0, dtype=np.float32)
    return apply_floors(
        matrix,
        pd.Index([f"cell_{i}" for i in range(n_cells)]),
        pd.Index([f"G{i}" for i in range(n_genes)]),
        min_genes_per_cell=min_genes,
        min_cells_per_gene=min_cells_per_gene,
    )


def test_a_floor_that_empties_the_object_fails_and_names_the_cause() -> None:
    """The exact reproduction: a 50-gene matrix against the 200-gene default."""

    floors = _floors(n_cells=80, n_genes=50, min_genes=200, min_cells_per_gene=1)
    assert int(floors.cell_keep.sum()) == 0, "fixture must empty the object"

    with pytest.raises(QCFloorError) as raised:
        require_non_empty_qc_result(floors, n_genes=50)

    message = str(raised.value)
    # The count, the attributed reason, and the way out all have to be in the message: the
    # failure is a configuration mistake, so the error is the whole diagnosis.
    assert "80" in message
    assert "fewer_than_200_genes" in message
    assert "min_genes_per_cell" in message
    assert "fail_on_empty_result" in message


def test_an_emptied_gene_axis_fails_too() -> None:
    """Removing every gene is as fatal as removing every cell, and less obvious."""

    floors = _floors(n_cells=4, n_genes=20, min_genes=1, min_cells_per_gene=100)
    assert int(floors.gene_keep.sum()) == 0

    with pytest.raises(QCFloorError, match="min_cells_per_gene"):
        require_non_empty_qc_result(floors, n_genes=0)


def test_a_surviving_object_passes() -> None:
    """The control. A guard that fires on a healthy run is worse than no guard."""

    floors = _floors(n_cells=80, n_genes=500, min_genes=200, min_cells_per_gene=3)
    assert int(floors.cell_keep.sum()) == 80
    require_non_empty_qc_result(floors, n_genes=500)


def test_one_surviving_cell_is_not_empty() -> None:
    """The boundary. One cell is a bad run, not an impossible one, and the guard is for zero.

    Reporting "almost everything was removed" is the job of the >50% warning that
    ``apply_floors`` already raises; this guard exists only for the case where nothing at all
    can be computed downstream.
    """

    matrix = np.zeros((10, 300), dtype=np.float32)
    matrix[0, :] = 3.0  # one cell detects everything, nine detect nothing
    floors = apply_floors(
        matrix,
        pd.Index([f"cell_{i}" for i in range(10)]),
        pd.Index([f"G{i}" for i in range(300)]),
        min_genes_per_cell=200,
        min_cells_per_gene=1,
    )
    assert int(floors.cell_keep.sum()) == 1

    require_non_empty_qc_result(floors, n_genes=int(floors.gene_keep.sum()))
    assert any("removed" in warning for warning in floors.warnings)
