"""One string form for a grouping label, shared by every consumer of one.

A label that looks like a number is still a label, and this engine's own upstream
stages produce exactly that: ``leiden`` (the default ``cell_type_col``) is "0",
"1", "2", and a subcluster column is numeric with NaN for the cells outside the
analysed focus -- which pandas stores as float64.

Left alone, that costs real results twice over. ``.astype(str)`` names such a
state **"1.0"**, while the same column written to CSV and read back by an R
backend returns **"1"**: one run, one state, two names, in two tables a reader is
meant to compare. And when the engine then joins those tables on the cell-type
key, pandas refuses -- "You are trying to merge on float64 and object columns for
key 'cell_type'" -- which is how the abundance stage failed on its first real use,
after two of its four methods had already written their CSVs.

This module exists so the answer is given once, in a place both a stage and a
statistics helper can import, rather than each ``astype(str)`` making its own.
"""

from __future__ import annotations

import pandas as pd


def as_label_strings(values: pd.Series) -> pd.Series:
    """Render grouping labels as the one string form the whole run will use.

    Integral floats lose the decimal point ("1.0" -> "1"), everything else becomes
    its own string, and missing stays missing -- the count-based methods have to be
    able to see and exclude an unlabelled cell, and a neighbourhood method has to
    be able to keep it.

    Args:
        values: Any Series used as a grouping label (cell type, state, donor,
            condition).

    Returns:
        An object-dtype Series of label strings with missing values preserved.
    """

    # A categorical's own dtype would be preserved by ``map``, which would put the
    # canonical strings back into non-canonical categories.
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype(object)

    def _one(value: object) -> object:
        if pd.isna(value):
            return value
        # np.float64 subclasses float, so this covers numpy scalars too.
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    return values.map(_one)


__all__ = ["as_label_strings"]
