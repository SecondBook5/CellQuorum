"""Reference datasets for benchmarking CellQuorum and for the tutorials.

    >>> from cellquorum import datasets
    >>> table = datasets.gse140819_manifest()                       # doctest: +SKIP
    >>> adata = datasets.gse140819("HTAPP-StJude-SMP-PDX1_cell")    # doctest: +SKIP

Only one dataset so far, and it earns its place by carrying **filtering ground truth** rather
than merely being public: GSE140819 (Slyper et al. 2020) ships the authors' own doublet,
empty-droplet and cell-type calls, so QC accuracy is measurable instead of arguable. See
:mod:`cellquorum.datasets._gse140819` for what that enables and why it is a module rather than a
script.

Nothing here is imported eagerly. The dataset functions pull in ``anndata`` and ``scanpy``, and
importing them at package import time would put a multi-second cost on ``import cellquorum`` for
every user who never touches a dataset — the exact regression the lazy-import work removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - types only
    import anndata as ad
    import pandas as pd

__all__ = [
    "gse140819",
    "gse140819_data_home",
    "gse140819_download",
    "gse140819_manifest",
    "gse140819_truth",
]


def gse140819(samples: str | list[str] | None = None, **kwargs: Any) -> ad.AnnData:
    """Load GSE140819 libraries with author ground truth attached.

    See :func:`cellquorum.datasets._gse140819.load` for the arguments, notably ``barcodes`` —
    ``"annotated"`` for benchmarking cell-type preservation, ``"all"`` for exercising cell
    calling on the full raw matrix.
    """
    from cellquorum.datasets._gse140819 import load

    return load(samples, **kwargs)


def gse140819_manifest() -> pd.DataFrame:
    """The 40-library table: assay, protocol, donor, annotated cell count.

    Downloads and extracts the archive on first use.
    """
    from cellquorum.datasets._gse140819 import manifest

    return manifest()


def gse140819_truth(samples: str | list[str] | None = None) -> pd.DataFrame:
    """Author ground truth — cell types, doublet and empty-droplet calls."""
    from cellquorum.datasets._gse140819 import truth

    return truth(samples)


def gse140819_download(*, force: bool = False) -> Any:
    """Fetch the archive without loading anything, for priming a cache offline."""
    from cellquorum.datasets._gse140819 import download

    return download(force=force)


def gse140819_data_home() -> Any:
    """Where datasets are stored. Override with ``CELLQUORUM_DATA_HOME``."""
    from cellquorum.datasets._gse140819 import data_home

    return data_home()
