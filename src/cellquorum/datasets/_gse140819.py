"""GSE140819 (Slyper et al. 2020): CellQuorum's reference benchmark and tutorial dataset.

The Broad/HTAPP single-cell *and* single-nucleus toolbox paper — 40 libraries across human
tumours, comparing dissociation protocols on matched biological samples. It is the reference
dataset because it supplies the three things no synthetic fixture can:

**Filtering ground truth.** Every library ships a metadata table with the authors' own calls:
``doublet`` (4,387 positives over 216,490 barcodes), ``emptydrop`` (2,082 positives across 17
libraries), and cell-type labels that are themselves QC judgements — ``Empty/Fibroblast``,
``Low quality Skeletal myoblast``, ``Doublet/Fibrocyte/Osteoblast``, 7,021 cells in total. So
"does QC agree with a human about which barcodes are junk" becomes a precision/recall number
rather than an opinion.

**The single-nucleus path.** ``nuclear_axis_applicable=False`` exists because MALAT1 is
nuclear-retained, so high MALAT1 is *expected* in snRNA rather than evidence of a ruptured cell.
23 of the 40 libraries are nuclei, and four donors have matched cell *and* nuclei libraries from
the same sample, so that axis can be tested instead of asserted.

**The populations QC destroys.** 412 mast cells and 145 lymphatic endothelial cells, labelled.
Both are low-RNA and high-mitochondrial by constitution, which is exactly the signature that a
sample-wide null mistakes for damage.

It also varies protocol on fixed biology — ``HTAPP-244-SMP-451`` has CST/EZ/NST/TST on one
tumour — which stresses per-sample nulls harder than any single-protocol cohort can.

## Why this is a module and not a script

Because a tutorial has to run for a stranger. A script in ``scripts/`` requires a repo checkout,
a manual download, and a working directory; a function does not. This follows the convention
users already know from ``scanpy.datasets``: call it, and the data appears.

    >>> from cellquorum import datasets
    >>> adata = datasets.gse140819("HTAPP-StJude-SMP-PDX1_cell")     # doctest: +SKIP

## Ground truth is prefixed, deliberately

Author calls land on ``obs`` as ``truth_annotate``, ``truth_doublet``, ``truth_emptydrop``,
``truth_percent_mito``. The prefix is not decoration: ground truth sitting unprefixed beside
computed QC columns is how a benchmark accidentally trains on its own answer, and how a figure
ends up plotting the label it was meant to predict.
"""

from __future__ import annotations

import gzip
import logging
import os
import re
import shutil
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - import cost
    import anndata as ad

logger = logging.getLogger(__name__)

#: GEO supplementary archive. 1.2 GB, never vendored into the package.
ARCHIVE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE140nnn/GSE140819/suppl/GSE140819_RAW.tar"
ARCHIVE_NAME = "GSE140819_RAW.tar"

#: Protocol tokens meaning single-NUCLEUS input. CST, NST and TST are nuclei isolation buffers
#: from the paper and EZ is a nuclei lysis kit; anything else (``fresh``, ``cell``, ``LD``) is
#: whole cells. Misreading this hands the nuclear-integrity axis the opposite of the truth, so it
#: is data rather than a heuristic.
NUCLEI_TOKENS: frozenset[str] = frozenset({"CST", "NST", "TST", "EZ", "nuclei"})

#: GSE140819 mixes CellRanger generations and the filename is the only signal: some libraries use
#: the v2 name ``raw_gene_bc_matrices_h5.h5``, others the v3 ``raw_feature_bc_matrix.h5``.
#: Matching one silently drops the other — it cost 11 libraries once, including a matched
#: cell/nuclei pair, which is most of the reason to use this dataset.
_MATRIX = re.compile(
    r"^(GSM\d+)_(.+?)_channel(\d+)_" r"(?:raw_gene_bc_matrices_h5|raw_feature_bc_matrix)\.h5\.gz$"
)
_METADATA = re.compile(r"^(GSM\d+)_metadata_(.+?)_channel(\d+)\.csv\.gz$")

#: The 10x cell barcode at the end of a metadata row label. Joining on this rather than on
#: the full label is not fastidiousness — the labels are inconsistent across libraries in
#: every way they could be. Most read ``<sample>_channel1-<barcode>``, some use
#: ``channel2``, some omit the channel, one has its separators mangled
#: (``HTAPP-951-SMP-4652TST-V2channel1_<barcode>``), and several bear no relation to the
#: sample name at all (``14MA``, ``NSC005bLT_FT``). Reconstructing the label from the sample
#: name silently produced zero overlap for two libraries and handed back all 737,280 raw
#: barcodes as though they were cells. The barcode itself is the only stable key.
_BARCODE = re.compile(r"([ACGT]{14,16})(?:-\d+)?$")

#: obs prefix for author ground truth. See the module docstring.
TRUTH_PREFIX = "truth_"


def data_home() -> Path:
    """Directory holding downloaded datasets.

    Resolution order, most explicit first:

    1. ``CELLQUORUM_DATA_HOME`` — the escape hatch, and the only one that works on a cluster
       where ``$HOME`` is small or read-only.
    2. ``<repo>/data`` when running from a source checkout, so a developer's copy is found
       rather than downloaded twice.
    3. ``~/.cache/cellquorum/data`` for an installed package.
    """
    explicit = os.environ.get("CELLQUORUM_DATA_HOME")
    if explicit:
        return Path(explicit).expanduser()

    # src/cellquorum/datasets/_gse140819.py -> repo root is four parents up.
    repo_data = Path(__file__).resolve().parents[3] / "data"
    if repo_data.is_dir():
        return repo_data

    return Path.home() / ".cache" / "cellquorum" / "data"


def dataset_dir() -> Path:
    """Directory for this dataset's files."""
    return data_home() / "gse140819"


def download(*, force: bool = False) -> Path:
    """Fetch the GEO archive if it is not already present.

    Args:
        force: Re-download even when the archive exists.

    Returns:
        Path to the archive.
    """
    target = dataset_dir()
    target.mkdir(parents=True, exist_ok=True)
    archive = target / ARCHIVE_NAME

    if archive.is_file() and not force:
        return archive

    import urllib.request

    logger.info("Downloading %s (~1.2 GB) to %s", ARCHIVE_URL, archive)
    # Written to a temporary name and renamed, so an interrupted download can never be mistaken
    # for a complete one on the next call.
    partial = archive.with_suffix(".tar.partial")
    with urllib.request.urlopen(ARCHIVE_URL) as response, partial.open("wb") as sink:
        shutil.copyfileobj(response, sink)
    partial.rename(archive)
    return archive


def extract(*, force: bool = False) -> tuple[Path, Path]:
    """Unpack the archive into per-library matrices and metadata.

    Decompressed on the way out because scanpy cannot read a gzipped h5, and because ~800 MB is
    a fair price for not decompressing 40 files on every call.

    Args:
        force: Re-extract even when the outputs exist.

    Returns:
        ``(raw_dir, metadata_dir)``.
    """
    archive = download()
    raw_dir = dataset_dir() / "raw"
    metadata_dir = dataset_dir() / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    unrecognised: list[str] = []
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            matrix, metadata = _MATRIX.match(name), _METADATA.match(name)
            if matrix is not None:
                destination = raw_dir / f"{matrix.group(2)}.h5"
            elif metadata is not None:
                destination = metadata_dir / f"{metadata.group(2)}.csv"
            else:
                unrecognised.append(name)
                continue

            if destination.exists() and not force:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            with gzip.open(handle) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)

    if unrecognised:
        # Reported, never silent: an unrecognised member means the archive layout changed and
        # some libraries are missing from every downstream result.
        logger.warning(
            "GSE140819: %d archive members did not match a known filename pattern and were "
            "skipped: %s",
            len(unrecognised),
            ", ".join(sorted(unrecognised)[:5]),
        )
    return raw_dir, metadata_dir


def _bare_barcode(label: object) -> str | None:
    """The 10x barcode at the end of a label, or None when there is not one."""
    match = _BARCODE.search(str(label))
    return match.group(1) if match else None


def _is_nuclei(sample: str) -> bool:
    """Whether a library name denotes single-nucleus input."""
    return bool(NUCLEI_TOKENS & set(sample.rsplit("_", 1)[-1].split("-")))


def manifest() -> pd.DataFrame:
    """One row per library: identity, assay, protocol, donor, and annotated cell count.

    ``assay`` is what a QC config keys ``nuclear_axis_applicable`` off, and ``donor_id`` strips
    the protocol suffix so the matched-protocol and matched-assay comparisons are expressible.

    Returns:
        The library table, indexed by ``sample_id``.
    """
    raw_dir, metadata_dir = extract()
    rows = []
    for matrix in sorted(raw_dir.glob("*.h5")):
        sample = matrix.stem
        metadata = metadata_dir / f"{sample}.csv"
        n_cells: int | str = ""
        n_types: int | str = ""
        if metadata.is_file():
            frame = pd.read_csv(metadata, index_col=0)
            n_cells = int(len(frame))
            n_types = int(frame["annotate"].nunique()) if "annotate" in frame.columns else ""

        rows.append(
            {
                "sample_id": sample,
                "path": str(matrix),
                "metadata_path": str(metadata) if metadata.is_file() else "",
                "donor_id": sample.rsplit("_", 1)[0] if "_" in sample else sample,
                "assay": "snRNA" if _is_nuclei(sample) else "scRNA",
                "protocol": sample.rsplit("_", 1)[-1] if "_" in sample else "unknown",
                "n_annotated_cells": n_cells,
                "n_annotated_types": n_types,
            }
        )
    return pd.DataFrame(rows).set_index("sample_id")


def truth(samples: str | list[str] | None = None) -> pd.DataFrame:
    """Author ground truth for one or more libraries, or all of them.

    Args:
        samples: Library name(s). None returns every library.

    Returns:
        A frame indexed by ``<sample>-<barcode>`` (the authors' own barcode format), carrying
        ``annotate``, ``doublet``, ``percent_mito``, ``nUMI``, ``nGene`` and — where the authors
        computed it — ``emptydrop``. Columns are unprefixed here because this frame *is* the
        truth; prefixing happens when it is attached to an AnnData.
    """
    _, metadata_dir = extract()
    wanted = _resolve_samples(samples)

    frames = []
    for sample in wanted:
        path = metadata_dir / f"{sample}.csv"
        if not path.is_file():
            logger.warning("GSE140819: no metadata for %s", sample)
            continue
        frame = pd.read_csv(path, index_col=0)
        frame["sample_id"] = sample
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def load(
    samples: str | list[str] | None = None,
    *,
    barcodes: Literal["annotated", "all"] = "annotated",
    with_truth: bool = True,
) -> ad.AnnData:
    """Load one or more libraries as an AnnData, with ground truth attached.

    Args:
        samples: Library name(s) from :func:`manifest`. None loads every library, which is
            ~216,000 annotated cells and is rarely what a tutorial wants.
        barcodes: ``"annotated"`` keeps only the barcodes the authors called and labelled — the
            right choice for benchmarking cell-type preservation. ``"all"`` returns the full
            raw matrix, 737,280 barcodes of which the vast majority are empty; that is the only
            way to exercise cell calling and ambient correction honestly, and it is why the raw
            matrices are used here rather than a filtered convenience copy.
        with_truth: Attach author calls to ``obs`` under the ``truth_`` prefix.

    Returns:
        The concatenated AnnData. ``obs`` carries ``sample_id``, ``donor_id``, ``assay`` and
        ``protocol``; gene names are made unique, since these matrices contain duplicates.
    """
    import anndata as ad_module
    import scanpy as sc

    table = manifest()
    wanted = _resolve_samples(samples)

    blocks = []
    for sample in wanted:
        row = table.loc[sample]
        adata = sc.read_10x_h5(row["path"])
        adata.var_names_make_unique()

        # Join on the bare barcode; see _BARCODE for why the labels cannot be trusted.
        adata.obs["barcode"] = [_bare_barcode(code) for code in adata.obs_names]
        adata.obs["sample_id"] = sample
        adata.obs["donor_id"] = row["donor_id"]
        adata.obs["assay"] = row["assay"]
        adata.obs["protocol"] = row["protocol"]

        if with_truth or barcodes == "annotated":
            labels = truth(sample)
            labels = labels.assign(barcode=[_bare_barcode(code) for code in labels.index])
            labels = labels[labels["barcode"].notna()].drop_duplicates("barcode")
            labels = labels.set_index("barcode")

            overlap = adata.obs["barcode"].isin(labels.index).to_numpy()
            if not overlap.any():
                # A total miss means the barcode convention changed again. Raising beats
                # returning 737,280 empty droplets dressed up as cells.
                raise ValueError(
                    f"GSE140819: no barcode overlap between the matrix and metadata for "
                    f"{sample!r}. The archive's barcode labelling may have changed; inspect "
                    f"{dataset_dir() / 'metadata' / f'{sample}.csv'}."
                )
            if barcodes == "annotated":
                adata = adata[overlap].copy()

            if with_truth:
                aligned = labels.reindex(adata.obs["barcode"].to_numpy())
                for column in ("annotate", "doublet", "emptydrop", "percent_mito", "nUMI", "nGene"):
                    if column in aligned.columns:
                        adata.obs[f"{TRUTH_PREFIX}{column}"] = aligned[column].to_numpy()

        blocks.append(adata)

    if len(blocks) == 1:
        return blocks[0]
    return ad_module.concat(blocks, label=None, index_unique=None)


def _resolve_samples(samples: str | list[str] | None) -> list[str]:
    """Normalise a sample selector into a list of library names."""
    _, metadata_dir = extract()
    if samples is None:
        raw_dir = dataset_dir() / "raw"
        return sorted(path.stem for path in raw_dir.glob("*.h5"))
    if isinstance(samples, str):
        return [samples]
    return list(samples)


__all__ = [
    "ARCHIVE_URL",
    "NUCLEI_TOKENS",
    "TRUTH_PREFIX",
    "data_home",
    "dataset_dir",
    "download",
    "extract",
    "load",
    "manifest",
    "truth",
]
