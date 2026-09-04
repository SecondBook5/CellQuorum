"""AnnData input/output utilities for CellQuorum."""

from __future__ import annotations

# Import PathLike for filesystem-like input path typing.
from os import PathLike

# Import Path for robust filesystem validation.
from pathlib import Path
from typing import TYPE_CHECKING

# Import AnnData for return type validation.
import anndata as ad

# Import numpy for the boolean row masks the subset/exclude rules build.
import numpy as np

# Import pandas for obs frame typing in the column helper.
import pandas as pd

# Import the layer-tag writer so a layer read into X keeps saying what it is.
from cellquorum.core.contracts.layer_tags import set_layer_tag

# Import shared CellQuorum data exception.
from cellquorum.core.exceptions import CellQuorumDataError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class AnnDataLoadError(CellQuorumDataError):
    """
    Report AnnData loading failures.

    This error is raised when a user-supplied AnnData path is missing, malformed,
    points to an unsupported file type, or cannot be read into an AnnData object.
    """


def normalize_adata_path(path: str | PathLike[str] | Path) -> Path:
    """
    Normalize a user-supplied AnnData path.

    Args:
        path: Candidate AnnData path.

    Returns:
        Expanded Path object.

    Raises:
        AnnDataLoadError: If the path is empty or has an invalid type.
    """

    # Reject empty string paths before Path("") turns into the current directory.
    if isinstance(path, str) and not path.strip():
        raise AnnDataLoadError("AnnData path cannot be empty.")

    # Convert the candidate path into a Path object.
    try:
        normalized_path = Path(path).expanduser()

    # Convert invalid path-like objects into AnnData loading errors.
    except TypeError as error:
        raise AnnDataLoadError(
            "AnnData path must be a string or filesystem path-like object. "
            f"Received: {type(path).__name__}."
        ) from error

    # Return the normalized path.
    return normalized_path


def validate_adata_path(path: str | PathLike[str] | Path) -> Path:
    """
    Validate an AnnData input path.

    Args:
        path: Candidate AnnData path.

    Returns:
        Validated Path object.

    Raises:
        AnnDataLoadError: If the path is missing, not a file, or unsupported.
    """

    # Normalize the input path.
    normalized_path = normalize_adata_path(path)

    # Reject missing paths.
    if not normalized_path.exists():
        raise AnnDataLoadError(f"AnnData file does not exist: {normalized_path}")

    # Reject directories.
    if not normalized_path.is_file():
        raise AnnDataLoadError(f"AnnData path is not a file: {normalized_path}")

    # Reject unsupported file suffixes.
    if normalized_path.suffix.lower() != ".h5ad":
        raise AnnDataLoadError(
            "CellQuorum currently supports AnnData input only as '.h5ad'. "
            f"Received: {normalized_path.name}"
        )

    # Return the validated path.
    return normalized_path


def _require_obs_column(obs: pd.DataFrame, column: str, *, setting: str) -> pd.Series:
    """
    Return an obs column as plain strings, failing loudly when it is absent.

    Args:
        obs: The obs frame to read from.
        column: Column name to fetch.
        setting: Config setting that named the column, quoted in the error.

    Returns:
        The column with every value coerced to ``str``.

    Raises:
        AnnDataLoadError: If the column is not in obs.
    """

    if column not in obs.columns:
        available = list(obs.columns)
        raise AnnDataLoadError(
            f"{setting} {column!r} not found in obs. "
            f"Available columns include: {available[:25]}"
        )
    # Compare as strings so categorical/object/nullable dtypes all match.
    return obs[column].astype("string").astype(object)


def _load_adata_subset(
    validated_path: Path,
    subset_column: str | None,
    subset_values: list[str],
    agreement_column: str | None = None,
    exclude_column: str | None = None,
    exclude_values: list[str] | None = None,
) -> ad.AnnData:
    """
    Load only the rows of an h5ad an inclusion and/or exclusion rule keeps.

    The object is opened in backed mode so its full X matrix never enters memory;
    obs (small) is read to build the row mask, and only the matching slice is
    materialized. This lets a hypothesis restrict a large shared global object to
    its cell type without a separate pre-sliced file and without the peak memory
    of loading every cell. The applied restriction is recorded on the returned
    object at ``uns['cellquorum_input_subset']`` so the run can log it.

    Exclusion exists because an inclusion list cannot express "everything except".
    Dropping one artifact cluster from a 39-cluster partition by inclusion means
    naming the 38 real ones, which is both unreadable and wrong the moment the
    object is re-clustered. The two rules compose: a lineage slice that also drops
    an artifact cluster needs both, and applying them in one pass keeps the counts
    in provenance attributable to the right rule.

    Args:
        validated_path: Validated h5ad path.
        subset_column: obs column to filter on, or None for exclusion only.
        subset_values: accepted values; a row is kept when its value is in here.
        agreement_column: Optional second annotation column that must carry the
            same value as ``subset_column``, dropping cells the two annotations
            disagree about.
        exclude_column: Optional obs column to drop rows on.
        exclude_values: Values of ``exclude_column`` whose rows are dropped. A
            value absent from the column raises, because a mask that names
            categories the object does not have removes nothing while reading in
            the config as though it had -- the failure mode when a cluster id set
            is carried across two different clusterings of the same cells.

    Returns:
        In-memory AnnData holding only the matching rows.

    Raises:
        AnnDataLoadError: If a column is absent, an agreement value is outside the
            agreement column's vocabulary, an excluded value is outside the exclude
            column's vocabulary, or no rows match.
    """

    # Open backed so the full expression matrix is never read into memory.
    backed = ad.read_h5ad(validated_path, backed="r")

    try:
        n_before = int(backed.n_obs)
        if subset_column is None:
            column_as_str = None
            wanted: set[str] = set()
            mask = np.ones(n_before, dtype=bool)
        else:
            column_as_str = _require_obs_column(
                backed.obs, subset_column, setting="input.subset.column"
            )
            wanted = {str(value) for value in subset_values}
            mask = column_as_str.isin(wanted).to_numpy()

        n_selected = int(mask.sum())
        n_discordant: int | None = None
        n_excluded: int | None = None

        if exclude_column is not None:
            exclude_as_str = _require_obs_column(
                backed.obs, exclude_column, setting="input.exclude.column"
            )
            unwanted = {str(value) for value in (exclude_values or [])}

            # Guard the VOCABULARY before the cells, for the same reason the
            # agreement check does: "these cells are not here" and "this column has
            # no word for them" are different facts, and only one of them means the
            # exclusion did its job. A leiden id set written against one clustering
            # names nothing in the next one, so silently excluding zero cells would
            # leave the real artifact in while the config claims otherwise.
            vocabulary = {str(value) for value in exclude_as_str.dropna().unique()}
            absent = sorted(unwanted - vocabulary)
            if absent:
                raise AnnDataLoadError(
                    f"input.exclude.values {absent} are not values of "
                    f"{exclude_column!r} in this object, so excluding them removes "
                    f"nothing. If these are cluster ids, they belong to a different "
                    f"clustering run than this object's: re-identify the clusters on "
                    f"this partition (cellquorum.stats.cluster_artifact_audit) rather "
                    f"than carrying ids across."
                )

            drop = exclude_as_str.isin(unwanted).to_numpy()
            n_excluded = int((mask & drop).sum())
            mask = mask & ~drop

        if agreement_column is not None:
            assert column_as_str is not None  # guarded by load_adata
            agreement_as_str = _require_obs_column(
                backed.obs, agreement_column, setting="input.subset.require_agreement"
            )

            # Guard the VOCABULARY before the cells. A label the second annotation
            # has no word for cannot agree with anything, so requiring concordance
            # would drop the entire population -- and the cell count alone cannot
            # tell that apart from a population the second annotation genuinely
            # rejects. Real case: a granular reference with no 'B cells' or
            # 'Neutrophils' level silently emptied both lineages.
            vocabulary = set(agreement_as_str.dropna().unique())
            missing = sorted(wanted - vocabulary)
            if missing:
                raise AnnDataLoadError(
                    f"input.subset.require_agreement {agreement_column!r} has no "
                    f"value {missing} in it, so no cell can agree and the slice "
                    f"would be empty. This is a vocabulary mismatch, not a "
                    f"filtering result: {agreement_column!r} cannot express "
                    f"{missing}. Either drop require_agreement for this subset or "
                    f"map the labels onto a shared vocabulary first."
                )

            agrees = (agreement_as_str == column_as_str).to_numpy()
            n_discordant = int((mask & ~agrees).sum())
            mask = mask & agrees

        n_after = int(mask.sum())

        # An empty slice is almost always a mislabeled value; fail rather than
        # silently run a pipeline on zero cells.
        if n_after == 0:
            rule = (
                f"input.subset {subset_column} in {sorted(wanted)}"
                if subset_column is not None
                else f"input.exclude {exclude_column} not in {sorted(exclude_values or [])}"
            )
            raise AnnDataLoadError(
                f"{rule} matched 0 of {n_before} cells. Check the value spelling "
                f"against the obs column."
            )

        # Materialize only the matching rows from the still-open backed file.
        subset_adata = backed[mask].to_memory()

    finally:
        # Release the backed file handle regardless of success.
        if backed.isbacked and backed.file is not None:
            backed.file.close()

    # Record the applied restriction for run provenance. n_selected is kept
    # alongside n_after so a reader can see what the agreement requirement cost
    # rather than only what survived it.
    subset_adata.uns["cellquorum_input_subset"] = {
        "column": subset_column,
        "values": sorted(wanted),
        "n_before": n_before,
        "n_selected": n_selected,
        "n_after": n_after,
        "require_agreement": agreement_column,
        "n_discordant": n_discordant,
        "exclude_column": exclude_column,
        "exclude_values": sorted(exclude_values or []) if exclude_column else None,
        "n_excluded": n_excluded,
    }

    return subset_adata


def load_adata(
    path: str | PathLike[str] | Path,
    *,
    subset_column: str | None = None,
    subset_values: list[str] | None = None,
    agreement_column: str | None = None,
    exclude_column: str | None = None,
    exclude_values: list[str] | None = None,
) -> ad.AnnData:
    """
    Load an AnnData object from an h5ad file, optionally restricted to a slice.

    Args:
        path: Path to an h5ad file.
        subset_column: Optional obs column to restrict rows on.
        subset_values: Values to keep for ``subset_column``; required when it is
            given. When both are ``None`` the full object is read (default).
        agreement_column: Optional second annotation column that must carry the
            same value as ``subset_column``; only meaningful with a subset.
        exclude_column: Optional obs column to DROP rows on, independent of the
            subset and composable with it.
        exclude_values: Values of ``exclude_column`` whose rows are dropped;
            required when it is given.

    Returns:
        Loaded AnnData object.

    Raises:
        AnnDataLoadError: If the file path is invalid, reading fails, or a
            partially specified subset or exclusion is given.
    """

    # Validate the h5ad path before reading.
    validated_path = validate_adata_path(path)

    # A subset needs both a column and its values; reject a half-specified one.
    if (subset_column is None) != (subset_values is None):
        raise AnnDataLoadError(
            "load_adata subset requires both subset_column and subset_values, " "or neither."
        )

    # Same for the exclusion rule, and for the same reason: half of a filter reads
    # like a filter and removes nothing.
    if (exclude_column is None) != (exclude_values is None):
        raise AnnDataLoadError(
            "load_adata exclusion requires both exclude_column and exclude_values, or neither."
        )

    # An agreement column with nothing to agree WITH is a config that reads as if
    # it filters and does not; refuse it rather than ignore it.
    if agreement_column is not None and subset_column is None:
        raise AnnDataLoadError(
            "load_adata agreement_column requires a subset_column for it to agree "
            "with; on its own it has no reference label to compare against."
        )

    # Try to read the AnnData object (full, or backed-mode slice).
    try:
        if subset_column is None and exclude_column is None:
            loaded_adata = ad.read_h5ad(validated_path)
        else:
            loaded_adata = _load_adata_subset(
                validated_path,
                subset_column,
                subset_values or [],
                agreement_column=agreement_column,
                exclude_column=exclude_column,
                exclude_values=exclude_values,
            )

    # Preserve CellQuorum-specific errors (e.g. bad subset) without re-wrapping.
    except AnnDataLoadError:
        raise

    # Convert AnnData/HDF5 read failures into CellQuorum-specific errors.
    except Exception as error:
        raise AnnDataLoadError(f"Failed to read AnnData file '{validated_path}'.") from error

    # Validate the loaded object defensively.
    if not isinstance(loaded_adata, ad.AnnData):
        raise AnnDataLoadError(
            "Expected an AnnData object from read_h5ad, but received "
            f"{type(loaded_adata).__name__}."
        )

    # Return the loaded AnnData object.
    return loaded_adata


def load_group_sample(
    path: str | PathLike[str] | Path,
    *,
    group_column: str,
    per_group: Mapping[str, int | None],
    agreement_column: str | None = None,
    genes: Sequence[str] | None = None,
    layer: str | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """
    Load a capped, optionally gene-restricted, per-group slice of a large atlas.

    A whole-atlas h5ad is the right thing to keep on disk and the wrong thing to load. An
    analysis that needs every cell type but only a few thousand cells of each -- which is
    every cross-population analysis: ligand-receptor, communication, per-type expression
    fractions -- cannot use :func:`load_adata`'s subset, because that has no per-group cap
    and would materialize the largest population in full. This reads the rows it was asked
    for, from one layer, for the genes it was asked for, and nothing else.

    Two decisions are recorded rather than taken silently, both on
    ``uns['cellquorum_group_sample']``:

    * **The agreement gate is applied per group, not globally.** A second annotation column
      often has no word for some of the first column's labels, and requiring concordance
      there deletes the whole population. So a group whose label is outside the agreement
      column's vocabulary keeps the ``group_column`` call, and the report says which gate
      each group got. Compare :func:`_load_adata_subset`, which refuses the whole load
      instead -- correct when the caller named one lineage, wrong when the caller named
      every cell type in the object and two of them are unrepresentable.
    * **Which cells the cap kept.** Sampling is seeded and the selected barcodes are the
      object's own ``obs_names``, so the sample is reproducible from the report. Each group is
      seeded from ``(seed, group name)`` rather than drawn from one shared stream, so a group's
      sample is a property of that group and not of which other groups were requested or in
      what order. Sharing a stream is the obvious implementation and it silently breaks the
      comparison this reader exists for: two runs that both ask for the same 2,000 cells of a
      cell type get *different* 2,000 cells if one of them also asked for something else first,
      and any difference between the runs then has a sampling explanation nobody can rule out.

    One thing travels rather than being recorded: the read layer's provenance tag. The values
    arrive in ``X`` under a new name, and every stage that declares an expected layer kind asks
    a contract what ``X`` is, so a tag left behind turns a correctly normalized atlas into an
    object the engine refuses -- with a message about a missing tag rather than about the data.
    The tag is copied under the name the values now have, and ``None`` stays ``None``: an
    untagged source is not given a tag here, because this function does not know what the layer
    holds, it only knows where it came from.

    Args:
        path: h5ad to read.
        group_column: obs column naming the groups.
        per_group: Group name to cap. ``None`` as a cap means every cell of that group.
            Groups absent from the object are reported, not raised on: a candidate-sender
            list is usually derived from a different object.
        agreement_column: Optional second annotation column that must carry the same value
            as ``group_column``, applied per group as described above.
        genes: Restrict to these var names, in the object's own order. Absent names are
            reported. ``None`` keeps every gene.
        layer: Layer to read as X. ``None`` reads X itself.
        seed: Seed for the per-group subsample.

    Returns:
        In-memory AnnData with the selected cells and genes, X taken from ``layer``.

    Raises:
        AnnDataLoadError: If a named column or layer is absent, or nothing was selected.
    """
    import hashlib

    import h5py
    from anndata.io import read_elem, sparse_dataset

    validated_path = validate_adata_path(path)

    def group_rng(name: str) -> np.random.Generator:
        """A stream that depends on the group's name and the seed, and on nothing else.

        ``blake2b`` rather than :func:`hash`, which is salted per process and so would make the
        sample reproducible within a run and different between runs -- the worst of both.
        """
        digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
        return np.random.default_rng([int(seed), int.from_bytes(digest, "big")])

    with h5py.File(validated_path, "r") as handle:
        obs = read_elem(handle["obs"])
        var = read_elem(handle["var"])
        groups_as_str = _require_obs_column(obs, group_column, setting="group_column")
        agreement_as_str = (
            _require_obs_column(obs, agreement_column, setting="agreement_column")
            if agreement_column is not None
            else None
        )
        vocabulary = (
            set(agreement_as_str.dropna().unique()) if agreement_as_str is not None else set()
        )

        def as_bool_mask(series: pd.Series) -> np.ndarray:
            """A comparison over object dtype yields pd.NA, not False, where a value is missing."""
            return series.where(series.notna(), False).to_numpy(dtype=bool)

        report: list[dict[str, object]] = []
        keep = np.zeros(len(obs), dtype=bool)
        for group, cap in per_group.items():
            name = str(group)
            in_group = as_bool_mask(groups_as_str == name)
            n_available = int(in_group.sum())
            gate = "group_column"
            if agreement_as_str is not None and name in vocabulary:
                in_group = in_group & as_bool_mask(agreement_as_str == groups_as_str)
                gate = f"agrees_with_{agreement_column}"
            n_eligible = int(in_group.sum())

            positions = np.flatnonzero(in_group)
            if cap is not None and len(positions) > int(cap):
                positions = np.sort(group_rng(name).choice(positions, size=int(cap), replace=False))
            keep[positions] = True
            report.append(
                {
                    "group": name,
                    "n_available": n_available,
                    "n_eligible": n_eligible,
                    "gate": gate,
                    "cap": None if cap is None else int(cap),
                    "n_selected": int(len(positions)),
                }
            )

        if not keep.any():
            raise AnnDataLoadError(
                f"load_group_sample selected 0 cells for {sorted(str(g) for g in per_group)} "
                f"in {group_column!r}. Check the value spelling against the obs column."
            )

        gene_index = var.index.astype(str)
        if genes is None:
            columns = np.arange(len(gene_index))
            missing_genes: list[str] = []
        else:
            wanted = {str(gene) for gene in genes}
            columns = np.flatnonzero(gene_index.isin(wanted))
            missing_genes = sorted(wanted - set(gene_index[columns]))
            if columns.size == 0:
                raise AnnDataLoadError(
                    f"load_group_sample matched 0 of {len(wanted)} requested genes in var."
                )

        key = "X" if layer is None else f"layers/{layer}"
        if key not in handle:
            raise AnnDataLoadError(f"{validated_path} has no {key!r}.")
        rows = np.flatnonzero(keep)
        matrix = sparse_dataset(handle[key])[rows][:, columns]

        # A layer's provenance has to travel with its values. This reader moves a named layer
        # into ``X``, and the stages downstream ask a contract what ``X`` *is* -- so without
        # this the engine rejects its own correctly tagged atlas the moment it is read through
        # here, with a message ("no provenance tag") that describes the reader rather than the
        # data. The tag is read under the name it was stored as and re-recorded under the name
        # the values now have.
        tag_key = f"uns/cellquorum/layer_tags/{'X' if layer is None else layer}"
        source_tag = read_elem(handle[tag_key]) if tag_key in handle else None

    selected_obs = obs.iloc[rows].copy()
    # A categorical sliced down to two of thirteen cell types keeps all thirteen levels, and a
    # groupby or an R factor built from it then carries groups holding no cells. Downstream that
    # is an empty-group error at best and a silently empty per-group statistic at worst.
    for name in selected_obs.columns:
        if isinstance(selected_obs[name].dtype, pd.CategoricalDtype):
            selected_obs[name] = selected_obs[name].cat.remove_unused_categories()

    selected = ad.AnnData(
        X=matrix,
        obs=selected_obs,
        var=var.iloc[columns].copy(),
    )
    if source_tag is not None:
        set_layer_tag(
            selected,
            "X",
            kind=str(source_tag["kind"]),
            recipe=None if source_tag.get("recipe") is None else str(source_tag["recipe"]),
        )
    selected.uns["cellquorum_group_sample"] = {
        "source": str(validated_path),
        "group_column": group_column,
        "agreement_column": agreement_column,
        "layer": layer,
        "layer_tag": None if source_tag is None else dict(source_tag),
        "seed": int(seed),
        "n_before": int(len(obs)),
        "n_after": int(selected.n_obs),
        "n_genes_requested": None if genes is None else len(set(str(g) for g in genes)),
        "n_genes_kept": int(selected.n_vars),
        "genes_absent": missing_genes,
        "groups": report,
    }
    return selected


__all__ = [
    "AnnDataLoadError",
    "load_adata",
    "load_group_sample",
    "normalize_adata_path",
    "validate_adata_path",
]
