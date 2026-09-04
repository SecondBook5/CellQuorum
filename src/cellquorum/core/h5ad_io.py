"""One h5ad writer for the engine, and the sanitation a real object needs to pass.

Why this exists
---------------
Stages used to call ``adata.write_h5ad`` directly, and each call site learned the
hard way about a different thing h5py refuses: a ``/`` in a key, a Python object
in ``uns`` that has no HDF5 encoding, a pandas nullable string column. The
knowledge accumulated in whichever module was last bitten, so the next writer
started from zero.

That is not merely untidy. Most of these writers are deliberately
skip-not-crash — a failed artifact write must not destroy a two-hour run — so a
writer that has not learned a given lesson does not fail, it records a one-line
note and carries on with a hole where the object should be. On the LEC
mechanotransduction run a single obs column (``donor_qc_qc_pass``, a boolean gate
verdict widened onto the whole object so unexamined cells held NaN) made every
h5ad write in the run raise ``TypeError: Can't implicitly convert non-string
objects to strings``. The run reported success. It had no final object, no
loadable checkpoint, and no velocity h5ad — which in turn silently cost CellRank
its velocity and CytoTRACE kernels, because those consume the file that was never
written.

So the lessons live here, once, and every writer goes through
:func:`write_h5ad`.

What "sanitize" does and does not mean
-------------------------------------
Only *representations* are changed, never the information: keys lose ``/``,
mixed-type label columns become string categoricals, and an ``uns`` value with no
HDF5 encoding becomes its JSON text (recoverable with ``json.loads``). Every
change is returned as a note so it lands in the stage record instead of happening
invisibly.

Columns that already write are left alone, including the ones that look
suspicious: a string column with NaN, or a bool column with NaN, both round-trip
fine, and rewriting them would be a dtype change nobody asked for.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

__all__ = [
    "H5adWriteError",
    "sanitize_for_h5ad",
    "write_h5ad",
]


class H5adWriteError(RuntimeError):
    """Raised when an AnnData could not be written to h5ad."""


def write_h5ad(adata: ad.AnnData, path: Path | str, *, sanitize: bool = True) -> list[str]:
    """Write ``adata`` to ``path``, atomically, returning notes on what was coerced.

    Atomic because a half-written h5ad is worse than none: it exists, so a later
    run treats it as a real object and fails somewhere far from the cause. The
    truncated checkpoints that prompted this module were exactly that — an
    ``adata.h5ad`` holding ``X`` and ``obs`` and nothing else, because the write
    died partway through and left the file behind.

    Args:
        adata: Object to write. Sanitation mutates it in place (see
            :func:`sanitize_for_h5ad`); it is the same object the pipeline carries
            forward, and the coercions are exactly the ones that let it be written
            again later.
        path: Destination ``.h5ad`` path. Parent directories are created.
        sanitize: Set False to write exactly what is in memory and let a
            non-writable element raise. For tests of the raw behaviour.

    Returns:
        Human-readable notes, one per coercion applied. Empty when the object was
        already writable.

    Raises:
        H5adWriteError: On any write failure, with the original error chained.
            Callers that must not fail a run catch this and record the note.
    """
    import anndata

    target = Path(path)
    # Same directory as the target, so the replace below is a rename within one
    # filesystem and therefore atomic.
    temp_path = target.with_name(target.name + ".tmp")
    notes: list[str] = []
    try:
        # Inside the try, along with the mkdir, because EVERY step here can fail
        # and callers that must not fail a run catch H5adWriteError only. A bug in
        # the sanitizer used to escape as a raw TypeError and kill an entire run
        # from inside a writer whose whole contract is skip-not-crash.
        if sanitize:
            notes = sanitize_for_h5ad(adata)
        target.parent.mkdir(parents=True, exist_ok=True)
        # anndata >= 0.11 refuses pandas nullable / Arrow-backed string columns
        # unless the caller opts in. Real objects — anything annotated by an
        # external tool — are full of them, so the objects most worth writing are
        # exactly the ones that fail without this.
        if hasattr(anndata.settings, "allow_write_nullable_strings"):
            anndata.settings.allow_write_nullable_strings = True
        adata.write_h5ad(temp_path)
        temp_path.replace(target)
    except Exception as exc:  # noqa: BLE001 — re-raised as a typed error below
        # Suppressed throughout: if the directory itself is the problem, unlink
        # raises too, and a failed cleanup must not replace the error that
        # explains the failure.
        with contextlib.suppress(OSError):
            temp_path.unlink(missing_ok=True)
        # Writing atomically means a failure leaves whatever was already at
        # ``target`` untouched — which is the wrong thing to keep. Consumers
        # resolve these files by convention (CellRank opens whichever
        # ``whole_object.h5ad`` is there) and cannot tell one run's output from
        # an earlier attempt's, so an object left behind here is read as this
        # run's result. Absent, it makes the consumer skip and say so, which is
        # visibly incomplete instead of invisibly wrong.
        stale = ""
        if target.exists():
            with contextlib.suppress(OSError):
                target.unlink()
                stale = " (removed the older file that was at this path)"
        raise H5adWriteError(f"could not write h5ad '{target}': {exc}{stale}") from exc
    return notes


def sanitize_for_h5ad(adata: ad.AnnData) -> list[str]:
    """Make ``adata`` writable to h5ad, in place, returning notes on each change.

    Four classes of problem, all of which have cost a real run:

    1. ``/`` in a key. h5py reads it as a path separator, so a label like
       ``"T/NK"`` cannot name a group — and labels leak into keys in four places:
       ``uns`` dicts, obs/var column names (scArches writes ``refprob_<label>``),
       and ``obsm``/``varm``.
    2. Mixed-type obs/var columns. A column holding strings for some cells and
       numbers or NaN-as-float for others has no single HDF5 encoding.
    3. Categoricals whose *categories* are mixed types — the same problem one
       level down, and easy to create by relabelling numeric clusters.
    4. ``uns`` values with no HDF5 encoding at all (lists of dicts, ragged
       structures) — stage payloads routinely stash these.

    Args:
        adata: Object to sanitize in place.

    Returns:
        One note per change, empty when nothing needed changing.
    """
    notes: list[str] = []
    notes += _sanitize_uns_keys(adata.uns)
    notes += _sanitize_frame_columns(adata.obs, axis="obs")
    notes += _sanitize_frame_columns(adata.var, axis="var")
    notes += _sanitize_mapping_keys(adata.obsm, name="obsm")
    notes += _sanitize_mapping_keys(adata.varm, name="varm")
    notes += _coerce_unwritable_columns(adata.obs, axis="obs")
    notes += _coerce_unwritable_columns(adata.var, axis="var")
    notes += _jsonify_unwritable_uns(adata.uns)
    return notes


def safe_h5_key(key: str, existing: object) -> str:
    """Return ``key`` with ``/`` replaced by ``_``, kept unique against ``existing``."""
    safe = key.replace("/", "_")
    while safe != key and safe in existing:
        safe += "_"
    return safe


def _sanitize_frame_columns(frame: pd.DataFrame, *, axis: str) -> list[str]:
    """Rename obs/var columns whose NAMES contain ``/``. Values are untouched."""
    renames: dict[str, str] = {}
    for col in list(frame.columns):
        if isinstance(col, str) and "/" in col:
            renames[col] = safe_h5_key(col, set(frame.columns) | set(renames.values()))
    if not renames:
        return []
    frame.rename(columns=renames, inplace=True)
    return [f"{axis}: renamed {len(renames)} column(s) containing '/' ({_first_few(renames)})"]


def _sanitize_mapping_keys(mapping: object, *, name: str) -> list[str]:
    """Rename obsm/varm keys containing ``/``."""
    bad = [k for k in list(mapping.keys()) if isinstance(k, str) and "/" in k]
    for key in bad:
        mapping[safe_h5_key(key, set(mapping.keys()))] = mapping.pop(key)
    if not bad:
        return []
    return [f"{name}: renamed {len(bad)} key(s) containing '/' ({', '.join(sorted(bad)[:3])})"]


def _sanitize_uns_keys(node: object, *, _notes: list[str] | None = None) -> list[str]:
    """Recursively replace ``/`` with ``_`` in dict keys inside an uns-like structure.

    Only KEYS are rewritten. Values — including category labels stored as arrays —
    are left alone, because those serialize fine and are the data.
    """
    notes = [] if _notes is None else _notes
    if isinstance(node, dict):
        for key in [k for k in node if isinstance(k, str) and "/" in k]:
            safe_key = safe_h5_key(key, node)
            node[safe_key] = node.pop(key)
            notes.append(f"uns: renamed key '{key}' -> '{safe_key}' ('/' is an h5 separator)")
        for value in node.values():
            _sanitize_uns_keys(value, _notes=notes)
    elif isinstance(node, list | tuple):
        for item in node:
            _sanitize_uns_keys(item, _notes=notes)
    return notes


def _coerce_unwritable_columns(frame: pd.DataFrame, *, axis: str) -> list[str]:
    """Coerce ``object`` obs/var columns h5py cannot encode into ones it can.

    An ``object`` column is where a dtype went to die, and the usual cause is
    widening: reindexing a per-subset column onto the whole object fills the gap
    with float NaN, so a bool column becomes ``{True, False, nan}`` and a
    categorical becomes loose strings. What each case needs differs, and picking
    by what the surviving values actually ARE is the only way to avoid inventing
    data:

    * booleans + missing → pandas nullable ``boolean``, which is precisely the
      three-valued thing the column has become (passed / failed / never assessed)
      and round-trips as such. Filling the gap with False instead would assert
      that unexamined cells passed a gate they never entered.
    * strings + missing → already writable; left alone.
    * anything genuinely mixed, and categoricals whose CATEGORIES are mixed → a
      string categorical: the encoding pandas uses for labels, and it keeps
      missing values missing instead of inventing a ``"nan"`` label.

    The bool case is not hypothetical. ``obs['donor_qc_qc_pass']`` in exactly that
    state raised ``TypeError: Can't implicitly convert non-string objects to
    strings`` — h5py had created the empty string dataset and could not fill it —
    and took a whole run's objects with it.
    """
    notes: list[str] = []
    for col in list(frame.columns):
        series = frame[col]

        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = list(series.cat.categories)
            kinds = {type(c).__name__ for c in categories}
            if len(kinds) <= 1:
                continue
            frame[col] = _as_string_categorical(series)
            notes.append(
                f"{axis}['{col}']: categories mixed {'/'.join(sorted(kinds))}; "
                "coerced to string categories for h5ad"
            )
            continue

        if series.dtype != object:
            continue

        present = series[series.notna()]
        kinds = {type(v).__name__ for v in present}
        if present.size == 0:
            # Entirely missing, and still ``object``: nothing says what it was, so
            # h5py has no type to write it as. An all-missing categorical is the
            # one encoding that says exactly that. This is the shape a column takes
            # when a projection matched no cells at all.
            frame[col] = pd.Categorical([None] * len(series))
            notes.append(
                f"{axis}['{col}']: entirely missing with no dtype; "
                "written as an empty categorical"
            )
            continue
        if all(isinstance(v, str) for v in present):
            # Strings with missing values write as they are.
            continue
        if (
            present.size
            and series.isna().any()
            and all(isinstance(v, bool | np.bool_) for v in present)
        ):
            frame[col] = pd.array(
                [None if pd.isna(v) else bool(v) for v in series], dtype="boolean"
            )
            notes.append(
                f"{axis}['{col}']: booleans with missing values; "
                "coerced to nullable boolean for h5ad"
            )
            continue
        if len(kinds) <= 1:
            continue
        frame[col] = _as_string_categorical(series)
        notes.append(
            f"{axis}['{col}']: mixed {'/'.join(sorted(kinds))} values; "
            "coerced to a string categorical for h5ad"
        )
    return notes


def _as_string_categorical(series: pd.Series) -> pd.Categorical:
    """Categorical of ``str(value)``, with missing values kept missing.

    ``notna`` decides what is missing rather than testing values one at a time, so
    None, ``np.nan`` and ``pd.NA`` are all treated as absent — which is what a
    reindex onto a larger axis produces, and the reason these columns exist.
    """
    values = series.astype(object)
    present = series.notna().to_numpy()
    out = np.array([str(v) if keep else None for v, keep in zip(values, present, strict=True)])
    return pd.Categorical(out)


def _jsonify_unwritable_uns(uns: object) -> list[str]:
    """Replace uns values anndata cannot write with their JSON text.

    Recurses through nested dicts — stages namespace payloads under
    ``uns['cellquorum'][<stage>]``, and scvelo/scanpy nest a couple of levels
    too — and coerces the *smallest* node that cannot be written, so a mostly
    writable namespace keeps its real values. The coercion is
    ``json.dumps(value, default=str)``, recoverable with ``json.loads``.

    What counts as writable is decided by :func:`_writable_uns_value`, and
    getting that judgement wrong is expensive in both directions:

    * Too strict and information is destroyed. Sparse matrices used to be
      coerced here, so ``uns['paga']['connectivities']`` — the PAGA graph
      itself — was stored as ``repr(matrix)`` in every object the engine wrote.
      ``default=str`` means such a coercion never raises; it just silently
      replaces the data with a description of it.
    * Too lenient and the whole write fails. An ``object``-dtype numpy array is
      written as variable-length strings, so one holding dicts raises
      ``TypeError: Can't implicitly convert non-string objects to strings`` —
      and because it aborts the write partway, it costs the entire object, not
      just that key.
    """
    notes: list[str] = []

    def _coerce(mapping: dict, prefix: str) -> None:
        for key in list(mapping.keys()):
            value = mapping[key]
            if _writable_uns_value(value):
                continue
            if isinstance(value, dict):
                restrung = _stringify_keys(value)
                if restrung is not None:
                    mapping[key] = value = restrung
                    notes.append(f"uns['{prefix}{key}']: {len(value)} key(s) converted to strings")
                if all(isinstance(k, str) for k in value):
                    _coerce(value, f"{prefix}{key}/")
                    continue
            if isinstance(value, tuple) and _writable_uns_value(list(value)):
                # Only the container type is wrong; anndata writes lists.
                mapping[key] = list(value)
                notes.append(f"uns['{prefix}{key}']: tuple stored as a list")
                continue
            # ``tolist`` first, or json.dumps hands the array itself to
            # ``default=str`` and stores its repr — data replaced by a
            # description of the data, which is the failure mode this whole
            # function is trying to avoid.
            payload = value.tolist() if isinstance(value, np.ndarray) else value
            mapping[key] = json.dumps(_json_safe(payload), default=str)
            notes.append(f"uns['{prefix}{key}']: stored as JSON text")

    _coerce(uns, "")
    return notes


def _string_keyed(mapping: dict) -> dict:
    """``mapping`` with every key as a string, keeping distinct keys distinct.

    A dict keyed by cluster id is the natural shape for a per-cluster payload, and
    numpy hands those ids over as ``numpy.int64``. h5py needs string group names,
    and ``json.dumps`` — the fallback — rejects numpy integer keys outright with
    ``keys must be str, int, float, bool or None``.

    ``str`` alone is not enough, because it can map two keys onto one name
    (``{1: ..., "1": ...}``) and a plain dict comprehension would then drop an
    entry without a word. Collisions get the trailing underscore
    :func:`safe_h5_key` already uses for the same problem in key names.
    """
    restrung: dict[str, object] = {}
    for key, value in mapping.items():
        name = str(key)
        while name in restrung:
            name += "_"
        restrung[name] = value
    return restrung


def _stringify_keys(mapping: dict) -> dict | None:
    """:func:`_string_keyed`, or None when the keys are already all strings.

    A dict whose only problem is its keys gets the keys fixed and keeps its values
    as real arrays, rather than being flattened wholesale into JSON text.
    """
    if all(isinstance(k, str) for k in mapping):
        return None
    return _string_keyed(mapping)


def _json_safe(value: object) -> object:
    """Recursively make ``value`` acceptable to ``json.dumps`` — keys included.

    ``default=str`` only rescues unserializable *values*; a non-string *key* raises
    with no hook to catch it, which is how a dict keyed by ``numpy.int64`` took
    down a whole run from inside the sanitizer.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in _string_keyed(value).items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _writable_uns_value(value: object) -> bool:
    """Whether anndata can write ``value`` into uns as itself.

    See :func:`_jsonify_unwritable_uns` for why both a false positive and a
    false negative here have already cost real data.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, np.ndarray):
        # The one numpy dtype h5py cannot always take: object arrays go out as
        # variable-length strings, so they must actually hold strings.
        if value.dtype == object:
            return all(isinstance(v, str) for v in value.ravel())
        return value.dtype.kind not in "OMm"
    module = type(value).__module__
    if module.startswith(("numpy", "pandas", "scipy.sparse")):
        return True
    if isinstance(value, list):
        # A list becomes an array, so it must be flat and all one family: all
        # text, or all numbers. Mixing the two is the object-array case again.
        return all(isinstance(v, str) for v in value) or all(
            isinstance(v, int | float | bool) for v in value
        )
    if isinstance(value, dict):
        return all(isinstance(k, str) and _writable_uns_value(v) for k, v in value.items())
    return False


def _first_few(renames: dict[str, str]) -> str:
    """Compact ``old -> new`` preview for a note."""
    items = sorted(renames.items())[:3]
    return ", ".join(f"{old} -> {new}" for old, new in items)
