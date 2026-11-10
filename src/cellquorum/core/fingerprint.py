"""Deterministic stage input fingerprints for resume/caching.

A stage's ``input_fingerprint`` answers one question on rerun: "are the inputs
to this stage the same as last time?" It is intentionally *conservative and
cheap* — it hashes the resolved stage config, the random seed, and the salient
shape/identity of the input AnnData (dimensions, var names, and which
layers/obsm/obs columns are present), but NOT the full expression matrices.
Hashing whole matrices would be slow and is unnecessary: within a single
project the upstream data is stable, and any config or dimensionality change
already flips the fingerprint. This is the key that stage-level resume compares.
"""

from __future__ import annotations

import csv
import glob
import json
from hashlib import file_digest, sha256
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad

# Bump when the fingerprint construction changes so stale fingerprints from an
# older CellQuorum version never compare equal to freshly-computed ones.
#
# v2 stopped hashing config keys whose value is None (see ``_without_unset``).
# Checkpoints written under v1 therefore cannot be compared against v2
# fingerprints; the version is recorded in each checkpoint sidecar so a resume can
# say *that* rather than blaming a setting nobody changed.
#
# v3 stopped hashing any ``outputs`` sub-block (see ``_without_outputs``). A new
# output-writing toggle (e.g. ``qc.outputs.removed_cells``) is a deliberate,
# non-None default, so v2's None-only rule could not catch it: adding the field
# flipped every downstream checkpoint's fingerprint on upgrade even though nothing
# QC computed changed. Checkpoints written under v2 cannot be compared against v3
# fingerprints for the same reason v1-vs-v2 couldn't.
# v5 also includes source files listed in the library manifest.
FINGERPRINT_SCHEMA_VERSION = 5


def _stable_json(payload: object) -> str:
    """Return a deterministic JSON string for hashing (sorted keys, no NaN)."""

    return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)


def _without_unset(value: object) -> object:
    """Drop dict keys whose value is None, recursively.

    Why a fingerprint must not see them
    -----------------------------------
    What gets hashed is the *resolved* config, and resolution fills in every field
    a config model declares — including the ones nobody set, which arrive as None.
    So adding one new optional setting to any model changes the resolved config of
    every run that never mentions it, and every checkpoint on disk becomes stale on
    upgrade. That happened: adding ``input.subset.require_agreement`` invalidated a
    finished run's checkpoints, and the guard reported "a setting changed" for a
    setting that did not exist when the checkpoint was written.

    Collapsing None to absent is not a leniency. For a resolved config the two say
    the same thing — this was not specified — so a fingerprint that distinguishes
    them is reporting a difference that has no effect on any result. A value that
    genuinely changes, including one explicitly cleared back to the default, still
    changes the hash: ``{"max_mito": 10}`` and ``{}`` do not collide.

    A sub-block left holding nothing goes too, for the same reason and one step out: a
    whole new optional *block* resolves to a dict of Nones, not to a single None, so
    pruning its fields is only half the job. ``{"subset": {"obs_key": None}}`` has to
    reach the same hash as no subset block at all, because it says the same thing.

    Notes:
        None *elements* of a list are kept, and an empty list is kept. There, position
        is meaning: dropping an element would shift every later one and silently equate
        two different lists, and an empty list is a value the config actually stated
        rather than something pruning produced.
    """

    if isinstance(value, dict):
        pruned = {key: _without_unset(item) for key, item in value.items()}
        # Typed emptiness, not truthiness: `if not item` would also drop 0, False and
        # "" — settings that were deliberately chosen and change what a stage does.
        return {
            key: item
            for key, item in pruned.items()
            if item is not None and not (isinstance(item, dict) and len(item) == 0)
        }
    if isinstance(value, list | tuple):
        return [_without_unset(item) for item in value]
    return value


def _without_outputs(value: object) -> object:
    """Drop every ``outputs`` sub-block, recursively.

    Why a fingerprint must not see it
    ----------------------------------
    An ``outputs`` block (e.g. ``QCOutputConfig``) controls what gets WRITTEN to
    disk -- tables, figures, an extra CSV -- never what a stage COMPUTES. That is
    the same reasoning ``compute_upstream_fingerprint`` already applies to the run
    block's non-seed fields ("output paths, verbosity, the checkpoint flags
    themselves -- cannot change any result"). It applies just as much to a stage's
    own output toggles, and unlike ``_without_unset`` it cannot be satisfied by
    pruning None: a new output toggle is a deliberate, non-None default (adding
    ``qc.outputs.removed_cells: bool = True`` is what exposed this), so it survives
    the None-only rule and still flips every downstream checkpoint on upgrade even
    though nothing QC computed changed.
    """

    if isinstance(value, dict):
        return {key: _without_outputs(item) for key, item in value.items() if key != "outputs"}
    if isinstance(value, list | tuple):
        return [_without_outputs(item) for item in value]
    return value


def _hash_str(text: str) -> str:
    """Return the hex SHA-256 digest of a UTF-8 string."""

    digest = sha256()
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _adata_signature(adata: ad.AnnData | None) -> dict[str, object]:
    """
    Build a cheap structural signature of an AnnData object.

    Captures shape, the ordered var-name digest, and the set of present
    layer/obsm/obs-column keys. Deliberately excludes matrix values.
    """

    if adata is None:
        return {"present": False}

    # Digest the ordered var names so a gene-space change flips the fingerprint
    # without storing every gene name in the fingerprint payload.
    var_names_digest = _hash_str("\0".join(str(v) for v in adata.var_names))

    return {
        "present": True,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "var_names_digest": var_names_digest,
        "layers": sorted(map(str, adata.layers.keys())),
        "obsm": sorted(map(str, adata.obsm.keys())),
        "obs_columns": sorted(map(str, adata.obs.columns)),
    }


def compute_input_fingerprint(
    *,
    stage_name: str,
    stage_config: dict | None,
    adata: ad.AnnData | None,
    random_seed: int | None,
) -> str:
    """
    Compute a deterministic input fingerprint for a stage.

    Args:
        stage_name: Registry stage name.
        stage_config: Resolved stage config sub-block (plain dict) or None.
        adata: Input AnnData for the stage (may be None before ingestion).
        random_seed: Run-level random seed.

    Returns:
        Hex SHA-256 fingerprint string.
    """

    payload = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "stage_name": stage_name,
        "random_seed": random_seed,
        "config": _without_outputs(_without_unset(stage_config or {})),
        "adata": _adata_signature(adata),
    }
    return _hash_str(_stable_json(payload))


def compute_upstream_fingerprint(
    *,
    config: dict,
    stage_order: dict[str, int],
    through_stage: str,
) -> str:
    """Hash everything that could have shaped the object a stage handed downstream.

    Why this exists separately from :func:`compute_input_fingerprint`
    ----------------------------------------------------------------
    The input fingerprint includes a signature of the stage's *input AnnData*, so it
    can only be computed while the pipeline is standing at that stage. On resume that
    object no longer exists — not having to rebuild it is the whole point of a
    checkpoint — so the input fingerprint is unrecomputable by construction and
    cannot be used to validate a checkpoint you are about to load.

    This one is deliberately config-only, which makes it computable at BOTH ends:
    when the checkpoint is written, and again when a later run resumes on it. It
    covers the random seed, the input spec, and the settings of every stage at or
    before ``through_stage`` — precisely the things that could have changed what the
    checkpoint contains.

    Scoped to upstream stages on purpose. Hashing the whole config would invalidate
    an early checkpoint whenever a late stage's parameter changed, which would make
    the stage-by-stage loop recompute constantly for no safety gain: a cell-cell
    communication setting cannot retroactively alter what QC produced.

    The input file, manifest and configured upstream reference files are content hashed.
    A fresh content hash also catches same-length rewrites on coarse-timestamp filesystems.
    """
    through_order = stage_order.get(through_stage)
    if through_order is None:
        raise KeyError(f"unknown stage '{through_stage}'; cannot scope an upstream fingerprint")

    upstream = sorted(name for name, order in stage_order.items() if order <= through_order)

    # Enablement flags matter even for stages with no config block of their own:
    # turning ambient correction on or off changes what QC received.
    stages_block = config.get("stages") or {}
    enablement = {name: stages_block.get(name) for name in upstream if name in stages_block}

    # Only the seed is taken from the run block. The rest of it — output paths,
    # verbosity, the checkpoint flags themselves — cannot change any result, and
    # including them would refuse resumes for cosmetic reasons.
    run_block = config.get("run") or {}

    return _hash_str(
        _stable_json(
            {
                "schema_version": FINGERPRINT_SCHEMA_VERSION,
                "through_stage": through_stage,
                "through_order": through_order,
                "random_seed": run_block.get("random_seed"),
                "study_design": _without_unset(config.get("design")),
                "cohort": _without_unset(config.get("cohort")),
                "input": _input_signature(config.get("input")),
                "manifest": _resource_digest((config.get("paths") or {}).get("manifest")),
                "library_resources": _library_resources(config, upstream, enablement),
                "upstream_resources": {
                    n: _stage_resources(config.get(n), n) for n in upstream if enablement.get(n)
                },
                "stage_enablement": _without_unset(enablement),
                "stage_config": _without_outputs(
                    _without_unset({n: config.get(n) for n in upstream if n in config})
                ),
            }
        )
    )


def _digest_file(path: Path, identity: tuple[int, int, int, int]) -> str:
    with path.open("rb") as handle:
        digest = file_digest(handle, "sha256").hexdigest()
    stat = path.stat()
    if identity != (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
        raise ValueError(f"Input changed while computing its SHA256: {path}")
    return digest


def _resource_digest(value: object) -> str | None:
    if not isinstance(value, Path | str) or not value:
        return None
    path = Path(value).resolve()
    if not path.is_file():
        return None
    stat = path.stat()
    return _digest_file(path, (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def resource_checksum(value: Path) -> str | None:
    """Hash a file or the selected components of a 10x Matrix Market directory."""
    if not value.is_dir():
        return _resource_digest(value)
    names = ("matrix.mtx", "barcodes.tsv", "features.tsv", "genes.tsv")
    files = {
        p.name: _resource_digest(p)
        for p in sorted(value.iterdir())
        if p.name.removesuffix(".gz") in names
    }
    return _hash_str(_stable_json(files)) if files else None


def _library_resources(config: dict, upstream: list[str], enabled: dict) -> dict:
    manifest = (config.get("paths") or {}).get("manifest")
    active = {name for name in upstream if enabled.get(name)}
    if (
        not manifest
        or not Path(manifest).is_file()
        or not active.intersection({"ambient_correction", "trajectory", "qc_splice_metrics"})
    ):
        return {}
    resources = {}
    keys: set[str] = set()
    if "ambient_correction" in active:
        keys.update(("raw_matrix_path", "filtered_matrix_path"))
    if active.intersection({"trajectory", "qc_splice_metrics"}):
        keys.update(("loom_path", "gtf_path", "bam_path"))
    source = Path(manifest)
    base_dir = source.parent if config["paths"].get("relative_to_config") else Path(".")
    with source.open(newline="", encoding="utf-8-sig") as stream:
        sep = "\t" if Path(manifest).suffix.lower() == ".tsv" else ","
        for row in csv.DictReader(stream, delimiter=sep):
            paths = {base_dir / row[key] for key in keys if row.get(key)}
            if "ambient_correction" in active and row.get("cellranger_path"):
                base = (
                    base_dir
                    / Path((config.get("ambient_correction") or {}).get("cellranger_root") or ".")
                    / row["cellranger_path"]
                )
                outs = base / "outs" if (base / "outs").is_dir() else base
                for key, name in (
                    ("raw_matrix_path", "raw_feature_bc_matrix"),
                    ("filtered_matrix_path", "filtered_feature_bc_matrix"),
                ):
                    if not row.get(key):
                        h5 = outs / (name + ".h5")
                        paths.add(h5 if h5.is_file() else outs / name)
            resources.update({str(path): resource_checksum(path) for path in paths})
    return resources


def _stage_resources(block: object, stage: str) -> dict[str, str | None]:
    if not isinstance(block, dict):
        return {}
    fields = {
        "grn": ("tfs_path", "motifs_path", "rankings_glob"),
        "reference_mapping": ("atlas_h5ad",),
        "cell_cell_communication": (
            "nichenet_lr_network",
            "nichenet_ligand_target_matrix",
            "nichenet_weighted_networks",
        ),
        "enrichment": ("gmt_path",),
    }
    resources = {}
    if stage == "trajectory":
        generation = (block.get("velocity") or {}).get("generation") or {}
        gtf = generation.get("gtf_path")
        if gtf:
            resources[str(gtf)] = _resource_digest(gtf)
    for entry in [block, *block.get("methods", [])]:
        if not isinstance(entry, dict):
            continue
        for key in fields.get(stage, ()):
            value = entry.get(key)
            if isinstance(value, str) and value:
                for filename in sorted(glob.glob(value)) if key.endswith("_glob") else [value]:
                    resources[filename] = _resource_digest(filename)
    return resources


def _input_signature(input_block: object) -> dict[str, object]:
    """Include both the declared input specification and the content of the count-bearing file."""
    if not isinstance(input_block, dict):
        return {"present": False}
    return {
        "present": True,
        "spec": _without_unset(input_block),
        "sha256": _resource_digest(input_block.get("h5ad")),
    }


__all__ = [
    "FINGERPRINT_SCHEMA_VERSION",
    "compute_input_fingerprint",
    "compute_upstream_fingerprint",
]
