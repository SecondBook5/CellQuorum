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

import json
from hashlib import sha256
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad

# Bump when the fingerprint construction changes so stale fingerprints from an
# older CellQuorum version never compare equal to freshly-computed ones.
FINGERPRINT_SCHEMA_VERSION = 1


def _stable_json(payload: object) -> str:
    """Return a deterministic JSON string for hashing (sorted keys, no NaN)."""

    return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)


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
        "config": stage_config or {},
        "adata": _adata_signature(adata),
    }
    return _hash_str(_stable_json(payload))


__all__ = ["FINGERPRINT_SCHEMA_VERSION", "compute_input_fingerprint"]
