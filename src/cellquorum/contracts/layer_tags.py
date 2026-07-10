"""Read and write per-layer provenance tags in ``adata.uns``.

A layer tag records what a layer *is* (its ``kind``) and, for normalized
layers, which named recipe produced it. Contracts read these tags to verify
that a layer named ``lognorm`` actually holds log-normalized values produced by
the expected recipe, rather than trusting the layer name alone.
"""

from __future__ import annotations

from typing import Any

import anndata as ad

# Allowed layer kinds. 'counts' = raw integer counts; 'lognorm' = log-normalized
# expression; 'scaled' = z-scored/scaled; 'imputed' = denoised (e.g. MAGIC).
_ALLOWED_KINDS = {"counts", "lognorm", "scaled", "imputed"}


def set_layer_tag(
    adata: ad.AnnData,
    layer: str,
    *,
    kind: str,
    recipe: str | None = None,
) -> None:
    """
    Record a provenance tag for one layer.

    Args:
        adata: AnnData object to annotate.
        layer: Layer name being tagged.
        kind: Layer kind, one of counts/lognorm/scaled/imputed.
        recipe: Named recipe that produced the layer, when applicable.

    Raises:
        ValueError: If ``kind`` is not an allowed layer kind.
    """

    # Reject unknown layer kinds so tags stay meaningful for contracts.
    if kind not in _ALLOWED_KINDS:
        raise ValueError(f"Unknown layer kind '{kind}'. Allowed: {sorted(_ALLOWED_KINDS)}.")

    # Ensure the cellquorum uns namespace exists.
    cq = adata.uns.setdefault("cellquorum", {})

    # Ensure the layer_tags sub-namespace exists.
    tags = cq.setdefault("layer_tags", {})

    # Store the tag for this layer.
    tags[layer] = {"kind": kind, "recipe": recipe}


def get_layer_tag(adata: ad.AnnData, layer: str) -> dict[str, Any] | None:
    """
    Return the provenance tag for one layer, or None if untagged.

    Args:
        adata: AnnData object to inspect.
        layer: Layer name to look up.

    Returns:
        The tag dictionary (keys ``kind`` and ``recipe``), or None.
    """

    # Walk the uns namespace defensively; any missing level means untagged.
    cq = adata.uns.get("cellquorum", {})
    tags = cq.get("layer_tags", {})
    return tags.get(layer)


def get_normalization_recipe(adata: ad.AnnData) -> str | None:
    """
    Return the normalization recipe recorded by the preprocessing stage.

    Reads the same provenance location that
    ``preprocessing/normalization.py::write_normalization_provenance`` writes,
    so contracts can verify normalization without re-deriving it.

    Args:
        adata: AnnData object to inspect.

    Returns:
        The recipe string, or None if no normalization provenance is present.
    """

    # Navigate the preprocessing provenance defensively.
    cq = adata.uns.get("cellquorum", {})
    preprocessing = cq.get("preprocessing", {})
    normalization = preprocessing.get("normalization", {})
    recipe = normalization.get("recipe")
    return recipe if isinstance(recipe, str) else None


__all__ = ["get_layer_tag", "get_normalization_recipe", "set_layer_tag"]
