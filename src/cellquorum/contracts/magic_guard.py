"""Standalone guard preventing MAGIC-imputed layers from reaching statistics.

MAGIC imputation is valuable for visualization and scoring but inflates
correlations and invalidates differential/statistical inference. Layers written
by an imputation stage are tagged ``kind='imputed'``; statistical stages call
``assert_not_imputed`` (or set ``forbid_imputed=True`` on a DataContract) to
refuse them.
"""

from __future__ import annotations

import anndata as ad

from cellquorum.contracts.exceptions import CellQuorumContractError
from cellquorum.contracts.layer_tags import get_layer_tag


def assert_not_imputed(adata: ad.AnnData, layer: str) -> None:
    """
    Raise if the given layer is tagged as imputed.

    Args:
        adata: Object to inspect.
        layer: Layer name to check.

    Raises:
        CellQuorumContractError: If the layer's tag kind is 'imputed'.
    """

    # Read the tag; an untagged layer is treated as not-known-imputed (permissive).
    tag = get_layer_tag(adata, layer)
    if tag is not None and tag.get("kind") == "imputed":
        raise CellQuorumContractError(
            f"Layer '{layer}' is tagged 'imputed' (e.g. MAGIC); it must not be used as "
            "input to a statistical/differential stage."
        )


__all__ = ["assert_not_imputed"]
