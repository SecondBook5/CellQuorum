"""Declarative, fail-loud data contract validated at stage boundaries.

A DataContract is what a stage says it needs (and promises to produce). It runs
three escalating levels of checks and raises on the first failure:

1. structural  — required layers / obs columns / var names / obsm keys exist
2. semantic    — the expression layer's provenance tag matches the required
                 kind and recipe
3. statistical — the expression layer's values are consistent with its kind
                 (the all-integer 'lognorm' guard, non-negativity, log-range)
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellquorum.config.base import StrictBaseModel
from cellquorum.core.contracts.exceptions import CellQuorumContractError
from cellquorum.core.contracts.layer_tags import get_layer_tag, get_normalization_recipe
from cellquorum.core.contracts.statistical import (
    assert_integer_valued,
    assert_log_range,
    assert_non_integer_or_zero,
)


class DataContract(StrictBaseModel):
    """
    Declarative contract for an AnnData handed across a stage boundary.

    Attributes are all optional; an empty contract validates any object. Stages
    build a contract describing exactly what they consume, then call
    ``validate(adata)`` on entry.
    """

    # Structural requirements.
    required_layers: list[str] = []
    required_obs: list[str] = []
    required_var: list[str] = []
    required_obsm: list[str] = []

    # Semantic + statistical target: the layer whose tag and values are checked.
    expression_layer: str | None = None
    expected_kind: str | None = None
    expected_recipe: str | None = None

    # Guardrail: reject imputed data where statistics are not allowed.
    forbid_imputed: bool = False

    def validate(self, adata: ad.AnnData) -> None:
        """
        Validate an AnnData object against this contract.

        Args:
            adata: Object handed into (or out of) a stage.

        Raises:
            CellQuorumContractError: On the first violated requirement.
        """

        # ---- Level 1: structural ---- #
        self._check_structural(adata)

        # Nothing further to check without a target expression layer.
        if self.expression_layer is None:
            return

        # The target layer must exist (X is addressed by the sentinel 'X').
        matrix = self._resolve_matrix(adata, self.expression_layer)

        # ---- Level 2: semantic (tag) ---- #
        self._check_semantic(adata)

        # ---- Level 3: statistical (values) ---- #
        self._check_statistical(matrix)

    def _check_structural(self, adata: ad.AnnData) -> None:
        """Verify all required containers are present."""

        # Required layers.
        for layer in self.required_layers:
            if layer not in adata.layers:
                raise CellQuorumContractError(
                    f"Required layer '{layer}' is missing. Present: {list(adata.layers)}."
                )

        # Required obs columns.
        for col in self.required_obs:
            if col not in adata.obs.columns:
                raise CellQuorumContractError(
                    f"Required obs column '{col}' is missing. Present: {list(adata.obs.columns)}."
                )

        # Required var names.
        for name in self.required_var:
            if name not in adata.var_names:
                raise CellQuorumContractError(
                    f"Required var/gene '{name}' is absent from var_names."
                )

        # Required obsm embeddings.
        for key in self.required_obsm:
            if key not in adata.obsm:
                raise CellQuorumContractError(
                    f"Required obsm '{key}' is missing. Present: {list(adata.obsm)}."
                )

    def _resolve_matrix(self, adata: ad.AnnData, layer: str) -> np.ndarray | sp.spmatrix:
        """Return the matrix for a layer name, or ``adata.X`` for the 'X' sentinel."""

        # Allow addressing the primary matrix explicitly.
        if layer == "X":
            return adata.X

        # Otherwise the named layer must exist.
        if layer not in adata.layers:
            raise CellQuorumContractError(
                f"Expression layer '{layer}' is missing. Present: {list(adata.layers)}."
            )
        return adata.layers[layer]

    def _check_semantic(self, adata: ad.AnnData) -> None:
        """Verify the target layer's tag matches the expected kind/recipe."""

        # Read the tag (may be None if the layer was never tagged).
        tag = get_layer_tag(adata, self.expression_layer)

        # Imputed guardrail: block imputed data where forbidden.
        if self.forbid_imputed:
            declared_kind = tag["kind"] if tag else None
            if declared_kind == "imputed":
                raise CellQuorumContractError(
                    f"Layer '{self.expression_layer}' is tagged 'imputed'; imputed data is "
                    "forbidden as input to this stage (statistics on imputed values are invalid)."
                )

        # Expected-kind check.
        if self.expected_kind is not None:
            if tag is None:
                raise CellQuorumContractError(
                    f"Layer '{self.expression_layer}' has no provenance tag; expected kind "
                    f"'{self.expected_kind}'."
                )
            if tag["kind"] != self.expected_kind:
                raise CellQuorumContractError(
                    f"Layer '{self.expression_layer}' kind '{tag['kind']}' != expected "
                    f"'{self.expected_kind}'."
                )

        # Expected-recipe check: prefer the layer tag, fall back to preprocessing provenance.
        # Use `is None` (not `or`) so an explicit empty-string recipe is not silently
        # treated as "unset" and does not trigger the fallback.
        if self.expected_recipe is not None:
            recipe = (tag or {}).get("recipe")
            if recipe is None:
                recipe = get_normalization_recipe(adata)
            if recipe != self.expected_recipe:
                raise CellQuorumContractError(
                    f"Layer '{self.expression_layer}' recipe '{recipe}' != expected "
                    f"'{self.expected_recipe}'."
                )

    def _check_statistical(self, matrix: np.ndarray | sp.spmatrix) -> None:
        """Run value-level checks appropriate to the expected kind."""

        # Counts: must be non-negative integers.
        if self.expected_kind == "counts":
            assert_integer_valued(matrix, layer=self.expression_layer)
            return

        # Log-normalized: not all-integer, within log range. Non-negativity is
        # deliberately NOT asserted here: the project default recipe
        # cellquorum_pf_log1p_pf_v1 is a shifted-CLR (centered) transform that
        # legitimately yields small negative values, and the historical
        # raw-counts-in-lognorm bug is caught by the all-integer guard below
        # (raw counts are non-negative, so a non-negativity check would miss it).
        if self.expected_kind == "lognorm":
            assert_non_integer_or_zero(matrix, layer=self.expression_layer)
            assert_log_range(matrix, layer=self.expression_layer)
            return

        # Scaled/imputed/unspecified: scaled data may legitimately be negative,
        # so non-negativity is not required and no statistical check is applied.


__all__ = ["DataContract"]
