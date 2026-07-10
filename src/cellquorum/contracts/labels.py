"""Canonical label and condition token validation.

Two lekc bug classes came from token drift: scripts keyed on a legacy label
column (``kc_named``) that no longer existed, and filtered on a hardcoded
condition literal (``'LE'``) when the real value was ``'Lymphedema'`` — both
silently selected zero cells. A LabelContract turns those silent empties into
hard errors.
"""

from __future__ import annotations

import anndata as ad

from cellquorum.config.base import StrictBaseModel
from cellquorum.contracts.exceptions import CellQuorumContractError


class LabelContract(StrictBaseModel):
    """
    Validate that canonical label/condition tokens exist in an AnnData.
    """

    # The obs column holding cell-type / state labels.
    label_col: str

    # Canonical label tokens that must all be present.
    expected_labels: list[str] = []

    # The obs column holding condition (optional).
    condition_col: str | None = None

    # Canonical condition tokens that must all be present.
    expected_conditions: list[str] = []

    @staticmethod
    def _present_tokens(adata: ad.AnnData, col: str) -> set[str]:
        """
        Return an obs column's distinct values as strings.

        Centralizing the str-coercion here keeps ``validate`` and ``select``
        in lockstep: a token that ``validate`` accepts is guaranteed selectable,
        because both sides compute "present values" the same way (categorical and
        object dtypes compare identically after coercion).

        Args:
            adata: Object whose obs column is inspected.
            col: obs column name.

        Returns:
            Set of the column's distinct values coerced to str.
        """

        # Coerce distinct values to str so categorical/object dtypes compare alike.
        return set(map(str, adata.obs[col].unique()))

    def validate(self, adata: ad.AnnData) -> None:
        """
        Verify the columns exist and all expected tokens are present.

        Args:
            adata: Object to validate.

        Raises:
            CellQuorumContractError: If a column is missing or a token is absent.
        """

        # The label column must exist.
        if self.label_col not in adata.obs.columns:
            raise CellQuorumContractError(
                f"Label column '{self.label_col}' is missing. "
                f"Present: {list(adata.obs.columns)}."
            )

        # Every expected label token must be present in the column.
        present_labels = self._present_tokens(adata, self.label_col)
        for token in self.expected_labels:
            if token not in present_labels:
                raise CellQuorumContractError(
                    f"Expected label '{token}' absent from column '{self.label_col}'. "
                    f"Present: {sorted(present_labels)}."
                )

        # Condition column + tokens, when specified.
        if self.condition_col is not None:
            if self.condition_col not in adata.obs.columns:
                raise CellQuorumContractError(
                    f"Condition column '{self.condition_col}' is missing. "
                    f"Present: {list(adata.obs.columns)}."
                )
            present_conditions = self._present_tokens(adata, self.condition_col)
            for token in self.expected_conditions:
                if token not in present_conditions:
                    raise CellQuorumContractError(
                        f"Expected condition '{token}' absent from column "
                        f"'{self.condition_col}'. Present: {sorted(present_conditions)}."
                    )

    def select(self, adata: ad.AnnData, label: str) -> ad.AnnData:
        """
        Return the subset of cells with one canonical label.

        Args:
            adata: Object to subset.
            label: Canonical label token.

        Returns:
            The subset AnnData (a copy).

        Raises:
            CellQuorumContractError: If the column is missing or the label is
                absent (guards against silent empty selections).
        """

        # Column must exist.
        if self.label_col not in adata.obs.columns:
            raise CellQuorumContractError(f"Label column '{self.label_col}' is missing.")

        # Label must be a real value in the column (same coercion as validate).
        present = self._present_tokens(adata, self.label_col)
        if label not in present:
            raise CellQuorumContractError(
                f"Label '{label}' absent from column '{self.label_col}'; refusing to "
                f"return an empty subset. Present: {sorted(present)}."
            )

        # Return the subset copy.
        mask = (adata.obs[self.label_col].astype(str) == label).to_numpy()
        return adata[mask].copy()


__all__ = ["LabelContract"]
