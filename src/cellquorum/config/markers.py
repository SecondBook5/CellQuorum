"""Named marker gene panels, resolved once and referenced by name anywhere.

A dataset declares its gene panels of interest (lineage markers, contamination
panels, signatures) once under ``markers.panels``; stages reference them by name.
Unknown panel names fail loud so a typo cannot silently score an empty gene set.
"""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel
from cellquorum.core.exceptions import CellQuorumConfigError


class MarkersConfig(StrictBaseModel):
    """Named gene panels addressable by name."""

    # Map panel name -> list of gene symbols.
    panels: dict[str, list[str]] = {}

    def names(self) -> list[str]:
        """Return the registered panel names."""

        # Return the panel keys as a list.
        return list(self.panels.keys())

    def panel(self, name: str) -> list[str]:
        """
        Return one named panel's gene list, or raise if the name is unknown.

        Args:
            name: Panel name.

        Returns:
            The gene symbols for the panel.

        Raises:
            CellQuorumConfigError: If the panel name is not registered.
        """

        # Fail loud on an unknown panel name (guards against typos/stale refs).
        if name not in self.panels:
            raise CellQuorumConfigError(
                f"Unknown marker panel '{name}'. Registered panels: {self.names()}."
            )
        return list(self.panels[name])


__all__ = ["MarkersConfig"]
