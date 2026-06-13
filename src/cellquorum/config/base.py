"""Shared base models for CellQuorum configuration schemas."""

from __future__ import annotations

# Import Pydantic primitives for strict runtime validation.
from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """
    Base model for strict CellQuorum configuration schemas.

    CellQuorum uses Hydra and OmegaConf for flexible config composition, but the
    final resolved configuration must be validated strictly before execution.
    This base model forbids unknown fields so spelling mistakes and unsupported
    options fail early instead of silently changing pipeline behavior.
    """

    # Forbid unknown fields in all child configuration models.
    model_config = ConfigDict(extra="forbid")


__all__ = [
    "StrictBaseModel",
]
