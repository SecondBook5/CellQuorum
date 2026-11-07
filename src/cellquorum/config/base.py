"""Shared base models for CellQuorum configuration schemas."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict


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


def _check_method_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require every ``methods:`` entry to name its method with a ``method:`` key.

    An entry without it is merged onto the stage's own settings, so the stage quietly runs its
    default method instead (a differential-abundance entry written ``{name: proportion_ttest}`` ran
    ``milo``, which was then skipped for lack of R, and the run reported success). A wrong key must
    fail at config load, before any stage runs.
    """
    for index, entry in enumerate(entries):
        if not str(entry.get("method", "")).strip():
            raise ValueError(
                f"methods[{index}] needs a `method:` key naming the method (it has keys "
                f"{sorted(entry) or 'none'}). Without it the stage would run its default method "
                "instead of the one you meant."
            )
    return entries


#: A ``methods:`` list: each entry is a dict that names its method with ``method:``.
MethodEntries = Annotated[list[dict[str, Any]], AfterValidator(_check_method_entries)]


__all__ = [
    "MethodEntries",
    "StrictBaseModel",
]
