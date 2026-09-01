"""Per-stage AnnData checkpoints, so a pipeline can be run one stage at a time.

Why this exists
---------------
``core.resume`` can already skip a completed stage on rerun, but only for stages
that do NOT transform the AnnData — its own docstring says why: the completion
sidecar records artifact paths and fingerprints, not the object state. So a run
could never stop after QC and resume at integration; it had to recompute from the
start every time. That made stage-by-stage verification impractical, which in turn
made it tempting to verify by hand instead of by running the engine.

A checkpoint closes that gap: after a stage, write the object it produced next to
a small sidecar recording which stage produced it and the fingerprint of the inputs
that went in. A later run can load the newest valid checkpoint and start after it.

Opt-in, and off by default
--------------------------
A full AnnData per stage is expensive — hundreds of gigabytes for a 15 GB atlas
across 32 stages — so production runs must not pay for it. Enable it while
developing or verifying a pipeline.

Staleness is the real hazard
----------------------------
A resume feature that silently loads state produced under different settings is
worse than no resume: results become irreproducible in a way nobody notices. So a
checkpoint records a fingerprint of everything upstream that could have shaped it —
:func:`cellquorum.core.fingerprint.compute_upstream_fingerprint` — and a resume
recomputes that same fingerprint and refuses loudly on a mismatch.

It has to be the *upstream* fingerprint and not the input fingerprint that stages
already record. The input fingerprint includes a signature of the stage's input
AnnData, which on resume no longer exists (not rebuilding it is the point of a
checkpoint), so it cannot be recomputed for comparison. The upstream fingerprint is
config-only and therefore computable at both ends. Scoping it to stages at or before
the checkpoint means editing a late-stage parameter does not needlessly invalidate an
early checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad

# v2 added upstream_fingerprint. A v1 checkpoint carries no upstream fingerprint and
# so can never be validated; read_checkpoint_record drops any version it does not
# recognise, which correctly makes those old checkpoints invisible rather than
# loadable-but-unchecked.
CHECKPOINT_SCHEMA_VERSION = 2
_SIDECAR_NAME = "checkpoint.json"
_OBJECT_NAME = "adata.h5ad"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be written, or is present but unusable."""


@dataclass(frozen=True)
class CheckpointRecord:
    """What a checkpoint knows about itself."""

    stage: str
    order: int
    input_fingerprint: str | None
    n_obs: int
    n_vars: int
    path: Path
    # The one a resume can actually check. See the module docstring for why the
    # input fingerprint above cannot serve this purpose.
    upstream_fingerprint: str | None = None

    @property
    def object_path(self) -> Path:
        return self.path / _OBJECT_NAME


def checkpoint_root(paths: object) -> Path:
    """Directory holding all checkpoints for a run."""
    objects = getattr(paths, "objects", None)
    base = Path(objects) if objects else Path(".")
    return base / "checkpoints"


def checkpoint_dir(paths: object, stage: str) -> Path:
    """Directory for one stage's checkpoint."""
    return checkpoint_root(paths) / stage


def should_checkpoint(run_config: object, stage: str) -> bool:
    """Is a checkpoint wanted after ``stage``?

    ``checkpoint_after`` empty or unset means every stage, which is the useful
    default once the feature is switched on at all: the reason to enable it is to
    be able to stop anywhere.
    """
    if not getattr(run_config, "checkpoint", False):
        return False
    wanted = getattr(run_config, "checkpoint_after", None)
    if not wanted:
        return True
    return stage in set(wanted)


def write_checkpoint(
    adata: ad.AnnData | None,
    paths: object,
    *,
    stage: str,
    order: int,
    input_fingerprint: str | None,
    upstream_fingerprint: str | None = None,
) -> CheckpointRecord | None:
    """Write ``adata`` as ``stage``'s checkpoint. None when there is nothing to write.

    Raises CheckpointError on a write failure rather than warning: a checkpoint
    that silently failed to write would send a later run back to the beginning
    with no explanation.
    """
    if adata is None:
        return None
    target = checkpoint_dir(paths, stage)
    target.mkdir(parents=True, exist_ok=True)
    object_path = target / _OBJECT_NAME
    try:
        import anndata

        # Real objects carry nullable string obs columns; without this opt-in the
        # write fails on exactly the objects most worth checkpointing.
        anndata.settings.allow_write_nullable_strings = True
        adata.write_h5ad(object_path)
    except Exception as exc:  # noqa: BLE001 — re-raised as a typed error below
        raise CheckpointError(f"could not write checkpoint for '{stage}': {exc}") from exc

    record = CheckpointRecord(
        stage=stage,
        order=int(order),
        input_fingerprint=input_fingerprint,
        n_obs=int(adata.n_obs),
        n_vars=int(adata.n_vars),
        path=target,
        upstream_fingerprint=upstream_fingerprint,
    )
    (target / _SIDECAR_NAME).write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "stage": record.stage,
                "order": record.order,
                "input_fingerprint": record.input_fingerprint,
                "upstream_fingerprint": record.upstream_fingerprint,
                "n_obs": record.n_obs,
                "n_vars": record.n_vars,
                "object": _OBJECT_NAME,
            },
            indent=2,
        )
        + "\n"
    )
    return record


def read_checkpoint_record(paths: object, stage: str) -> CheckpointRecord | None:
    """Read a stage's checkpoint sidecar, or None when absent/unreadable."""
    target = checkpoint_dir(paths, stage)
    sidecar = target / _SIDECAR_NAME
    if not sidecar.is_file() or not (target / _OBJECT_NAME).is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text())
    except Exception:  # noqa: BLE001 — a corrupt sidecar is simply "no checkpoint"
        return None
    if int(payload.get("schema_version", 0)) != CHECKPOINT_SCHEMA_VERSION:
        return None
    return CheckpointRecord(
        stage=str(payload.get("stage", stage)),
        order=int(payload.get("order", -1)),
        input_fingerprint=payload.get("input_fingerprint"),
        n_obs=int(payload.get("n_obs", -1)),
        n_vars=int(payload.get("n_vars", -1)),
        path=target,
        upstream_fingerprint=payload.get("upstream_fingerprint"),
    )


def available_checkpoints(paths: object) -> list[CheckpointRecord]:
    """Every readable checkpoint, ordered by pipeline order."""
    root = checkpoint_root(paths)
    if not root.is_dir():
        return []
    records = [
        record
        for directory in sorted(root.iterdir())
        if directory.is_dir()
        and (record := read_checkpoint_record(paths, directory.name)) is not None
    ]
    return sorted(records, key=lambda r: r.order)


def resolve_start_checkpoint(
    paths: object,
    *,
    from_stage: str,
    stage_order: dict[str, int],
) -> CheckpointRecord:
    """The checkpoint to resume from when starting at ``from_stage``.

    That is the newest checkpoint strictly BEFORE ``from_stage``: resuming *at* a
    stage means its own inputs must come from the stage before it. Raises
    CheckpointError with the available options when none qualifies, because the
    alternative — silently starting from raw input — would look like a resume and
    produce different numbers.
    """
    if from_stage not in stage_order:
        raise CheckpointError(
            f"unknown stage '{from_stage}'; expected one of {sorted(stage_order)}"
        )
    target_order = stage_order[from_stage]
    candidates = [r for r in available_checkpoints(paths) if r.order < target_order]
    if not candidates:
        existing = [r.stage for r in available_checkpoints(paths)]
        raise CheckpointError(
            f"cannot start at '{from_stage}': no checkpoint exists before it "
            f"(available: {existing or 'none'}). Re-run with run.checkpoint enabled "
            "to create one."
        )
    return candidates[-1]


def load_checkpoint(
    record: CheckpointRecord,
    *,
    expected_fingerprint: str | None = None,
    expected_upstream_fingerprint: str | None = None,
) -> ad.AnnData:
    """Load a checkpoint's AnnData, refusing a fingerprint mismatch.

    ``expected_upstream_fingerprint`` is the check the resume path uses, because it is
    the only one it can recompute (see the module docstring). Passing it and having it
    disagree means a setting at or before this stage changed since the checkpoint was
    written, so the object no longer represents what this run asked for.

    Passing an expected upstream fingerprint against a record that has none is also
    refused: the checkpoint cannot be validated, and loading it unchecked is the exact
    silent-mismatch this guard exists to prevent.

    ``expected_fingerprint`` compares the stage's recorded *input* fingerprint. Either
    check is skipped when its expected value is None, for callers that genuinely
    cannot compute one.
    """
    if expected_upstream_fingerprint is not None:
        if record.upstream_fingerprint is None:
            raise CheckpointError(
                f"checkpoint for '{record.stage}' cannot be validated: it records no "
                "upstream fingerprint, so there is no way to tell whether it was "
                "written under the settings this run is using. Delete it and re-run "
                "the stage to create a checkpoint that can be checked."
            )
        if expected_upstream_fingerprint != record.upstream_fingerprint:
            raise CheckpointError(
                f"checkpoint for '{record.stage}' is stale: it was written with "
                f"upstream fingerprint {record.upstream_fingerprint[:12]}… but this "
                f"run computes {expected_upstream_fingerprint[:12]}…. A setting at or "
                f"before '{record.stage}' — or the input, or the random seed — "
                "changed. Delete the checkpoint to recompute, or revert the change."
            )
    if (
        expected_fingerprint is not None
        and record.input_fingerprint is not None
        and expected_fingerprint != record.input_fingerprint
    ):
        raise CheckpointError(
            f"checkpoint for '{record.stage}' is stale: it was written with input "
            f"fingerprint {record.input_fingerprint[:12]}… but this run computes "
            f"{expected_fingerprint[:12]}…. The config or upstream data changed. "
            "Delete the checkpoint to recompute, or revert the change."
        )
    import anndata

    return anndata.read_h5ad(record.object_path)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "CheckpointRecord",
    "available_checkpoints",
    "checkpoint_dir",
    "checkpoint_root",
    "load_checkpoint",
    "read_checkpoint_record",
    "resolve_start_checkpoint",
    "should_checkpoint",
    "write_checkpoint",
]
