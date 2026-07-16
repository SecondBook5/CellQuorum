"""Shared adapter for the thin notebook API (cq.pp / cq.tl / cq.diag).

The notebook namespaces are deliberately thin: each ``cq.pp.qc(adata, ...)``
builds a minimal config + context, runs the SAME registered stage the pipeline
would run, and returns the resulting AnnData. No stage logic is duplicated —
config resolution, skip semantics, and artifact emission all come from the
stage's own ``run``. This keeps the notebook surface honest: what you get
interactively is exactly what the config-driven engine produces.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.core.executor import build_default_stage_registry

if TYPE_CHECKING:
    import anndata as ad

    from cellquorum.core.stage import StageResult


@dataclass
class NotebookStageOutput:
    """
    The return value of a notebook stage call.

    Behaves as the updated AnnData for convenience (``adata`` attribute), while
    also exposing the full ``StageResult`` for artifacts, metrics, notes, and
    warnings.
    """

    adata: ad.AnnData
    result: StageResult

    @property
    def metrics(self) -> dict[str, Any]:
        """Return the stage's structured metrics."""

        return self.result.metrics

    @property
    def warnings(self) -> list[str]:
        """Return the stage's warnings."""

        return list(self.result.warnings)

    @property
    def notes(self) -> list[str]:
        """Return the stage's notes."""

        return list(self.result.notes)


def run_stage(
    stage_name: str,
    adata: ad.AnnData,
    *,
    config: CellQuorumConfig | dict | None = None,
    output_dir: str | Path | None = None,
    **stage_kwargs: Any,
) -> NotebookStageOutput:
    """
    Run one registered stage over an AnnData object, notebook-style.

    Args:
        stage_name: Registry stage name (e.g. ``qc``, ``clustering``).
        adata: Input AnnData. It is not mutated in place unless the stage does;
            the returned object is the stage's output.
        config: Optional base config (a CellQuorumConfig or dict). Stage kwargs
            are merged into this stage's config block, which is also enabled.
        output_dir: Optional directory for any artifacts the stage writes. A
            temporary directory is used when omitted.
        **stage_kwargs: Stage-specific settings merged into the stage's config
            block (e.g. ``mode="report_only"`` for qc).

    Returns:
        A NotebookStageOutput carrying the resulting adata and StageResult.

    Raises:
        KeyError: If ``stage_name`` is not a registered stage.
    """

    registry = build_default_stage_registry()
    stage = registry.get(stage_name)
    if stage is None:
        available = ", ".join(sorted(registry.registered_stage_names()))
        raise KeyError(f"Unknown stage '{stage_name}'. Registered stages: {available}.")

    # Build the effective config: start from the caller's base (if any), then
    # overlay this stage's kwargs into its own config block and enable it.
    base_dict: dict[str, Any]
    if config is None:
        base_dict = {}
    elif isinstance(config, CellQuorumConfig):
        base_dict = config.model_dump()
    else:
        base_dict = dict(config)

    stage_block = dict(base_dict.get(stage_name, {}))
    stage_block.update(stage_kwargs)
    stage_block.setdefault("enabled", True)
    base_dict[stage_name] = stage_block

    # Make sure the stage-selection flag is on so any enabled checks pass.
    stages_block = dict(base_dict.get("stages", {}))
    stages_block[stage_name] = True
    base_dict["stages"] = stages_block

    effective_config = CellQuorumConfig.model_validate(base_dict)

    # Resolve an output directory for artifact-writing stages.
    with _resolve_output_dir(output_dir) as resolved_dir:
        paths = PipelinePaths.from_output_dir(resolved_dir)
        paths.ensure_directories()
        context = PipelineContext(
            config=effective_config,
            paths=paths,
            adata=adata,
            random_seed=effective_config.run.random_seed,
        )
        result = stage.run(context)

    return NotebookStageOutput(adata=result.adata, result=result)


class _resolve_output_dir:
    """Context manager yielding a stable dir, or a self-cleaning temp dir."""

    def __init__(self, output_dir: str | Path | None) -> None:
        self._output_dir = output_dir
        self._tmp: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> Path:
        if self._output_dir is not None:
            return Path(self._output_dir)
        self._tmp = tempfile.TemporaryDirectory(prefix="cellquorum-nb-")
        return Path(self._tmp.name)

    def __exit__(self, *exc: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()


__all__ = ["NotebookStageOutput", "run_stage"]
