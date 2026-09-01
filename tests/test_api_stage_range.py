"""The public API must preserve the run config and honour the stage range.

Two bugs motivated these tests, both of the same shape: an argument that LOOKS
accepted but is discarded, so the run silently does something other than what was
asked.

1. ``quiet=True`` was implemented as ``model_copy(update={"run": {"verbose":
   False}})``. Pydantic's ``update`` does not merge into a nested model — it
   replaces it — so ``config.run`` became a one-key dict and every other run
   setting vanished. The next attribute access raised AttributeError on a dict.

2. ``from_stage``/``until_stage`` were threaded into every ``run_pipeline`` branch
   EXCEPT the plain YAML-path one, which is the branch the CLI uses. So
   ``cellquorum run --until-stage qc config.yaml`` ran all 32 stages while
   reporting that it had been restricted.

Both are silent-wrong-answer bugs, which is why they are tested at the signature
level rather than through a full run.
"""

from __future__ import annotations

import inspect

from cellquorum.api.pipeline import _quieted, run_pipeline
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.pipeline import (
    execute_pipeline_run,
    execute_pipeline_run_from_config_file,
)

# --------------------------------------------------------------------------- #
# quiet must not flatten the run config
# --------------------------------------------------------------------------- #


def test_quiet_keeps_run_a_model_not_a_dict():
    config = _quieted(CellQuorumConfig())
    assert not isinstance(config.run, dict)
    # The attribute that used to blow up first, in build_pipeline_context.
    assert hasattr(config.run, "run_id")


def test_quiet_turns_verbose_off():
    base = CellQuorumConfig.model_validate({"run": {"verbose": True}})
    assert _quieted(base).run.verbose is False


def test_quiet_preserves_every_other_run_setting():
    # The flattening bug's real damage: unrelated settings silently reverting to
    # defaults, so a checkpointed run quietly stopped checkpointing.
    base = CellQuorumConfig.model_validate(
        {"run": {"verbose": True, "checkpoint": True, "checkpoint_after": ["qc"]}}
    )
    quiet = _quieted(base)
    assert quiet.run.checkpoint is True
    assert quiet.run.checkpoint_after == ["qc"]


def test_quiet_leaves_the_original_untouched():
    base = CellQuorumConfig.model_validate({"run": {"verbose": True}})
    _quieted(base)
    assert base.run.verbose is True


# --------------------------------------------------------------------------- #
# the stage range must reach the executor from every entry point
# --------------------------------------------------------------------------- #

_RANGE_ARGS = {"from_stage", "until_stage"}


def test_run_pipeline_accepts_the_stage_range():
    assert _RANGE_ARGS <= set(inspect.signature(run_pipeline).parameters)


def test_both_executors_accept_the_stage_range():
    # The config-file executor is the one that was missing them.
    for function in (execute_pipeline_run, execute_pipeline_run_from_config_file):
        missing = _RANGE_ARGS - set(inspect.signature(function).parameters)
        assert not missing, f"{function.__name__} is missing {sorted(missing)}"


def test_every_execute_call_in_the_api_forwards_the_stage_range():
    """No branch of run_pipeline may drop the stage range.

    Checked by reading the source rather than by exercising all four branches:
    two of them need a YAML file and a real h5ad, and the failure being guarded
    against is a missing keyword argument, which is visible in the text.
    """
    source = inspect.getsource(run_pipeline)
    calls = [line for line in source.splitlines() if "return execute_pipeline_run" in line]
    # Two executor functions x (quiet, non-quiet) x (model, dict, path) branches.
    assert len(calls) >= 4, f"expected several execute calls, found {len(calls)}"
    for argument in _RANGE_ARGS:
        assert source.count(f"{argument}={argument},") == len(
            calls
        ), f"{argument} is forwarded by fewer than all {len(calls)} execute calls"
