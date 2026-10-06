"""Checkpoint retention: a run keeps only the newest verified checkpoint when asked to.

A full AnnData per stage is tens of gigabytes on a large atlas (27 GB for one fibroblast
run), and nothing reads any checkpoint but the newest one before the stage a resume starts at.
The danger is the opposite one: deleting the checkpoint a resume needs, or deleting before the
replacement is known to be good. So the tests here are mostly about what must NOT be pruned.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest

from cellquorum.config.models import RunConfig
from cellquorum.core.checkpoint import (
    CheckpointError,
    available_checkpoints,
    checkpoint_dir,
    prune_checkpoints,
    resolve_start_checkpoint,
    write_checkpoint,
)
from cellquorum.core.executor import PipelineExecutor

STAGE_ORDER = {"qc": 20, "preprocessing": 30, "integration": 60, "clustering": 80}


def _paths(tmp_path):
    return SimpleNamespace(objects=str(tmp_path / "objects"))


def _adata():
    adata = ad.AnnData(X=np.ones((6, 3), dtype="float32"))
    adata.obs_names = [f"c{i}" for i in range(6)]
    adata.var_names = [f"g{j}" for j in range(3)]
    return adata


def _write(paths, stage):
    return write_checkpoint(
        _adata(), paths, stage=stage, order=STAGE_ORDER[stage], input_fingerprint=None
    )


def _stages(paths):
    return [r.stage for r in available_checkpoints(paths)]


def test_latest_removes_every_checkpoint_before_the_kept_one(tmp_path):
    paths = _paths(tmp_path)
    for stage in ("qc", "preprocessing", "integration"):
        _write(paths, stage)

    pruned = prune_checkpoints(paths, keep_stage="integration", retention="latest")

    assert sorted(pruned) == ["preprocessing", "qc"]
    assert _stages(paths) == ["integration"]


def test_all_removes_nothing(tmp_path):
    paths = _paths(tmp_path)
    for stage in ("qc", "preprocessing"):
        _write(paths, stage)

    assert prune_checkpoints(paths, keep_stage="preprocessing", retention="all") == []
    assert _stages(paths) == ["qc", "preprocessing"]


def test_a_later_checkpoint_than_the_kept_one_is_never_removed(tmp_path):
    # Pruning after stage N must not touch a checkpoint a previous, longer run left for stage N+k.
    paths = _paths(tmp_path)
    for stage in ("qc", "preprocessing", "clustering"):
        _write(paths, stage)

    prune_checkpoints(paths, keep_stage="preprocessing", retention="latest")

    assert _stages(paths) == ["preprocessing", "clustering"]


def test_nothing_is_removed_when_the_kept_checkpoint_is_not_readable(tmp_path):
    # The replacement is not known to be good, so the older one is all a resume has.
    paths = _paths(tmp_path)
    _write(paths, "qc")
    _write(paths, "preprocessing")
    (checkpoint_dir(paths, "preprocessing") / "checkpoint.json").write_text("{not json")

    pruned = prune_checkpoints(paths, keep_stage="preprocessing", retention="latest")

    assert pruned == []
    assert (checkpoint_dir(paths, "qc")).is_dir()


def test_nothing_is_removed_when_the_kept_stage_has_no_checkpoint(tmp_path):
    paths = _paths(tmp_path)
    _write(paths, "qc")

    assert prune_checkpoints(paths, keep_stage="integration", retention="latest") == []
    assert _stages(paths) == ["qc"]


def test_pruning_leaves_a_record_of_what_was_removed_and_why(tmp_path):
    paths = _paths(tmp_path)
    for stage in ("qc", "preprocessing"):
        _write(paths, stage)

    prune_checkpoints(paths, keep_stage="preprocessing", retention="latest")

    record = json.loads((checkpoint_dir(paths, "preprocessing").parent / "pruned.json").read_text())
    assert record["retention"] == "latest"
    assert [entry["stage"] for entry in record["pruned"]] == ["qc"]
    assert record["pruned"][0]["bytes"] > 0
    assert record["kept"] == "preprocessing"


def test_the_record_accumulates_across_successive_prunes(tmp_path):
    paths = _paths(tmp_path)
    _write(paths, "qc")
    _write(paths, "preprocessing")
    prune_checkpoints(paths, keep_stage="preprocessing", retention="latest")
    _write(paths, "integration")
    prune_checkpoints(paths, keep_stage="integration", retention="latest")

    record = json.loads((checkpoint_dir(paths, "qc").parent / "pruned.json").read_text())
    assert [entry["stage"] for entry in record["pruned"]] == ["qc", "preprocessing"]


def test_resume_into_a_pruned_range_explains_the_pruning(tmp_path):
    paths = _paths(tmp_path)
    for stage in ("qc", "preprocessing", "integration"):
        _write(paths, stage)
    prune_checkpoints(paths, keep_stage="integration", retention="latest")

    with pytest.raises(CheckpointError) as caught:
        resolve_start_checkpoint(paths, from_stage="preprocessing", stage_order=STAGE_ORDER)

    message = str(caught.value)
    assert "pruned" in message
    assert "checkpoint_retention" in message
    assert "qc" in message


def test_run_config_defaults_to_keeping_everything():
    assert RunConfig().checkpoint_retention == "all"


def test_run_config_rejects_an_unknown_retention():
    with pytest.raises(ValueError):
        RunConfig(checkpoint_retention="newest")


class _Reporter:
    def __init__(self):
        self.notes = []

    def stage_note(self, stage, note):
        self.notes.append((stage, note))


def _context(paths, adata, retention):
    run = SimpleNamespace(checkpoint=True, checkpoint_after=[], checkpoint_retention=retention)
    config = SimpleNamespace(
        run=run,
        model_dump=lambda: {"run": {"checkpoint": True}, "input": {"h5ad": "x"}, "stages": {}},
    )
    return SimpleNamespace(config=config, paths=paths, adata=adata)


def _after(stage, context, reporter):
    PipelineExecutor._maybe_write_checkpoint(
        SimpleNamespace(),
        context=context,
        planned_stage=SimpleNamespace(name=stage),
        stage_result=SimpleNamespace(input_fingerprint=None),
        reporter=reporter,
    )


def test_the_executor_prunes_earlier_checkpoints_once_the_new_one_is_written(tmp_path):
    paths = _paths(tmp_path)
    reporter = _Reporter()
    _after("qc", _context(paths, _adata(), "latest"), reporter)
    _after("preprocessing", _context(paths, _adata(), "latest"), reporter)

    assert _stages(paths) == ["preprocessing"]
    assert any("pruned" in note for _, note in reporter.notes)


def test_the_executor_keeps_everything_by_default(tmp_path):
    paths = _paths(tmp_path)
    _after("qc", _context(paths, _adata(), "all"), _Reporter())
    _after("preprocessing", _context(paths, _adata(), "all"), _Reporter())

    assert _stages(paths) == ["qc", "preprocessing"]


def test_a_failed_write_never_prunes_the_checkpoint_before_it(tmp_path):
    paths = _paths(tmp_path)
    reporter = _Reporter()
    _after("qc", _context(paths, _adata(), "latest"), reporter)

    class Unwritable:
        n_obs = 1
        n_vars = 1

        def write_h5ad(self, *args, **kwargs):
            raise OSError("disk full")

    _after("preprocessing", _context(paths, Unwritable(), "latest"), reporter)

    assert _stages(paths) == ["qc"]
