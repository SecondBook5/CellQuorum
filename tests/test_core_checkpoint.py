"""Per-stage checkpoints: round-trip, ordering, and the staleness guard.

The staleness guard is the load-bearing test here. A checkpoint that silently
loads state produced under a different config is worse than no checkpoint: the
run looks like it resumed, and the numbers quietly come from settings nobody
chose. So the mismatch case is tested more thoroughly than the happy path.
"""

from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest

from cellquorum.core.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    available_checkpoints,
    checkpoint_root,
    load_checkpoint,
    read_checkpoint_record,
    resolve_start_checkpoint,
    should_checkpoint,
    write_checkpoint,
)
from cellquorum.core.fingerprint import FINGERPRINT_SCHEMA_VERSION

STAGE_ORDER = {"qc": 20, "preprocessing": 30, "integration": 60, "clustering": 80}


def _paths(tmp_path):
    return SimpleNamespace(objects=str(tmp_path / "objects"))


def _adata(n_obs=12, n_vars=5, seed=0):
    rng = np.random.default_rng(seed)
    adata = ad.AnnData(X=rng.normal(size=(n_obs, n_vars)).astype("float32"))
    adata.obs_names = [f"cell{i}" for i in range(n_obs)]
    adata.var_names = [f"gene{j}" for j in range(n_vars)]
    adata.obs["batch"] = ["a"] * (n_obs // 2) + ["b"] * (n_obs - n_obs // 2)
    return adata


# --------------------------------------------------------------------------- #
# should_checkpoint
# --------------------------------------------------------------------------- #


def test_disabled_by_default_writes_nothing():
    # Production runs must not pay the cost of an AnnData per stage.
    run = SimpleNamespace(checkpoint=False, checkpoint_after=[])
    assert should_checkpoint(run, "qc") is False


def test_enabled_with_empty_list_checkpoints_every_stage():
    # The reason to switch it on is to be able to stop anywhere, so "on" with no
    # explicit list means every stage rather than none.
    run = SimpleNamespace(checkpoint=True, checkpoint_after=[])
    assert should_checkpoint(run, "qc") is True
    assert should_checkpoint(run, "clustering") is True


def test_explicit_list_narrows_to_named_stages():
    run = SimpleNamespace(checkpoint=True, checkpoint_after=["qc", "integration"])
    assert should_checkpoint(run, "qc") is True
    assert should_checkpoint(run, "clustering") is False


def test_missing_attributes_are_treated_as_disabled():
    # A config object from an older schema must not crash the executor.
    assert should_checkpoint(SimpleNamespace(), "qc") is False


# --------------------------------------------------------------------------- #
# write / read round-trip
# --------------------------------------------------------------------------- #


def test_round_trip_preserves_matrix_and_obs(tmp_path):
    paths = _paths(tmp_path)
    adata = _adata()
    record = write_checkpoint(adata, paths, stage="qc", order=20, input_fingerprint="fp-abc123")
    assert record is not None
    assert record.object_path.is_file()

    loaded = load_checkpoint(record, expected_fingerprint="fp-abc123")
    np.testing.assert_allclose(np.asarray(loaded.X), np.asarray(adata.X))
    assert list(loaded.obs_names) == list(adata.obs_names)
    assert list(loaded.obs["batch"]) == list(adata.obs["batch"])


def test_sidecar_records_stage_order_and_shape(tmp_path):
    paths = _paths(tmp_path)
    write_checkpoint(_adata(n_obs=7, n_vars=3), paths, stage="qc", order=20, input_fingerprint="fp")
    record = read_checkpoint_record(paths, "qc")
    assert record is not None
    assert (record.stage, record.order, record.n_obs, record.n_vars) == ("qc", 20, 7, 3)


def test_write_with_no_adata_returns_none(tmp_path):
    # A side-effect-only stage has nothing to checkpoint; that is not an error.
    assert (
        write_checkpoint(None, _paths(tmp_path), stage="qc", order=20, input_fingerprint=None)
        is None
    )


def test_nullable_string_obs_does_not_break_the_write(tmp_path):
    # Real run objects carry pandas nullable string columns, which the h5ad writer
    # refuses unless opted in. The objects most worth checkpointing are exactly
    # these, so the writer must handle them.
    import pandas as pd

    adata = _adata()
    adata.obs["sample_id"] = pd.array([f"s{i}" for i in range(adata.n_obs)], dtype="string")
    record = write_checkpoint(adata, _paths(tmp_path), stage="qc", order=20, input_fingerprint="fp")
    assert record is not None
    assert load_checkpoint(record).n_obs == adata.n_obs


def test_missing_checkpoint_reads_as_none(tmp_path):
    assert read_checkpoint_record(_paths(tmp_path), "qc") is None


def test_corrupt_sidecar_reads_as_none_rather_than_raising(tmp_path):
    paths = _paths(tmp_path)
    write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint="fp")
    (checkpoint_root(paths) / "qc" / "checkpoint.json").write_text("{not json")
    assert read_checkpoint_record(paths, "qc") is None


def test_future_schema_version_is_ignored(tmp_path):
    # A checkpoint from a newer engine must be ignored, not misread.
    import json

    paths = _paths(tmp_path)
    write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint="fp")
    sidecar = checkpoint_root(paths) / "qc" / "checkpoint.json"
    payload = json.loads(sidecar.read_text())
    payload["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 1
    sidecar.write_text(json.dumps(payload))
    assert read_checkpoint_record(paths, "qc") is None


# --------------------------------------------------------------------------- #
# the staleness guard
# --------------------------------------------------------------------------- #


def test_fingerprint_mismatch_refuses_to_load(tmp_path):
    paths = _paths(tmp_path)
    record = write_checkpoint(
        _adata(), paths, stage="qc", order=20, input_fingerprint="fp-original"
    )
    with pytest.raises(CheckpointError, match="stale"):
        load_checkpoint(record, expected_fingerprint="fp-changed")


def test_mismatch_message_names_both_fingerprints_and_the_remedy(tmp_path):
    # The error has to be actionable: which checkpoint, and what to do about it.
    paths = _paths(tmp_path)
    record = write_checkpoint(
        _adata(), paths, stage="integration", order=60, input_fingerprint="aaaaaaaaaaaa1111"
    )
    with pytest.raises(CheckpointError) as excinfo:
        load_checkpoint(record, expected_fingerprint="bbbbbbbbbbbb2222")
    message = str(excinfo.value)
    assert "integration" in message
    assert "aaaaaaaaaaaa" in message
    assert "bbbbbbbbbbbb" in message
    assert "Delete the checkpoint" in message


# --------------------------------------------------------------------------- #
# the upstream fingerprint — the guard the resume path can actually apply
# --------------------------------------------------------------------------- #


def test_upstream_fingerprint_mismatch_refuses_to_load(tmp_path):
    paths = _paths(tmp_path)
    record = write_checkpoint(
        _adata(),
        paths,
        stage="qc",
        order=20,
        input_fingerprint="fp",
        upstream_fingerprint="up-original",
    )
    with pytest.raises(CheckpointError, match="stale"):
        load_checkpoint(record, expected_upstream_fingerprint="up-changed")


def test_upstream_mismatch_message_names_the_stage_and_the_remedy(tmp_path):
    paths = _paths(tmp_path)
    record = write_checkpoint(
        _adata(),
        paths,
        stage="integration",
        order=60,
        input_fingerprint="fp",
        upstream_fingerprint="cccccccccccc3333",
    )
    with pytest.raises(CheckpointError) as excinfo:
        load_checkpoint(record, expected_upstream_fingerprint="dddddddddddd4444")
    message = str(excinfo.value)
    assert "integration" in message
    assert "cccccccccccc" in message
    assert "dddddddddddd" in message
    assert "Delete the checkpoint" in message


def test_matching_upstream_fingerprint_loads(tmp_path):
    paths = _paths(tmp_path)
    record = write_checkpoint(
        _adata(),
        paths,
        stage="qc",
        order=20,
        input_fingerprint="fp",
        upstream_fingerprint="up-same",
    )
    assert load_checkpoint(record, expected_upstream_fingerprint="up-same").n_obs == 12


def test_checkpoint_without_upstream_fingerprint_is_refused_when_checking(tmp_path):
    """Unvalidatable is not the same as valid.

    Loading a checkpoint that records no upstream fingerprint while the caller HAS one
    to compare would be exactly the silent mismatch this guard exists to prevent, so
    it is refused rather than waved through.
    """
    paths = _paths(tmp_path)
    record = write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint="fp")
    assert record.upstream_fingerprint is None
    with pytest.raises(CheckpointError, match="cannot be validated"):
        load_checkpoint(record, expected_upstream_fingerprint="anything")


def test_upstream_fingerprint_survives_the_sidecar_round_trip(tmp_path):
    paths = _paths(tmp_path)
    write_checkpoint(
        _adata(),
        paths,
        stage="qc",
        order=20,
        input_fingerprint="fp",
        upstream_fingerprint="up-persisted",
    )
    assert read_checkpoint_record(paths, "qc").upstream_fingerprint == "up-persisted"


def test_matching_fingerprint_loads(tmp_path):
    paths = _paths(tmp_path)
    record = write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint="fp-same")
    assert load_checkpoint(record, expected_fingerprint="fp-same").n_obs == 12


def test_no_expected_fingerprint_skips_the_check(tmp_path):
    # Callers that genuinely cannot compute a fingerprint may still load.
    paths = _paths(tmp_path)
    record = write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint="fp")
    assert load_checkpoint(record, expected_fingerprint=None).n_obs == 12


def test_checkpoint_without_recorded_fingerprint_is_loadable(tmp_path):
    # Written before fingerprints were available: loadable, just unverified.
    paths = _paths(tmp_path)
    record = write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint=None)
    assert load_checkpoint(record, expected_fingerprint="anything").n_obs == 12


# --------------------------------------------------------------------------- #
# whose fault is it — an engine upgrade or a setting
# --------------------------------------------------------------------------- #
#
# Both refuse the checkpoint, and both should. What differs is the message, and the
# message is the whole value of the guard: told "a setting changed", someone goes
# looking through a config for an edit they never made.


def _age_the_fingerprint_schema(paths, stage: str) -> None:
    """Rewrite a sidecar as if an older engine's fingerprint scheme had written it."""
    import json

    sidecar = checkpoint_root(paths) / stage / "checkpoint.json"
    payload = json.loads(sidecar.read_text())
    payload["fingerprint_schema_version"] = FINGERPRINT_SCHEMA_VERSION - 1
    sidecar.write_text(json.dumps(payload))


def test_an_older_fingerprint_schema_is_reported_as_an_upgrade_not_a_setting_change(tmp_path):
    paths = _paths(tmp_path)
    write_checkpoint(
        _adata(), paths, stage="qc", order=20, input_fingerprint="fp", upstream_fingerprint="up"
    )
    _age_the_fingerprint_schema(paths, "qc")
    record = read_checkpoint_record(paths, "qc")

    with pytest.raises(CheckpointError) as excinfo:
        load_checkpoint(record, expected_upstream_fingerprint="up")
    message = str(excinfo.value)
    assert "not comparable" in message
    assert "engine upgrade" in message
    assert "Delete the checkpoint" in message
    # The old message would have been raised even on identical hashes, and would have
    # accused a setting at or before the stage of changing.
    assert "A setting" not in message


def test_an_older_fingerprint_schema_is_refused_even_when_the_hashes_happen_to_match(tmp_path):
    """Equal hashes from two different constructions prove nothing.

    They can only coincide, so accepting them would load a checkpoint whose settings
    were never actually checked — the silent mismatch the guard exists to prevent.
    """
    paths = _paths(tmp_path)
    write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint="fp-identical")
    _age_the_fingerprint_schema(paths, "qc")
    record = read_checkpoint_record(paths, "qc")

    with pytest.raises(CheckpointError, match="not comparable"):
        load_checkpoint(record, expected_fingerprint="fp-identical")


def test_an_older_fingerprint_schema_still_loads_when_nothing_is_being_compared(tmp_path):
    # The object itself is fine; only the hashes are incomparable. A caller with no
    # fingerprint to check was never relying on one, so refusing would break
    # `--from-stage` for no safety gained.
    paths = _paths(tmp_path)
    write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint="fp")
    _age_the_fingerprint_schema(paths, "qc")
    record = read_checkpoint_record(paths, "qc")
    assert load_checkpoint(record).n_obs == 12


def test_a_sidecar_with_no_recorded_fingerprint_schema_is_read_as_the_first_one(tmp_path):
    # Every sidecar written before the field existed came from schema v1. Defaulting
    # to the current version would claim those are comparable when they are not.
    import json

    paths = _paths(tmp_path)
    write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint="fp")
    sidecar = checkpoint_root(paths) / "qc" / "checkpoint.json"
    payload = json.loads(sidecar.read_text())
    del payload["fingerprint_schema_version"]
    sidecar.write_text(json.dumps(payload))

    assert read_checkpoint_record(paths, "qc").fingerprint_schema_version == 1


def test_the_fingerprint_schema_survives_the_sidecar_round_trip(tmp_path):
    paths = _paths(tmp_path)
    write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint="fp")
    record = read_checkpoint_record(paths, "qc")
    assert record.fingerprint_schema_version == FINGERPRINT_SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# resolving a start point
# --------------------------------------------------------------------------- #


def test_available_checkpoints_are_ordered_by_pipeline_order(tmp_path):
    paths = _paths(tmp_path)
    # Written out of order on purpose; ordering must come from `order`, not name.
    write_checkpoint(_adata(), paths, stage="clustering", order=80, input_fingerprint="c")
    write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint="a")
    write_checkpoint(_adata(), paths, stage="integration", order=60, input_fingerprint="b")
    assert [r.stage for r in available_checkpoints(paths)] == [
        "qc",
        "integration",
        "clustering",
    ]


def test_start_resolves_to_the_newest_checkpoint_strictly_before_the_stage(tmp_path):
    # Resuming AT a stage means its inputs come from the stage before it, so its
    # own checkpoint must not be chosen even when one exists.
    paths = _paths(tmp_path)
    write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint="a")
    write_checkpoint(_adata(), paths, stage="integration", order=60, input_fingerprint="b")
    record = resolve_start_checkpoint(paths, from_stage="clustering", stage_order=STAGE_ORDER)
    assert record.stage == "integration"


def test_start_ignores_a_checkpoint_for_the_requested_stage_itself(tmp_path):
    paths = _paths(tmp_path)
    write_checkpoint(_adata(), paths, stage="qc", order=20, input_fingerprint="a")
    write_checkpoint(_adata(), paths, stage="integration", order=60, input_fingerprint="b")
    record = resolve_start_checkpoint(paths, from_stage="integration", stage_order=STAGE_ORDER)
    assert record.stage == "qc"


def test_start_with_no_earlier_checkpoint_raises_and_lists_what_exists(tmp_path):
    # Silently starting from raw input would look like a resume and produce
    # different numbers, so this must fail loudly and say what IS available.
    paths = _paths(tmp_path)
    write_checkpoint(_adata(), paths, stage="clustering", order=80, input_fingerprint="c")
    with pytest.raises(CheckpointError) as excinfo:
        resolve_start_checkpoint(paths, from_stage="qc", stage_order=STAGE_ORDER)
    assert "clustering" in str(excinfo.value)
    assert "run.checkpoint" in str(excinfo.value)


def test_start_with_unknown_stage_name_raises(tmp_path):
    with pytest.raises(CheckpointError, match="unknown stage"):
        resolve_start_checkpoint(
            _paths(tmp_path), from_stage="not_a_stage", stage_order=STAGE_ORDER
        )


def test_available_checkpoints_on_a_fresh_run_is_empty(tmp_path):
    assert available_checkpoints(_paths(tmp_path)) == []


def test_a_failed_checkpoint_write_leaves_no_half_checkpoint(tmp_path, monkeypatch):
    """Discovery keys on the sidecar, so a sidecar without its object is a trap.

    ``--from-stage`` would find the checkpoint, then fail on read. When a previous
    run of the same stage left both behind, a failed write must clear both rather
    than leave the old pair standing in for this run's resume point.
    """
    from cellquorum.core import h5ad_io
    from cellquorum.core.checkpoint import checkpoint_dir

    paths = _paths(tmp_path)
    adata = _adata()

    assert (
        write_checkpoint(adata, paths, stage="clustering", order=80, input_fingerprint="fp1")
        is not None
    )
    target = checkpoint_dir(paths, "clustering")
    assert (target / "checkpoint.json").exists()
    assert (target / "adata.h5ad").exists()

    def _boom(_adata):
        raise RuntimeError("sanitizer exploded")

    monkeypatch.setattr(h5ad_io, "sanitize_for_h5ad", _boom)
    with pytest.raises(CheckpointError):
        write_checkpoint(adata, paths, stage="clustering", order=80, input_fingerprint="fp2")

    assert not (target / "checkpoint.json").exists()
    assert not (target / "adata.h5ad").exists()
