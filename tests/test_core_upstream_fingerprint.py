"""The upstream fingerprint: what it must notice, and what it must ignore.

Both halves are load-bearing. If it misses an upstream change, a resume silently
mixes settings. If it reacts to a downstream change, every checkpoint is invalidated
by unrelated edits and the stage-by-stage loop recomputes from scratch for no safety
gain — which is what would push someone back to verifying by hand.
"""

from __future__ import annotations

import pytest

from cellquorum.core.fingerprint import compute_upstream_fingerprint

STAGE_ORDER = {
    "ambient_correction": 10,
    "qc": 20,
    "preprocessing": 30,
    "cell_cell_communication": 200,
}


def _config(**overrides) -> dict:
    config = {
        "run": {"random_seed": 1337, "verbose": True, "checkpoint": True},
        "input": {"h5ad": "/nonexistent/input.h5ad"},
        "stages": {"ambient_correction": True, "qc": True, "preprocessing": True},
        "ambient_correction": {"enabled": True},
        "qc": {"mode": "filter", "min_genes": 200},
        "preprocessing": {"normalization": {"overwrite": False}},
        "cell_cell_communication": {"n_permutations": 1000},
    }
    config.update(overrides)
    return config


def _fp(config: dict, through: str = "qc") -> str:
    return compute_upstream_fingerprint(
        config=config, stage_order=STAGE_ORDER, through_stage=through
    )


def test_identical_config_gives_identical_fingerprint():
    assert _fp(_config()) == _fp(_config())


# --------------------------------------------------------------------------- #
# must NOTICE: anything that could have shaped the checkpoint
# --------------------------------------------------------------------------- #


def test_changing_the_checkpointed_stage_config_changes_the_fingerprint():
    changed = _config(qc={"mode": "filter", "min_genes": 500})
    assert _fp(changed) != _fp(_config())


def test_changing_an_earlier_stage_config_changes_the_fingerprint():
    # Ambient correction runs before QC, so it shaped what QC received.
    changed = _config(ambient_correction={"enabled": False})
    assert _fp(changed) != _fp(_config())


def test_disabling_an_earlier_stage_changes_the_fingerprint():
    # Enablement lives in the planner block, not the stage block, and matters even
    # for stages that have no config block of their own.
    changed = _config(stages={"ambient_correction": False, "qc": True, "preprocessing": True})
    assert _fp(changed) != _fp(_config())


def test_changing_the_random_seed_changes_the_fingerprint():
    changed = _config(run={"random_seed": 999999, "verbose": True, "checkpoint": True})
    assert _fp(changed) != _fp(_config())


def test_changing_the_input_changes_the_fingerprint():
    changed = _config(input={"h5ad": "/nonexistent/other.h5ad"})
    assert _fp(changed) != _fp(_config())


def test_input_size_is_part_of_the_identity(tmp_path):
    target = tmp_path / "input.h5ad"
    target.write_bytes(b"x" * 10)
    before = _fp(_config(input={"h5ad": str(target)}))
    target.write_bytes(b"x" * 20)
    assert _fp(_config(input={"h5ad": str(target)})) != before


# --------------------------------------------------------------------------- #
# must IGNORE: things that cannot have shaped the checkpoint
# --------------------------------------------------------------------------- #


def test_changing_a_downstream_stage_config_does_not_change_the_fingerprint():
    """The whole reason this is scoped rather than hashing the entire config.

    Cell-cell communication runs long after QC, so it cannot retroactively alter what
    QC produced. Invalidating the QC checkpoint over it would force a full recompute
    for no safety gain.
    """
    changed = _config(cell_cell_communication={"n_permutations": 50})
    assert _fp(changed, through="qc") == _fp(_config(), through="qc")


def test_a_downstream_change_still_counts_once_it_is_upstream():
    # Same edit, but for a checkpoint written after that stage ran.
    changed = _config(cell_cell_communication={"n_permutations": 50})
    assert _fp(changed, through="cell_cell_communication") != _fp(
        _config(), through="cell_cell_communication"
    )


def test_cosmetic_run_settings_do_not_change_the_fingerprint():
    # Verbosity and the checkpoint switches cannot change a result; refusing a resume
    # over them would be pure friction.
    changed = _config(run={"random_seed": 1337, "verbose": False, "checkpoint": False})
    assert _fp(changed) == _fp(_config())


# --------------------------------------------------------------------------- #
# unset settings: the upgrade path
# --------------------------------------------------------------------------- #
#
# What is hashed is the RESOLVED config, and resolution emits every field a model
# declares — so adding one optional setting adds a None to every run that never
# mentions it. If that changed the fingerprint, every checkpoint on disk would go
# stale on upgrade and the guard would report a setting change nobody made. It did:
# adding `input.subset.require_agreement` invalidated a finished run's checkpoints.


def test_a_newly_added_optional_stage_setting_does_not_invalidate_a_checkpoint():
    before = _config()
    after = _config(qc={"mode": "filter", "min_genes": 200, "max_mito_quantile": None})
    assert _fp(after) == _fp(before)


def test_a_newly_added_optional_input_field_does_not_invalidate_a_checkpoint():
    # The exact shape of the real regression: a nested optional block resolving to
    # None inside the input spec.
    before = _config()
    after = _config(input={"h5ad": "/nonexistent/input.h5ad", "subset": None})
    assert _fp(after) == _fp(before)

    nested = _config(
        input={
            "h5ad": "/nonexistent/input.h5ad",
            "subset": {"obs_key": None, "require_agreement": None},
        }
    )
    assert _fp(nested) == _fp(before)


def test_setting_that_optional_field_to_a_value_does_change_the_fingerprint():
    # Ignoring None must not become ignoring the setting. Once it holds a value it
    # shaped the run, and a checkpoint written without it is genuinely stale.
    before = _config()
    after = _config(qc={"mode": "filter", "min_genes": 200, "max_mito_quantile": 0.98})
    assert _fp(after) != _fp(before)


def test_clearing_a_setting_back_to_its_default_still_changes_the_fingerprint():
    # None collapses to absent, and absent is not the same as the old value: going
    # from min_genes 200 to unset changed what QC did.
    assert _fp(_config(qc={"mode": "filter", "min_genes": None})) != _fp(_config())


def test_an_unset_enablement_flag_is_the_same_as_not_listing_the_stage():
    before = _config(stages={"qc": True, "preprocessing": True})
    after = _config(stages={"ambient_correction": None, "qc": True, "preprocessing": True})
    assert _fp(after) == _fp(before)


def test_a_falsy_setting_is_not_treated_as_unset():
    # The rule is None, not falsiness. `min_genes: 0`, `enabled: false` and `""` are
    # all deliberate choices that change what a stage does, and a checkpoint written
    # under a different one of them IS stale.
    assert _fp(_config(qc={"mode": "filter", "min_genes": 0})) != _fp(
        _config(qc={"mode": "filter"})
    )
    assert _fp(_config(ambient_correction={"enabled": False})) != _fp(
        _config(ambient_correction={})
    )
    assert _fp(_config(qc={"mode": ""})) != _fp(_config(qc={}))


def test_a_stage_block_that_specifies_nothing_matches_having_no_block():
    # Enablement is hashed separately, so an empty config block adds no information.
    assert _fp(_config(qc={})) == _fp({k: v for k, v in _config().items() if k != "qc"})


def test_a_none_inside_a_list_is_kept_because_position_is_meaning():
    # Dropping it would shift every later element and silently equate two different
    # lists — e.g. per-batch overrides given positionally.
    with_hole = _config(qc={"mode": "filter", "per_batch": [1, None, 3]})
    compacted = _config(qc={"mode": "filter", "per_batch": [1, 3]})
    assert _fp(with_hole) != _fp(compacted)


# --------------------------------------------------------------------------- #
# scoping and errors
# --------------------------------------------------------------------------- #


def test_different_stages_get_different_fingerprints():
    config = _config()
    assert _fp(config, through="qc") != _fp(config, through="preprocessing")


def test_unknown_stage_raises_rather_than_hashing_nothing():
    # Silently returning a fingerprint scoped to no stages would compare equal
    # across genuinely different configs.
    with pytest.raises(KeyError, match="unknown stage"):
        _fp(_config(), through="not_a_stage")


def test_missing_blocks_are_tolerated():
    # A minimal config must still fingerprint rather than raise.
    assert isinstance(_fp({"run": {}, "input": None, "stages": {}}), str)
