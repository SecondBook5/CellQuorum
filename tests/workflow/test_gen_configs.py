from __future__ import annotations

import copy

import pytest

from cellquorum.cli.workflow import scaffold
from cellquorum.cli.workflow.gen_configs import (
    ManifestError,
    accounting,
    gen_configs,
    resolve_methods,
)
from cellquorum.config.loader import validate_config_dict
from cellquorum.config.models import StageSelectionConfig


def test_resolve_full_scaffold_when_no_skip_or_blocked(manifest) -> None:
    resolved = resolve_methods(manifest["emt_krt"], scaffold.SCAFFOLD)
    assert set(resolved["run"]) == set(scaffold.SCAFFOLD)
    assert resolved["skip"] == []
    assert resolved["blocked"] == []


def test_resolve_subtracts_skip_and_blocked(manifest) -> None:
    resolved = resolve_methods(manifest["il33_axis"], scaffold.SCAFFOLD)
    assert "pseudobulk" not in resolved["run"]
    assert "rna_velocity" not in resolved["run"]
    assert resolved["skip"] == ["pseudobulk"]
    assert resolved["blocked"] == ["rna_velocity"]
    # run + skip + blocked exactly reconstitutes the scaffold.
    assert set(resolved["run"]) | set(resolved["skip"]) | set(resolved["blocked"]) == set(
        scaffold.SCAFFOLD
    )


def test_gen_configs_emits_one_config_per_cell_type(manifest, template) -> None:
    out = gen_configs(manifest, template)
    assert set(out) == {"il33_axis__KC", "il33_axis__ILC", "emt_krt__KC"}


def test_generated_configs_validate(manifest, template) -> None:
    out = gen_configs(manifest, template)
    for _key, cfg in out.items():
        validate_config_dict(cfg)  # raises if invalid


def test_stage_flags_reflect_resolved_methods(manifest, template) -> None:
    out = gen_configs(manifest, template)
    kc = out["il33_axis__KC"]
    stages = kc["stages"]
    # skipped pseudobulk -> its stages off
    assert stages["differential_expression"] is False
    # blocked rna_velocity -> its stages off
    assert stages["trajectory"] is False
    # a run method (pathway_enrichment) -> its stages on
    assert stages["enrichment"] is True
    # mandatory stage always on
    assert stages["qc"] is True
    # an unrelated optional stage off
    assert stages["grn"] is False


def test_gene_programs_and_overrides_merged(manifest, template) -> None:
    out = gen_configs(manifest, template)
    kc = out["il33_axis__KC"]
    assert kc["run"]["random_seed"] == 7  # from config_overrides
    assert kc["input"]["h5ad"] == "/data/kc.h5ad"  # per-cell-type input
    assert kc["project"]["name"] == "il33_axis__KC"
    # gene_programs merged into markers.panels (the correct schema location)
    assert kc["markers"]["panels"]["alarmin"] == ["Il33", "Il1rl1", "Il13"]


def test_subset_on_emits_input_subset_block(manifest, template) -> None:
    # When a manifest entry declares subset_on, every generated config points at
    # the shared object and carries an input.subset that restricts it to that
    # cell_type at load time (no per-cell-type pre-sliced file).
    subset_manifest = copy.deepcopy(manifest)
    entry = subset_manifest["emt_krt"]
    entry["subset_on"] = "cell_type"
    entry["inputs"] = {"KC": "/data/global.h5ad"}

    out = gen_configs(subset_manifest, template)
    kc = out["emt_krt__KC"]

    assert kc["input"]["h5ad"] == "/data/global.h5ad"
    assert kc["input"]["subset"] == {"column": "cell_type", "values": ["KC"]}


def test_subset_on_configs_still_validate(manifest, template) -> None:
    # The input.subset block must be accepted by the config schema, otherwise the
    # generated configs would fail to load at run time.
    subset_manifest = copy.deepcopy(manifest)
    entry = subset_manifest["emt_krt"]
    entry["subset_on"] = "cell_type"
    entry["inputs"] = {"KC": "/data/global.h5ad"}

    out = gen_configs(subset_manifest, template)
    validate_config_dict(out["emt_krt__KC"])  # raises if invalid


def test_no_subset_on_omits_input_subset_block(manifest, template) -> None:
    # Without subset_on the input block stays subset-free (full-object behavior).
    out = gen_configs(manifest, template)
    assert "subset" not in out["emt_krt__KC"]["input"]


def test_require_agreement_rides_along_on_the_subset_block(manifest, template) -> None:
    # A second annotation column that must call the cell the same thing. Declared
    # once per hypothesis, so every cell_type in the manifest is filtered the same
    # way -- applying it to one lineage and not another is the confound it removes.
    subset_manifest = copy.deepcopy(manifest)
    entry = subset_manifest["emt_krt"]
    entry["subset_on"] = "cell_type"
    entry["require_agreement"] = "ref_state"
    entry["inputs"] = {"KC": "/data/global.h5ad"}

    out = gen_configs(subset_manifest, template)

    assert out["emt_krt__KC"]["input"]["subset"] == {
        "column": "cell_type",
        "values": ["KC"],
        "require_agreement": "ref_state",
    }
    validate_config_dict(out["emt_krt__KC"])  # raises if the schema rejects it


def test_require_agreement_without_subset_on_emits_nothing(manifest, template) -> None:
    # require_agreement is a modifier on the subset. With no subset there is no
    # selected label to agree with, and a stray input.require_agreement would be
    # rejected by the strict schema rather than quietly ignored.
    subset_manifest = copy.deepcopy(manifest)
    subset_manifest["emt_krt"]["require_agreement"] = "ref_state"

    out = gen_configs(subset_manifest, template)

    assert "subset" not in out["emt_krt__KC"]["input"]
    assert "require_agreement" not in out["emt_krt__KC"]["input"]


def test_a_column_cannot_be_asked_to_agree_with_itself(manifest, template) -> None:
    # Naming the same column twice filters nothing, because a column always agrees
    # with itself -- a config that reads as concordance-filtered and is not.
    subset_manifest = copy.deepcopy(manifest)
    entry = subset_manifest["emt_krt"]
    entry["subset_on"] = "cell_type"
    entry["require_agreement"] = "cell_type"
    entry["inputs"] = {"KC": "/data/global.h5ad"}

    out = gen_configs(subset_manifest, template)

    with pytest.raises(Exception, match="DIFFERENT obs column"):
        validate_config_dict(out["emt_krt__KC"])


def test_exclude_on_emits_an_input_exclude_block(manifest, template) -> None:
    # Dropping an audited artifact cluster is not expressible as a subset: a subset
    # names what to KEEP, so leaving out one cluster of a 39-cluster partition would
    # mean listing the other 38, which is unreadable and stale after any re-clustering.
    subset_manifest = copy.deepcopy(manifest)
    entry = subset_manifest["emt_krt"]
    entry["subset_on"] = "cell_type"
    entry["exclude_on"] = "leiden"
    entry["exclude_values"] = ["22"]
    entry["inputs"] = {"KC": "/data/global.h5ad"}

    out = gen_configs(subset_manifest, template)
    kc = out["emt_krt__KC"]

    # Both rules travel together: the lineage slice and the artifact drop.
    assert kc["input"]["subset"] == {"column": "cell_type", "values": ["KC"]}
    assert kc["input"]["exclude"] == {"column": "leiden", "values": ["22"]}
    validate_config_dict(kc)  # raises if the schema rejects it


def test_exclude_on_works_without_a_subset(manifest, template) -> None:
    # A whole-object analysis has no lineage to subset to and still has to drop the
    # artifact cluster, so the exclusion must not be a modifier on the subset.
    subset_manifest = copy.deepcopy(manifest)
    entry = subset_manifest["emt_krt"]
    entry["exclude_on"] = "leiden"
    entry["exclude_values"] = ["22"]

    out = gen_configs(subset_manifest, template)
    kc = out["emt_krt__KC"]

    assert "subset" not in kc["input"]
    assert kc["input"]["exclude"] == {"column": "leiden", "values": ["22"]}
    validate_config_dict(kc)


def test_exclude_values_are_emitted_as_strings(manifest, template) -> None:
    # Cluster ids are written unquoted in YAML more often than not, and an int 22 in
    # the config would be compared against an obs column the loader reads as strings.
    subset_manifest = copy.deepcopy(manifest)
    entry = subset_manifest["emt_krt"]
    entry["exclude_on"] = "leiden"
    entry["exclude_values"] = [22, 30]

    out = gen_configs(subset_manifest, template)

    assert out["emt_krt__KC"]["input"]["exclude"]["values"] == ["22", "30"]


def test_half_an_exclusion_rule_raises(manifest, template) -> None:
    # A column with nothing to drop from it, or values with no column, reads like a
    # filter and removes nothing.
    subset_manifest = copy.deepcopy(manifest)
    subset_manifest["emt_krt"]["exclude_on"] = "leiden"

    with pytest.raises(ValueError, match="half of an exclusion rule"):
        gen_configs(subset_manifest, template)


def test_no_exclude_on_omits_the_input_exclude_block(manifest, template) -> None:
    # Default behaviour is unchanged: no exclusion declared, no exclusion emitted.
    out = gen_configs(manifest, template)
    assert "exclude" not in out["emt_krt__KC"]["input"]


def test_unknown_method_raises(manifest) -> None:
    bad = copy.deepcopy(manifest)
    bad["emt_krt"]["skip"] = {"not_a_method": "typo"}
    with pytest.raises(ManifestError, match="unknown method"):
        resolve_methods(bad["emt_krt"], scaffold.SCAFFOLD)


def test_method_in_two_categories_raises(manifest) -> None:
    bad = copy.deepcopy(manifest)
    bad["il33_axis"]["skip"] = {"rna_velocity": "also blocked"}  # already in blocked
    with pytest.raises(ManifestError, match="two categories|both"):
        resolve_methods(bad["il33_axis"], scaffold.SCAFFOLD)


def test_accounting_shape(manifest) -> None:
    acct = accounting(manifest)
    assert acct["il33_axis"]["skip"] == ["pseudobulk"]
    assert acct["il33_axis"]["blocked"] == ["rna_velocity"]
    assert set(acct["emt_krt"]["run"]) == set(scaffold.SCAFFOLD)


def test_stage_flags_are_emitted_in_a_deterministic_order(manifest, template) -> None:
    """The ``stages:`` block must come out in one fixed order, every process.

    It was built by iterating ``ALL_OPTIONAL_STAGES``, a frozenset -- and the
    iteration order of a set of strings is randomized per interpreter, so three
    consecutive regenerations of the same manifest produced three different key
    orders. Every ``gen-configs`` run therefore rewrote the whole block, which
    makes a real change (a stage flipping on) invisible in the diff and forces a
    semantic YAML comparison to answer "did anything actually change".

    The declaration order of ``StageSelectionConfig`` is the canonical one: it is
    stable, and it reads roughly in pipeline order rather than hash order.
    """
    canonical = list(StageSelectionConfig.model_fields)
    for _key, cfg in gen_configs(manifest, template).items():
        assert list(cfg["stages"]) == canonical


def test_stage_flags_cover_every_declared_stage(manifest, template) -> None:
    """No stage may be silently absent: an absent flag defaults elsewhere."""
    cfg = gen_configs(manifest, template)["il33_axis__KC"]
    assert set(cfg["stages"]) == set(StageSelectionConfig.model_fields)


def test_manifest_stage_declaration_beats_the_computed_flags(manifest, template) -> None:
    # `qc` is a mandatory stage, so the computed block always turns it on. A project
    # whose QC ran once on a shared atlas has to be able to say so in the manifest and
    # be believed -- before this, the computed block silently won and every repo
    # worked around it with its own post-generation patch script.
    off_manifest = copy.deepcopy(manifest)
    entry = off_manifest["emt_krt"]
    entry.setdefault("config_overrides", {})["stages"] = {"qc": False}

    out = gen_configs(off_manifest, template)

    assert out["emt_krt__KC"]["stages"]["qc"] is False
    validate_config_dict(out["emt_krt__KC"])  # raises if invalid


def test_manifest_can_turn_on_a_stage_no_scaffold_method_claims(manifest, template) -> None:
    # Stages outside the scaffold's method map (state scoring, discovery) have no
    # method to enable them, so the manifest is the only place they can come from.
    on_manifest = copy.deepcopy(manifest)
    entry = on_manifest["emt_krt"]
    entry.setdefault("config_overrides", {})["stages"] = {"state_scoring": True}

    out = gen_configs(on_manifest, template)

    assert out["emt_krt__KC"]["stages"]["state_scoring"] is True


def test_stage_overrides_preserve_the_canonical_key_order(manifest, template) -> None:
    # The emitted order has to stay identical across processes, otherwise
    # regenerating from an unchanged manifest rewrites the whole block and buries
    # the real change in the diff.
    off_manifest = copy.deepcopy(manifest)
    entry = off_manifest["emt_krt"]
    entry.setdefault("config_overrides", {})["stages"] = {"qc": False}

    plain = gen_configs(manifest, template)["emt_krt__KC"]["stages"]
    patched = gen_configs(off_manifest, template)["emt_krt__KC"]["stages"]

    assert list(patched) == list(plain)


def test_an_unknown_stage_name_in_the_manifest_is_an_error(manifest, template) -> None:
    # A typo'd stage name would otherwise turn a stage on or off in the manifest
    # text only, which is the failure this whole mechanism exists to end.
    typo_manifest = copy.deepcopy(manifest)
    entry = typo_manifest["emt_krt"]
    entry.setdefault("config_overrides", {})["stages"] = {"qq": False}

    with pytest.raises(ManifestError, match="unknown stage"):
        gen_configs(typo_manifest, template)


def test_disabling_a_stage_a_running_method_needs_points_at_skip(manifest, template) -> None:
    # skip/blocked record an omission in the accounting; a stage flag records
    # nothing. Allowing this would let the accounting report `pseudobulk` as running
    # with its differential expression stage switched off.
    contradictory = copy.deepcopy(manifest)
    entry = contradictory["emt_krt"]
    entry.setdefault("config_overrides", {})["stages"] = {"differential_expression": False}

    with pytest.raises(ManifestError, match="skip"):
        gen_configs(contradictory, template)
