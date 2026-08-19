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
