"""Pure expansion of a hypothesis manifest into per-(hypothesis, cell_type)
cellquorum config dicts, with a completeness check that makes a forgotten
scaffold method a hard error rather than a silent omission.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from cellquorum.cli.workflow.scaffold import (
    MANDATORY_STAGES,
    SCAFFOLD,
    SCAFFOLD_METHOD_STAGES,
    STAGE_ORDER,
)


class ManifestError(ValueError):
    """Raised when a hypothesis manifest is incomplete or inconsistent."""


def _method_keys(section: Any) -> list[str]:
    """A skip/blocked section may be a list or a {method: reason} mapping."""
    if section is None:
        return []
    if isinstance(section, Mapping):
        return list(section.keys())
    if isinstance(section, list):
        return list(section)
    raise ManifestError(f"skip/blocked must be a list or mapping, got {type(section).__name__}")


def resolve_methods(entry: Mapping[str, Any], scaffold: list[str]) -> dict[str, list[str]]:
    skip = _method_keys(entry.get("skip"))
    blocked = _method_keys(entry.get("blocked"))

    scaffold_set = set(scaffold)
    for name in [*skip, *blocked]:
        if name not in scaffold_set:
            raise ManifestError(f"unknown method {name!r}; not in scaffold {scaffold}")

    overlap = set(skip) & set(blocked)
    if overlap:
        raise ManifestError(
            f"method(s) {sorted(overlap)} listed in two categories (skip and blocked)"
        )

    run = [m for m in scaffold if m not in set(skip) and m not in set(blocked)]

    # Completeness: run + skip + blocked must reconstitute the whole scaffold.
    accounted = set(run) | set(skip) | set(blocked)
    if accounted != scaffold_set:
        missing = scaffold_set - accounted
        raise ManifestError(f"scaffold methods unaccounted for: {sorted(missing)}")

    return {"run": run, "skip": skip, "blocked": blocked}


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _stage_flags(
    run_methods: list[str], method_stages: dict[str, list[str]], mandatory: list[str]
) -> dict[str, bool]:
    enabled: set[str] = set(mandatory)
    for method in run_methods:
        enabled.update(method_stages[method])
    unknown = enabled - set(STAGE_ORDER)
    if unknown:
        # Previously these were emitted as extra ``stages:`` keys and surfaced only
        # as a pydantic validation error naming no method, so a typo in a method's
        # stage list was hard to trace back to the method.
        raise ManifestError(
            f"method stage(s) {sorted(unknown)} are not fields of "
            f"StageSelectionConfig; fix SCAFFOLD_METHOD_STAGES or MANDATORY_STAGES."
        )
    # Built from the canonical order, not from ``enabled``/ALL_OPTIONAL_STAGES: the
    # emitted key order has to be identical across processes (see STAGE_ORDER).
    return {stage: stage in enabled for stage in STAGE_ORDER}


def _apply_stage_overrides(
    flags: dict[str, bool],
    overrides: Mapping[str, Any],
    run_methods: list[str],
    method_stages: dict[str, list[str]],
) -> dict[str, bool]:
    """
    Let a manifest's ``config_overrides.stages`` win over the computed flags.

    Without this the computed block silently clobbered whatever the manifest
    declared, so a manifest could say ``qc: false``, be believed by everyone who
    read it, and still generate a config with QC on -- which every repo then
    worked around with its own post-generation patch script. That is the same
    decision written twice per repo, and the copy that loses is the readable one.

    The one thing an override may NOT do is switch off a stage that a method
    declared to run depends on. ``skip``/``blocked`` exist to record an omission in
    the accounting; a stage flag records nothing, so the same omission expressed
    that way would make the accounting claim a method ran when its stage was off.

    Args:
        flags: Stage flags computed from the resolved methods, in STAGE_ORDER.
        overrides: The manifest's ``config_overrides.stages`` mapping, possibly empty.
        run_methods: Scaffold methods resolved as running for this hypothesis.
        method_stages: Map of scaffold method to the stages it turns on.

    Returns:
        The flags with the overrides applied, still in STAGE_ORDER.

    Raises:
        ManifestError: If an override names an unknown stage, or disables a stage a
            running method needs.
    """

    if not overrides:
        return flags

    unknown = sorted(set(overrides) - set(STAGE_ORDER))
    if unknown:
        raise ManifestError(
            f"config_overrides.stages names unknown stage(s) {unknown}; they are not "
            f"fields of StageSelectionConfig. A silently ignored typo here would turn "
            f"a stage on or off in the manifest text only."
        )

    # Which running method is counting on each stage.
    claimed: dict[str, str] = {}
    for method in run_methods:
        for stage in method_stages[method]:
            claimed.setdefault(stage, method)

    resolved = dict(flags)
    for stage, value in overrides.items():
        if not value and stage in claimed:
            raise ManifestError(
                f"config_overrides.stages.{stage}=false contradicts method "
                f"{claimed[stage]!r}, which this manifest declares as running and "
                f"which needs that stage. List {claimed[stage]!r} under `skip:` with a "
                f"reason instead, so the omission appears in the accounting rather "
                f"than only in a stage flag."
            )
        resolved[stage] = bool(value)
    return resolved


def gen_configs(
    manifest: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    scaffold: list[str] = SCAFFOLD,
    method_stages: dict[str, list[str]] = SCAFFOLD_METHOD_STAGES,
    mandatory_stages: list[str] = MANDATORY_STAGES,
) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for hyp_id, entry in manifest.items():
        resolved = resolve_methods(entry, scaffold)
        stages = _stage_flags(resolved["run"], method_stages, mandatory_stages)
        # Fold the manifest's own stage declarations in BEFORE the final merge, which
        # emits `stages` last and would otherwise overwrite them.
        stages = _apply_stage_overrides(
            stages,
            (entry.get("config_overrides") or {}).get("stages") or {},
            resolved["run"],
            method_stages,
        )
        cell_types = entry["cell_types"]
        inputs = entry["inputs"]
        # Stage declarations are pulled OUT of the overrides: they are folded into
        # the computed block above, and leaving a copy here would also merge them
        # into the template first, where they seed the emitted `stages:` mapping and
        # put whichever stage the manifest happened to name at the front of it.
        overrides = {
            key: value
            for key, value in (entry.get("config_overrides") or {}).items()
            if key != "stages"
        }
        programs = entry.get("gene_programs", {})
        # When ``subset_on`` names an obs column, every cell_type points at the
        # same shared object and the engine restricts it to that cell_type at
        # load time (recorded in provenance) — no per-cell-type pre-sliced file.
        subset_on = entry.get("subset_on")
        # ``require_agreement`` names a SECOND annotation column that must call the
        # cell the same thing, dropping the cells the two annotations disagree
        # about. Declared once per hypothesis rather than per cell_type: a slice
        # built by agreement and one built by a single label are not filtered
        # equally, so applying it unevenly across the cell types of one manifest is
        # exactly the confound it exists to remove.
        require_agreement = entry.get("require_agreement")
        # ``exclude_on`` + ``exclude_values`` DROP rows, which a subset cannot express:
        # a subset says which values to keep, so leaving out one artifact cluster of a
        # 39-cluster partition would mean naming the other 38 — unreadable, and silently
        # incomplete the next time the object is re-clustered. The intended use is a
        # data artifact identified by ``cellquorum.stats.cluster_artifact_audit``.
        # Declared once per hypothesis, for the same reason as ``require_agreement``:
        # cell types filtered unevenly inside one manifest are not comparable, and
        # comparing them is what the manifest is for.
        exclude_on = entry.get("exclude_on")
        exclude_values = entry.get("exclude_values")
        if (exclude_on is None) != (exclude_values is None):
            raise ValueError(
                f"hypothesis {hyp_id!r} declares only half of an exclusion rule: "
                f"exclude_on={exclude_on!r}, exclude_values={exclude_values!r}. Half a "
                f"filter reads like a filter and removes nothing."
            )
        for cell_type in cell_types:
            key = f"{hyp_id}__{cell_type}"
            cfg = _deep_merge(dict(template), overrides)
            input_block: dict[str, Any] = {"h5ad": inputs[cell_type]}
            if subset_on is not None:
                subset_block: dict[str, Any] = {"column": subset_on, "values": [cell_type]}
                if require_agreement is not None:
                    subset_block["require_agreement"] = require_agreement
                input_block["subset"] = subset_block
            if exclude_on is not None:
                input_block["exclude"] = {
                    "column": exclude_on,
                    "values": [str(value) for value in exclude_values or []],
                }
            cfg = _deep_merge(
                cfg,
                {
                    "project": {"name": key},
                    "input": input_block,
                    "stages": stages,
                },
            )
            if programs:
                cfg = _deep_merge(cfg, {"markers": {"panels": programs}})
            configs[key] = cfg
    return configs


def accounting(
    manifest: Mapping[str, Any], *, scaffold: list[str] = SCAFFOLD
) -> dict[str, dict[str, list[str]]]:
    return {hyp_id: resolve_methods(entry, scaffold) for hyp_id, entry in manifest.items()}
