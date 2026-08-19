"""Pure expansion of a hypothesis manifest into per-(hypothesis, cell_type)
cellquorum config dicts, with a completeness check that makes a forgotten
scaffold method a hard error rather than a silent omission.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from cellquorum.cli.workflow.scaffold import (
    ALL_OPTIONAL_STAGES,
    MANDATORY_STAGES,
    SCAFFOLD,
    SCAFFOLD_METHOD_STAGES,
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
    flags = {stage: False for stage in ALL_OPTIONAL_STAGES}
    flags.update({stage: True for stage in mandatory})
    for stage in enabled:
        flags[stage] = True
    return flags


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
        cell_types = entry["cell_types"]
        inputs = entry["inputs"]
        overrides = entry.get("config_overrides", {})
        programs = entry.get("gene_programs", {})
        # When ``subset_on`` names an obs column, every cell_type points at the
        # same shared object and the engine restricts it to that cell_type at
        # load time (recorded in provenance) — no per-cell-type pre-sliced file.
        subset_on = entry.get("subset_on")
        for cell_type in cell_types:
            key = f"{hyp_id}__{cell_type}"
            cfg = _deep_merge(dict(template), overrides)
            input_block: dict[str, Any] = {"h5ad": inputs[cell_type]}
            if subset_on is not None:
                input_block["subset"] = {"column": subset_on, "values": [cell_type]}
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
