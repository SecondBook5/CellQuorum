"""Every implemented stage module opens with a step header pinned to the catalog.

The header's `order=<N>` and stage name are matched against the registered
StageSpec, so the header cannot drift from the decorator that defines the
stage's real order/identity. The trailing role prose is free text.
"""

import inspect
import re

from cellquorum.core.stages import all_stage_specs

HEADER_RE = re.compile(r"^# Pipeline step \(order=(\d+)\): (\S+) — .+\.$")


def _first_line(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.readline().rstrip("\n")


def test_every_implemented_stage_has_matching_step_header():
    for spec in all_stage_specs():
        if not spec.is_implemented:
            continue
        path = inspect.getsourcefile(spec.factory)
        m = HEADER_RE.match(_first_line(path))
        assert m is not None, f"{spec.name}: missing/malformed step header in {path}"
        assert int(m.group(1)) == spec.order, f"{spec.name}: header order {m.group(1)} != {spec.order}"
        assert m.group(2) == spec.name, f"{spec.name}: header name {m.group(2)!r} != {spec.name!r}"
