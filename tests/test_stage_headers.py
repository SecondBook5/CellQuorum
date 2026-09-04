"""Every implemented stage module opens with a step header pinned to the catalog.

The header's `order=<N>` and stage name are matched against the registered
StageSpec, so the header cannot drift from the decorator that defines the
stage's real order/identity. The trailing role prose is free text.

The header must be ONE line and fit the project line length. Those two rules
pull against each other -- a wrapped header fails the regex, an unwrapped long
one fails ruff's E501 -- so both are asserted here, where the author is looking,
instead of leaving the second to surface as an unrelated lint failure.
"""

import inspect
import re
import tomllib
from pathlib import Path

from cellquorum.core.stages import all_stage_specs

HEADER_RE = re.compile(r"^# Pipeline step \(order=(\d+)\): (\S+) — .+\.$")

# Read the limit from pyproject rather than repeating it, so raising the
# project line length cannot leave this test enforcing the old number.
LINE_LENGTH = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
)["tool"]["ruff"]["line-length"]


def _first_line(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.readline().rstrip("\n")


def test_every_implemented_stage_has_matching_step_header():
    for spec in all_stage_specs():
        if not spec.is_implemented:
            continue
        path = inspect.getsourcefile(spec.factory)
        first_line = _first_line(path)
        m = HEADER_RE.match(first_line)
        assert m is not None, f"{spec.name}: missing/malformed step header in {path}"
        assert len(first_line) <= LINE_LENGTH, (
            f"{spec.name}: step header is {len(first_line)} chars (limit {LINE_LENGTH}) "
            f"in {path}. Shorten the role prose; the header cannot wrap."
        )
        assert (
            int(m.group(1)) == spec.order
        ), f"{spec.name}: header order {m.group(1)} != {spec.order}"
        assert m.group(2) == spec.name, f"{spec.name}: header name {m.group(2)!r} != {spec.name!r}"
