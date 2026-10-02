"""The stages/ ordered map matches the registered catalog exactly.

Parses the markdown table in stages/README.md and asserts its (order, name)
rows equal the catalog's, so the map cannot drift when stages are added,
removed, or reordered.
"""

import re
from pathlib import Path

import cellquorum
from cellquorum.core.stages import all_stage_specs

README = Path(cellquorum.__file__).parent / "stages" / "README.md"
ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|")


def _map_rows():
    rows = []
    for line in README.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if m:
            rows.append((int(m.group(1)), m.group(2)))
    return rows


def test_stage_map_matches_catalog():
    expected = [(s.order, s.name) for s in all_stage_specs()]
    assert _map_rows() == expected
