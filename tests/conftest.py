"""Root pytest configuration for the CellQuorum test suite.

Deliberately thin. The shared helpers a test actually calls — the external-dataset
opt-in and the cached R-package probe — live in :mod:`tests._external_data` so they can
be imported explicitly:

```python
from _external_data import require_cellranger_library, require_r_package
```

They are plain functions in a plain module rather than fixtures in this file for two
reasons. ``tests/`` is not a package, and a second ``conftest.py`` exists under
``tests/workflow/``, so ``from conftest import ...`` is ambiguous and fragile. And a
skip helper reads better called inline (``require_r_package("SoupX")``) than threaded
through a fixture argument. pytest prepends ``tests/`` to ``sys.path``, which is what
makes the bare ``from _external_data import ...`` resolve.

See :mod:`tests._external_data` for the environment variables that enable the
integration tests against real data.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _restore_matplotlib_style() -> Iterator[None]:
    """Give every test the matplotlib style it started with.

    Figure code restyles matplotlib for the whole process (``apply_theme``). A run through the
    executor undoes that when each stage ends, but a test that calls a stage or a plotting
    function directly does not, and the figure-gallery pixel comparisons then fail depending on
    which test happened to run before them.
    """
    try:
        from cellquorum.core.executor import _contained_matplotlib_style
    except ModuleNotFoundError as exc:
        # The lightweight CI jobs (environment locks) install no cellquorum, so there is no
        # figure code to restyle matplotlib. Only the package itself being absent is excused: a
        # missing dependency inside an installed cellquorum still fails the test.
        if exc.name != "cellquorum":
            raise
        yield
        return

    with _contained_matplotlib_style():
        yield
