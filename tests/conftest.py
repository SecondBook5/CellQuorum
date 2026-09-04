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
