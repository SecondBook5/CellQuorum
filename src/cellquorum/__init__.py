"""Public API for CellQuorum.

The public surface resolves **lazily** (:pep:`562`). ``__version__`` is bound
eagerly because it is cheap and universally wanted; every other name is imported
on first attribute access.

Eagerly importing this surface used to cost ~3.7s and pull 7300+ modules — torch,
lightning, jax, and scvi among them — into any process that touched the package,
because the notebook namespaces reach through :mod:`cellquorum.config.models`,
which aggregates all 30 stage config modules. That made ``cellquorum --version``
a five-second command and put a multi-second import tax on every one of the
~294 test modules. Deferring resolution keeps ``from cellquorum import pp`` and
``cq.run_pipeline(...)`` working exactly as before while making the bare import
essentially free; ``tests/test_import_cost.py`` pins the invariant so it cannot
silently regress again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Import the canonical package version eagerly: it is a plain string constant with
# no dependencies, and downstream tools expect `cellquorum.__version__` to resolve
# without triggering the analysis stack.
from cellquorum.version import __version__

# Map each lazily exported public name to the module that owns it. The api
# namespaces are modules; run_pipeline is a function re-exported from the package.
_LAZY_EXPORTS: dict[str, str] = {
    "diag": "cellquorum.api",
    "evidence": "cellquorum.api",
    "pp": "cellquorum.api",
    "run_pipeline": "cellquorum.api",
    "tl": "cellquorum.api",
    "utils": "cellquorum.utils",
}

# Give type checkers, IDEs, and mkdocstrings the real bindings. This block never
# executes, so it adds no import cost, but without it the lazy names below would
# look undefined to static analysis.
if TYPE_CHECKING:
    from cellquorum import utils
    from cellquorum.api import diag, evidence, pp, run_pipeline, tl

# Define the public symbols exposed by `from cellquorum import *`.
__all__: list[str] = [
    "__version__",
    "diag",
    "evidence",
    "pp",
    "run_pipeline",
    "tl",
    "utils",
]


def __getattr__(name: str) -> object:
    """
    Resolve a lazily exported public name on first access (:pep:`562`).

    Args:
        name: Attribute name requested on the ``cellquorum`` package.

    Returns:
        The object of that name, imported from its owning module.

    Raises:
        AttributeError: If the name is not part of the public surface.
    """

    # Look up which module owns this name.
    module_path = _LAZY_EXPORTS.get(name)

    # Reject unknown names exactly as a normal module would, so typos and
    # `hasattr` probes behave the same as before the surface became lazy.
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Import the owning module.
    from importlib import import_module

    module = import_module(module_path)

    # `utils` is itself the target module, so return it directly; the api names are
    # attributes hanging off cellquorum.api.
    value = module if name == "utils" else getattr(module, name)

    # Cache on the package so later lookups bypass __getattr__ entirely.
    globals()[name] = value

    # Return the resolved object to the caller.
    return value


def __dir__() -> list[str]:
    """
    List the package's public surface for ``dir()`` and tab completion.

    Returns:
        Sorted public attribute names.
    """

    # Report the documented public surface regardless of what has been resolved.
    return sorted(__all__)
