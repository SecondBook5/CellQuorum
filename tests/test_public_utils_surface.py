"""Public analytical-utility surface (consolidation spec Move 5, DoD item 5).

The consolidation design promised that the reused building blocks the analysis
scripts reach for — ``de_table_to_ranking``, ``get_net``, ``aggregate_pseudobulk``
— become a first-class, versioned public surface under ``cellquorum.utils`` rather
than deep-internal reach-in, while the old deep-import paths keep resolving to the
same objects. This file locks that contract: the public names must BE the canonical
objects (re-export, never a copy), the surface must be reachable as ``cq.utils``,
and importing it must not eagerly pull a heavy optional dependency (the
skip-not-crash invariant).
"""

from __future__ import annotations

import importlib


def test_utils_exposes_the_three_building_blocks():
    utils = importlib.import_module("cellquorum.utils")
    assert hasattr(utils, "de_table_to_ranking")
    assert hasattr(utils, "get_net")
    assert hasattr(utils, "aggregate_pseudobulk")


def test_utils_names_are_the_canonical_objects_not_copies():
    # Re-export, not reimplementation: the public name must BE the canonical one,
    # so a bug fixed in the engine is a bug fixed for power-user scripts too.
    from cellquorum import utils
    from cellquorum.comparative.differential_expression.pseudobulk import (
        aggregate_pseudobulk,
    )
    from cellquorum.comparative.enrichment.priors import get_net
    from cellquorum.comparative.enrichment.ranking import de_table_to_ranking

    assert utils.de_table_to_ranking is de_table_to_ranking
    assert utils.get_net is get_net
    assert utils.aggregate_pseudobulk is aggregate_pseudobulk


def test_companion_types_exposed():
    # A public function is only usable with the types it returns/raises.
    from cellquorum import utils
    from cellquorum.comparative.differential_expression.pseudobulk import (
        PseudobulkResult,
    )
    from cellquorum.comparative.enrichment.priors import PriorFetchError

    assert utils.PseudobulkResult is PseudobulkResult
    assert utils.PriorFetchError is PriorFetchError


def test_utils_declares_all():
    from cellquorum import utils

    for name in (
        "de_table_to_ranking",
        "get_net",
        "aggregate_pseudobulk",
        "PseudobulkResult",
        "PriorFetchError",
    ):
        assert name in utils.__all__


def test_utils_accessible_as_top_level_attribute():
    # `cq.utils` resolves without a separate import — a first-class surface.
    import cellquorum

    assert cellquorum.utils.de_table_to_ranking is not None
    assert "utils" in cellquorum.__all__


def test_old_deep_import_paths_still_resolve_to_same_objects():
    # Zero downstream breakage: the pre-consolidation deep paths the 14 analysis
    # scripts use keep working and point at the very same objects utils exposes.
    from cellquorum import utils
    from cellquorum.differential_expression.pseudobulk import (
        aggregate_pseudobulk as old_pb,
    )
    from cellquorum.enrichment.ranking import de_table_to_ranking as old_rank

    assert old_rank is utils.de_table_to_ranking
    assert old_pb is utils.aggregate_pseudobulk


def test_importing_utils_does_not_eagerly_import_decoupler():
    # get_net lazy-imports decoupler inside the call; merely exposing it publicly
    # must not drag a heavy optional dependency in at import time.
    import sys

    sys.modules.pop("cellquorum.utils", None)
    sys.modules.pop("decoupler", None)
    importlib.import_module("cellquorum.utils")
    assert "decoupler" not in sys.modules
