"""Move 3: enrichment -> comparative.enrichment.

The old top-level import paths must still resolve to the exact objects now living
under cellquorum.comparative — including the ranking/priors paths that 7 analysis
scripts import directly.
"""

from __future__ import annotations

import importlib

import pytest

_MODULE_PAIRS = [
    ("cellquorum.enrichment", "cellquorum.comparative.enrichment"),
    ("cellquorum.enrichment.activity_method", "cellquorum.comparative.enrichment.activity_method"),
    ("cellquorum.enrichment.config", "cellquorum.comparative.enrichment.config"),
    ("cellquorum.enrichment.gsea_method", "cellquorum.comparative.enrichment.gsea_method"),
    ("cellquorum.enrichment.gsva_method", "cellquorum.comparative.enrichment.gsva_method"),
    ("cellquorum.enrichment.ora_method", "cellquorum.comparative.enrichment.ora_method"),
    ("cellquorum.enrichment.priors", "cellquorum.comparative.enrichment.priors"),
    ("cellquorum.enrichment.ranking", "cellquorum.comparative.enrichment.ranking"),
    ("cellquorum.enrichment.viz", "cellquorum.comparative.enrichment.viz"),
    ("cellquorum.enrichment.viz.config", "cellquorum.comparative.enrichment.viz.config"),
    ("cellquorum.enrichment.viz.io", "cellquorum.comparative.enrichment.viz.io"),
    ("cellquorum.enrichment.viz.plots", "cellquorum.comparative.enrichment.viz.plots"),
    ("cellquorum.enrichment.viz.viz_methods", "cellquorum.comparative.enrichment.viz.viz_methods"),
]


@pytest.mark.parametrize("old_path,new_path", _MODULE_PAIRS)
def test_old_path_reexports_public_api(old_path, new_path):
    old_mod = importlib.import_module(old_path)
    new_mod = importlib.import_module(new_path)
    assert new_mod.__all__, f"{new_path} defines no public __all__"
    assert old_mod.__all__ == new_mod.__all__
    for name in new_mod.__all__:
        assert getattr(old_mod, name) is getattr(new_mod, name)


def test_ranking_and_priors_downstream_identity():
    from cellquorum.comparative.enrichment.priors import get_net as new_get_net
    from cellquorum.comparative.enrichment.ranking import de_table_to_ranking as new_rank
    from cellquorum.enrichment.priors import get_net as old_get_net
    from cellquorum.enrichment.ranking import de_table_to_ranking as old_rank

    assert old_rank is new_rank
    assert old_get_net is new_get_net
