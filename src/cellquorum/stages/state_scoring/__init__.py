"""Cell-state program scoring stage package."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.state_scoring.aucell_method import AucellMethod
from cellquorum.stages.state_scoring.config import StateScoringConfig
from cellquorum.stages.state_scoring.programs import STATE_PROGRAMS, resolve_programs
from cellquorum.stages.state_scoring.score_genes_method import ScoreGenesMethod
from cellquorum.stages.state_scoring.ucell_method import UcellMethod

for _method in (ScoreGenesMethod, AucellMethod, UcellMethod):
    if not METHOD_REGISTRY.has("state_scoring", _method.name):
        METHOD_REGISTRY.register(_method)

# StateScoreUmapMethod lives in embeddings/__init__.py despite its module
# living here -- see state_score_umap_method.py's own docstring for why
# (needs both a scored program from this stage AND embeddings' UMAP/PHATE,
# which only exists after THIS stage has already run).

__all__ = [
    "STATE_PROGRAMS",
    "AucellMethod",
    "ScoreGenesMethod",
    "StateScoringConfig",
    "UcellMethod",
    "resolve_programs",
]
