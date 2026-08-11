"""Table 0 core scaffold: the seven methods every hypothesis runs by default,
and the mapping from each method to the engine stage flags it enables.

This is the single source of truth for translating a hypothesis manifest's
method selection into ``CellQuorumConfig.stages`` booleans. Stage-flag names
here MUST match fields of ``cellquorum.config.models.StageSelectionConfig``;
``tests/workflow/test_scaffold.py`` enforces that.
"""

from __future__ import annotations

from cellquorum.config.models import StageSelectionConfig

# The seven Table 0 methods, in track-sheet order.
SCAFFOLD: list[str] = [
    "pseudobulk",
    "subclustering",
    "pathway_enrichment",
    "rna_velocity",
    "phate_pseudotime",
    "cell_cell_communication",
    "progeny",
]

# Upstream stages every run needs regardless of method selection: load ->
# QC -> preprocessing -> dimensionality -> clustering -> integration ->
# annotation and its consensus/diagnostics. These are prerequisites for
# every downstream method, so they are always enabled.
MANDATORY_STAGES: list[str] = [
    "qc",
    "preprocessing",
    "feature_selection",
    "dimensionality",
    "clustering",
    "integration",
    "annotation",
    "annotation_consensus",
    "annotation_diagnostics",
    "population_identity",
]

# Each scaffold method -> the optional stage flags it turns on. Values are
# verified members of StageSelectionConfig. Rationale per method:
#   pseudobulk            -> pseudobulk differential expression + its figures
#   subclustering         -> cell-state subclustering + adjudication
#   pathway_enrichment    -> enrichment + enrichment figures
#   rna_velocity          -> trajectory (scVelo/CellRank) + trajectory figures
#   phate_pseudotime      -> embeddings (PHATE) driving pseudotime ordering
#   cell_cell_communication -> CCC + CCC figures + network analysis
#   progeny               -> pathway-activity inference (molecular_inference)
SCAFFOLD_METHOD_STAGES: dict[str, list[str]] = {
    "pseudobulk": ["differential_expression", "de_viz"],
    "subclustering": ["subclustering", "adjudication"],
    "pathway_enrichment": ["enrichment", "enrichment_viz"],
    "rna_velocity": ["trajectory", "trajectory_viz"],
    "phate_pseudotime": ["embeddings"],
    "cell_cell_communication": [
        "cell_cell_communication",
        "ccc_viz",
        "network_analysis",
    ],
    "progeny": ["molecular_inference"],
}

# Every stage flag gen_configs may toggle OFF (all legal flags minus mandatory).
ALL_OPTIONAL_STAGES: frozenset[str] = frozenset(StageSelectionConfig.model_fields) - set(
    MANDATORY_STAGES
)
