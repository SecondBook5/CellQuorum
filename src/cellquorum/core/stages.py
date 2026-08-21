"""Stage discovery aggregator.

Importing this module imports every implemented stage module exactly once,
firing each stage's ``@register_stage`` decorator and populating the
process-wide catalog. It also registers the planned-but-unimplemented stages
(which have no class to decorate).

The executor and planner import :func:`all_stage_specs` from here LAZILY (inside
the functions that use it) so reading the catalog always sees the fully
populated set without pulling this module into their import time.

This single explicit import list replaces the executor's 30 scattered stage
imports. Auto-discovery (``pkgutil.walk_packages``) is deliberately avoided: it
would import non-stage modules and risk eagerly importing heavy optional
dependencies, violating the skip-not-crash lazy-import invariant.

The stage-import block below is ordered to mirror the canonical pipeline flow
so a reader learns the analysis order top-to-bottom; the *authoritative* order
is still each spec's ``order=`` argument. The block is bracketed by
``# isort: off`` / ``# isort: on`` because ruff's import sorter (rule ``I``)
would otherwise re-alphabetize it and break that narrative reading order.
(``# isort: off`` is the linter directive; ``# fmt: off`` only steers the
formatter and does not stop rule ``I``.)
"""

from __future__ import annotations

# isort: off
from cellquorum.core.stage_catalog import (
    StageSpec,
    iter_stage_specs,
    register_planned_stage,
)

# --- Implemented stages (import fires @register_stage) ---
# Backbone: QC → preprocessing → dim-reduction → integration.
import cellquorum.ambient_correction.stage  # noqa: F401
import cellquorum.qc.stage  # noqa: F401
import cellquorum.preprocessing.stage  # noqa: F401
import cellquorum.preprocessing.feature_selection.stage  # noqa: F401
import cellquorum.preprocessing.dimensionality.stage  # noqa: F401
import cellquorum.integration.stage  # noqa: F401

# Clustering → annotation → sub-structure → label reconciliation.
import cellquorum.clustering.stage  # noqa: F401
import cellquorum.annotation.stage  # noqa: F401
import cellquorum.clustering.subclustering.stage  # noqa: F401
import cellquorum.annotation.adjudication.stage  # noqa: F401
import cellquorum.annotation.reference_mapping.stage  # noqa: F401
import cellquorum.annotation.consensus.stage  # noqa: F401

# Diagnostics run after reference mapping so transferred labels can be audited.
import cellquorum.annotation.diagnostics.stage  # noqa: F401

# Population identity is evidence-driven (reference > annotation > clusters).
import cellquorum.annotation.population_identity.stage  # noqa: F401
import cellquorum.integration.benchmark.stage  # noqa: F401
import cellquorum.integration.embeddings.stage  # noqa: F401

# Comparison + discovery-tail tracks.
import cellquorum.comparative.differential_expression.stage  # noqa: F401
import cellquorum.comparative.differential_abundance.stage  # noqa: F401
import cellquorum.comparative.enrichment.stage  # noqa: F401
import cellquorum.comparative.enrichment.viz.stage  # noqa: F401
import cellquorum.comparative.differential_expression.viz.stage  # noqa: F401
import cellquorum.gene_regulation.coexpression.stage  # noqa: F401
import cellquorum.gene_regulation.grn.stage  # noqa: F401
import cellquorum.gene_regulation.perturbation.stage  # noqa: F401
import cellquorum.trajectory.stage  # noqa: F401
import cellquorum.trajectory.viz.stage  # noqa: F401

# CCC chain runs producer-before-consumer: communication writes the LR tables,
# ccc_network derives topology/curvature, ccc_viz renders from both and MUST be
# last. multicellular_programs sits alongside the CCC track.
import cellquorum.cell_cell_communication.stage  # noqa: F401
import cellquorum.multicellular_programs.stage  # noqa: F401
import cellquorum.cell_cell_communication.network.stage  # noqa: F401
import cellquorum.cell_cell_communication.viz.stage  # noqa: F401
# isort: on

# --- Planned-but-unimplemented stages (no class to decorate) ---
# integration_gate ranks embeddings BEFORE committing expensive clustering.
register_planned_stage(name="integration_gate", order=70, config_flag="integration_gate")
register_planned_stage(name="state_scoring", order=170, config_flag="state_scoring")
register_planned_stage(name="discovery", order=180, config_flag="discovery")
register_planned_stage(name="composition", order=190, config_flag="composition")
register_planned_stage(name="molecular_inference", order=290, config_flag="molecular_inference")


def all_stage_specs() -> tuple[StageSpec, ...]:
    """All registered stage specs (implemented + planned), sorted by order."""
    return iter_stage_specs()
