"""Configuration for the ccc_network (topology + curvature) stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class CCCNetworkConfig(StrictBaseModel):
    """Graph topology + Ollivier-Ricci curvature over an LR edge table.

    All biology comes from config/input. ``source_key`` names the ``uns`` key
    holding the upstream LR result (spec #1's ``liana_res``); everything else is
    structural/numeric. No study assumptions live in code.

    Attributes:
        enabled: Whether the stage runs.
        methods: Method sub-configs (empty -> default [topology, ricci] injected).
        source_key: uns key holding the upstream per-sample LR frame.
        weight_from: Column in the source frame used to derive edge weight.
        build_gci: Whether to build the gene-channel (GCI) network.
        gci_max_edges: Skip GCI (with a note) when it would exceed this many edges.
        ricci_alpha: Ollivier-Ricci laziness parameter.
        pagerank_alpha: Bayesian prior for the comparative PageRank log-ratio.
        seed: Random seed for stochastic steps.
        min_edges: Minimum edges for a network to be scored.
    """

    enabled: bool = True
    methods: list[dict] = []
    source_key: str = "liana_res"
    weight_from: str = "magnitude_rank"
    build_gci: bool = True
    gci_max_edges: int = 200_000
    ricci_alpha: float = 0.5
    pagerank_alpha: float = 0.01
    seed: int = 42
    min_edges: int = 1


__all__ = ["CCCNetworkConfig"]
