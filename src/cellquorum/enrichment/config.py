"""Configuration for the enrichment / pathway-activity stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class EnrichmentConfig(StrictBaseModel):
    """Gene-set enrichment and TF/pathway activity analysis.

    Runs GSEA/ORA over upstream DE results and GSVA / decoupler activity over the
    dataset. All biological specifics (organism, collections, columns, gene sets)
    come from config/design — no study assumptions in code.

    Attributes:
        enabled: Whether the stage runs.
        methods: Enrichment method registry keys (empty → default list injected).
        layer: Log-normalized expression layer read by GSVA/activity.
        cell_type_col: obs column with cell-type labels (activity aggregation).
        de_results_filename: DE table filename GSEA/ORA read from the results dir.
        gene_set_collections: Collections for GSEA/ORA (via priors.get_net).
        activity_resources: Weighted nets for the activity method.
        gmt_path: Optional user .gmt overriding fetched collections.
        seed: Random seed for stochastic methods.
        min_size: Minimum targets per source (decoupler tmin).
        max_size: Maximum gene-set size (post-filter).
        gsea_permutations: GSEA permutation count.
        lfc_threshold: |logFC| cut for ORA foreground.
        fg_padj: DE FDR cut for ORA foreground.
        min_foreground_genes: Minimum foreground genes to run an ORA direction.
        fdr: Reported significance threshold.
        fdr_method: statsmodels multipletests method.
        license: decoupler license mode.
        timeout_seconds: Reserved for parity with other stages.
    """

    enabled: bool = True
    methods: list[dict] = []
    layer: str = "cellquorum_normalized"
    counts_layer: str = "counts"
    cell_type_col: str = "cell_type"
    de_results_filename: str = "de_pseudobulk_edger.csv"
    gene_set_collections: list[str] = ["hallmark", "reactome"]
    activity_resources: list[str] = ["collectri", "progeny"]
    gmt_path: str | None = None
    seed: int = 42
    min_size: int = 10
    max_size: int = 500
    gsea_permutations: int = 1000
    lfc_threshold: float = 0.0
    fg_padj: float = 0.05
    min_foreground_genes: int = 5
    fdr: float = 0.05
    fdr_method: str = "fdr_bh"
    license: str = "academic"
    timeout_seconds: int = 1800


__all__ = ["EnrichmentConfig"]
