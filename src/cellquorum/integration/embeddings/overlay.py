"""Generic feature-overlay resolver + opt-in scoped MAGIC imputation.

resolve_features expands a config OverlayConfig into a flat list of per-cell
value vectors (genes, program module-scores, cell-cycle scores, obs columns),
skipping anything it cannot resolve with a collected warning (skip-not-crash).
impute_magic_scoped imputes ONLY the requested genes and tags the layer
'imputed' so the statistics guard blocks it.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.integration.embeddings.config import OverlayConfig


class MagicUnavailable(Exception):
    """The optional 'magic' package is not importable."""


@dataclass(frozen=True)
class FeatureValues:
    """One resolved overlay feature.

    Attributes:
        label: Display label / figure stem component.
        values: Per-cell float vector (length n_obs).
        kind: One of 'gene', 'program', 'cell_cycle', 'obs'.
    """

    label: str
    values: np.ndarray
    kind: str


def _gene_vector(adata: ad.AnnData, gene: str, layer: str | None) -> np.ndarray:
    """Return the per-cell expression vector for a gene from X or a layer."""
    idx = adata.var_names.get_loc(gene)
    matrix = adata.layers[layer] if layer is not None else adata.X
    col = matrix[:, idx]
    if sp.issparse(col):
        col = col.toarray()
    return np.asarray(col).ravel().astype(float)


def resolve_features(
    adata: ad.AnnData,
    overlay_cfg: OverlayConfig,
    *,
    random_state: int,
    layer: str | None = None,
) -> tuple[list[FeatureValues], list[str]]:
    """Expand an OverlayConfig into resolved per-cell feature vectors.

    Unresolvable features are skipped with a collected warning. When `layer` is
    given (e.g. the MAGIC layer), gene values are read from that layer.
    """
    features: list[FeatureValues] = []
    warnings: list[str] = []

    # Genes.
    for gene in overlay_cfg.genes:
        if gene in adata.var_names:
            features.append(FeatureValues(gene, _gene_vector(adata, gene, layer), "gene"))
        else:
            warnings.append(f"overlay: gene '{gene}' absent from var_names (skipped)")

    # Programs -> score_genes -> obs column.
    for name, gene_list in overlay_cfg.programs.items():
        present = [g for g in gene_list if g in adata.var_names]
        if not present:
            warnings.append(f"overlay: program '{name}' has no genes present (skipped)")
            continue
        sc.tl.score_genes(adata, present, score_name=name, random_state=random_state)
        features.append(FeatureValues(name, adata.obs[name].to_numpy().astype(float), "program"))

    # Cell cycle.
    if overlay_cfg.cell_cycle:
        s_present = [g for g in overlay_cfg.s_genes if g in adata.var_names]
        g2m_present = [g for g in overlay_cfg.g2m_genes if g in adata.var_names]
        if s_present and g2m_present:
            sc.tl.score_genes_cell_cycle(
                adata, s_genes=s_present, g2m_genes=g2m_present, random_state=random_state
            )
            for col in ("S_score", "G2M_score"):
                features.append(
                    FeatureValues(col, adata.obs[col].to_numpy().astype(float), "cell_cycle")
                )
        else:
            warnings.append("overlay: cell_cycle requested but s_genes/g2m_genes not present")

    # Arbitrary obs columns.
    for col in overlay_cfg.obs_columns:
        if col in adata.obs.columns:
            values = adata.obs[col]
            try:
                arr = values.to_numpy().astype(float)
            except (TypeError, ValueError):
                arr = values.astype("category").cat.codes.to_numpy().astype(float)
            features.append(FeatureValues(col, arr, "obs"))
        else:
            warnings.append(f"overlay: obs column '{col}' absent (skipped)")

    return features, warnings


def impute_magic_scoped(
    adata: ad.AnnData,
    genes: list[str],
    *,
    knn: int,
    solver: str,
    random_state: int,
    layer_out: str = "magic",
) -> list[str]:
    """Impute ONLY `genes` (those present) into layers[layer_out]; tag imputed.

    The imputed layer is full-width (zeros for non-imputed genes) so it stays
    shape-compatible with X, and is tagged kind='imputed' so the statistics
    guard rejects it.
    """
    try:
        import magic
    except ImportError as exc:
        raise MagicUnavailable("magic not installed") from exc

    present = [g for g in genes if g in adata.var_names]
    if not present:
        return []

    dense_X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    operator = magic.MAGIC(knn=knn, solver=solver, random_state=random_state, verbose=0)
    imputed = operator.fit_transform(dense_X, genes="all_genes")
    imputed = np.asarray(imputed)

    out = np.zeros_like(dense_X, dtype=float)
    for gene in present:
        j = adata.var_names.get_loc(gene)
        out[:, j] = imputed[:, j]
    adata.layers[layer_out] = out
    set_layer_tag(adata, layer_out, kind="imputed")
    return present


__all__ = ["FeatureValues", "MagicUnavailable", "resolve_features", "impute_magic_scoped"]
