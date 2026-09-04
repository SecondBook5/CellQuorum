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
from cellquorum.stages.integration.embeddings.config import OverlayConfig


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


def _expression_layer(
    adata: ad.AnnData, overlay_cfg: OverlayConfig, layer: str | None
) -> tuple[str | None, list[str]]:
    """Which layer the values come from: the imputed one, else the declared one.

    ``layer`` is MAGIC's output when imputation ran, and it wins because that is
    what the caller asked to paint. Otherwise the declared expression layer is used,
    and falling back to ``adata.X`` is a warning rather than a default: X is raw
    counts in this engine, ``score_genes`` over counts is a library-depth readout,
    and the resulting score is written to ``obs`` where nothing downstream can tell
    which of the two it got.
    """
    if layer is not None:
        return layer, []
    declared = overlay_cfg.layer
    if declared is None:
        return None, []
    if declared in adata.layers:
        return declared, []
    return None, [
        f"overlay: layer '{declared}' absent, so genes and program scores are read "
        "from adata.X — if X holds raw counts these values are depth-driven"
    ]


def resolve_features(
    adata: ad.AnnData,
    overlay_cfg: OverlayConfig,
    *,
    random_state: int,
    layer: str | None = None,
) -> tuple[list[FeatureValues], list[str]]:
    """Expand an OverlayConfig into resolved per-cell feature vectors.

    Unresolvable features are skipped with a collected warning. Values are read
    from ``layer`` when one is given (the MAGIC layer), else from the overlay's
    declared expression layer.
    """
    features: list[FeatureValues] = []
    layer, warnings = _expression_layer(adata, overlay_cfg, layer)

    # Genes.
    for gene in overlay_cfg.genes:
        if gene in adata.var_names:
            features.append(FeatureValues(gene, _gene_vector(adata, gene, layer), "gene"))
        else:
            warnings.append(f"overlay: gene '{gene}' absent from var_names (skipped)")

    # Programs -> score_genes -> obs column. Scored on the same layer the gene
    # panels are read from: a program whose members are drawn on the normalized
    # layer but whose score comes off counts is not a summary of the panel beside it.
    for name, gene_list in overlay_cfg.programs.items():
        present = [g for g in gene_list if g in adata.var_names]
        if not present:
            warnings.append(f"overlay: program '{name}' has no genes present (skipped)")
            continue
        sc.tl.score_genes(adata, present, score_name=name, random_state=random_state, layer=layer)
        features.append(FeatureValues(name, adata.obs[name].to_numpy().astype(float), "program"))

    # Cell cycle.
    if overlay_cfg.cell_cycle:
        s_present = [g for g in overlay_cfg.s_genes if g in adata.var_names]
        g2m_present = [g for g in overlay_cfg.g2m_genes if g in adata.var_names]
        if s_present and g2m_present:
            sc.tl.score_genes_cell_cycle(
                adata,
                s_genes=s_present,
                g2m_genes=g2m_present,
                random_state=random_state,
                layer=layer,
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
    layer_in: str | None = None,
) -> list[str]:
    """Impute ONLY `genes` (those present) into layers[layer_out]; tag imputed.

    The imputed layer is full-width (zeros for non-imputed genes) so it stays
    shape-compatible with X, and is tagged kind='imputed' so the statistics
    guard rejects it.

    ``layer_in`` names the expression the imputation runs on, and it matters for the
    same reason it matters to ``score_genes``: MAGIC expects library-size-normalized,
    log-transformed input, and ``adata.X`` in this engine is raw counts. ``None``
    keeps the historical behaviour of reading X.
    """
    try:
        import magic
    except ImportError as exc:
        raise MagicUnavailable("magic not installed") from exc

    present = [g for g in genes if g in adata.var_names]
    if not present:
        return []

    use_layer = layer_in is not None and layer_in in adata.layers
    source = adata.layers[layer_in] if use_layer else adata.X
    dense_X = source.toarray() if sp.issparse(source) else np.asarray(source)
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
