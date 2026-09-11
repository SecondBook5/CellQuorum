"""Shared provenance recording for the integration methods (Harmony, scVI, scANVI).

Every method writes the same two records into `adata.uns['cellquorum']`: a single-method
slot (`integration`, backward-compatible, last method to run wins) and a per-method slot
(`integration_methods`, namespaced by `output_rep`) that survives when multiple methods
run against the same object. Extracted so a field added for one method cannot be forgotten
on the other two, which is how they stayed in lockstep by hand until now.
"""

from __future__ import annotations

import anndata as ad


def record_integration_provenance(
    adata: ad.AnnData,
    *,
    method: str,
    batch_key: str,
    output_rep: str,
    **extra: object,
) -> dict:
    """Write both provenance records for one integration method's run.

    Args:
        adata: The object the method wrote its embedding onto.
        method: Registry name of the method ("harmony", "scvi", "scanvi").
        batch_key: The batch column corrected on.
        output_rep: The obsm key the embedding was written to; namespaces the per-method
            record so multiple methods on the same object don't overwrite each other.
        **extra: Method-specific fields (e.g. `input_rep`, `n_latent`, `label_key`).

    Returns:
        The record that was written (same dict content in both slots).
    """
    cq = adata.uns.setdefault("cellquorum", {})
    record = {"method": method, "batch_key": batch_key, "output_rep": output_rep, **extra}
    # Single-method provenance (backward-compatible path, last-wins).
    cq["integration"] = record
    # Per-method provenance (multi-method path, namespaced by output_rep).
    cq.setdefault("integration_methods", {})[output_rep] = dict(record)
    return record


__all__ = ["record_integration_provenance"]
