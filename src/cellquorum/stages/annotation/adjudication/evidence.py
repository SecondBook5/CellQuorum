"""Build adjudication evidence from current AnnData state."""

from __future__ import annotations

import anndata as ad
import pandas as pd

from cellquorum.stages.annotation.adjudication.adjudicate import ClusterEvidence
from cellquorum.stages.annotation.adjudication.config import AdjudicationConfig


def build_cluster_evidence_table(
    adata: ad.AnnData,
    *,
    config: AdjudicationConfig,
    donor_key: str,
    condition_key: str,
) -> list[ClusterEvidence]:
    """
    Build cluster-level evidence records from ``adata.obs``.

    Args:
        adata: AnnData object with cluster, donor, and condition metadata.
        config: Adjudication configuration.
        donor_key: Resolved donor obs column.
        condition_key: Resolved condition obs column.

    Returns:
        One ClusterEvidence object per cluster.

    Raises:
        KeyError: If required obs columns are absent.
    """

    # Validate required metadata columns up front.
    required = [config.cluster_key, donor_key, condition_key]
    missing = [column for column in required if column not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing required adjudication obs column(s): {missing}.")

    # Work with strings for stable grouping/reporting.
    obs = adata.obs.copy()
    obs[config.cluster_key] = obs[config.cluster_key].astype(str)
    obs[donor_key] = obs[donor_key].astype(str)
    obs[condition_key] = obs[condition_key].astype(str)

    # Resolve optional support columns.
    marker_key = _resolve_marker_support_key(obs, config)
    technical_score_key = config.technical_score_key
    technical_flag_key = config.technical_flag_key
    # qc_state_final (written by qc_finalization, order=135) is QC's own final verdict —
    # a cluster dominated by cells that verdict never established as valid is exactly what
    # "technical validity" should mean here, so it outranks the cruder predicted_doublet
    # proxy whenever it is present and the caller has not asked for a specific column.
    use_qc_state_final = technical_score_key is None and "qc_state_final" in obs.columns

    evidence_rows: list[ClusterEvidence] = []
    for cluster_id, cluster_obs in obs.groupby(config.cluster_key, observed=False):
        n_cells = int(cluster_obs.shape[0])
        donor_counts = _value_counts(cluster_obs[donor_key])
        condition_counts = _value_counts(cluster_obs[condition_key])

        marker_support = _mean_optional_numeric(cluster_obs, marker_key)
        qc_state_final_fraction = (
            _qc_technical_fraction(cluster_obs["qc_state_final"]) if use_qc_state_final else None
        )
        technical_score = _mean_optional_numeric(cluster_obs, technical_score_key)
        technical_flag_fraction = _mean_optional_boolean(cluster_obs, technical_flag_key)
        if technical_score is None:
            technical_score = qc_state_final_fraction
        if technical_score is None:
            technical_score = technical_flag_fraction

        evidence_rows.append(
            ClusterEvidence(
                cluster_id=str(cluster_id),
                n_cells=n_cells,
                donor_counts=donor_counts,
                condition_counts=condition_counts,
                marker_support=marker_support,
                technical_score=technical_score,
                notes=_cluster_notes(
                    marker_key=marker_key,
                    technical_score_key=technical_score_key,
                    technical_flag_key=technical_flag_key,
                    used_qc_state_final=qc_state_final_fraction is not None
                    and technical_score == qc_state_final_fraction,
                    used_technical_flag=technical_score is not None
                    and technical_score == technical_flag_fraction,
                ),
            )
        )

    return evidence_rows


def _qc_technical_fraction(qc_state_final: pd.Series) -> float | None:
    """Fraction of a cluster's cells QC's final verdict never established as valid.

    ``core`` and ``rescued`` cells passed QC's own check; ``quarantine`` and
    ``unresolved_borderline`` did not. Neither ``predicted_doublet`` alone (a raw
    detector call the adjudication rules already treat as a technical proxy) nor an
    empty column can express this, so it needs a dedicated reduction.
    """

    if qc_state_final.empty:
        return None
    states = qc_state_final.astype(str)
    return float(states.isin(["quarantine", "unresolved_borderline"]).mean())


def cluster_evidence_to_dataframe(evidence_rows: list[ClusterEvidence]) -> pd.DataFrame:
    """
    Convert cluster evidence records to a flat table.

    Args:
        evidence_rows: Cluster evidence records.

    Returns:
        DataFrame suitable for CSV artifact output.
    """

    rows = []
    for evidence in evidence_rows:
        rows.append(
            {
                "cluster_id": evidence.cluster_id,
                "n_cells": evidence.n_cells,
                "n_donors": evidence.n_donors,
                "n_conditions": evidence.n_conditions,
                "dominant_donor_fraction": evidence.dominant_donor_fraction,
                "dominant_condition_fraction": evidence.dominant_condition_fraction,
                "marker_support": evidence.marker_support,
                "reproducibility_score": evidence.reproducibility_score,
                "technical_score": evidence.technical_score,
                "continuity_score": evidence.continuity_score,
                "split_support": evidence.split_support,
                "donor_counts": dict(evidence.donor_counts),
                "condition_counts": dict(evidence.condition_counts),
            }
        )
    return pd.DataFrame(rows)


def _value_counts(series: pd.Series) -> dict[str, int]:
    """Return stable string-keyed value counts."""

    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _resolve_marker_support_key(obs: pd.DataFrame, config: AdjudicationConfig) -> str | None:
    """Resolve marker support column from config or common annotation defaults."""

    if config.marker_support_key is not None:
        return config.marker_support_key
    for candidate in ("cell_type_conf", "annotation_confidence", "marker_support"):
        if candidate in obs.columns:
            return candidate
    return None


def _mean_optional_numeric(obs: pd.DataFrame, key: str | None) -> float | None:
    """Return the mean of an optional numeric obs column."""

    if key is None or key not in obs.columns:
        return None
    values = pd.to_numeric(obs[key], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _mean_optional_boolean(obs: pd.DataFrame, key: str | None) -> float | None:
    """Return the true fraction of an optional boolean-like obs column."""

    if key is None or key not in obs.columns:
        return None
    values = obs[key]
    if values.empty:
        return None
    if pd.api.types.is_bool_dtype(values):
        return float(values.mean())
    normalized = (
        values.astype(str)
        .str.lower()
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
    )
    normalized = normalized.dropna()
    if normalized.empty:
        return None
    return float(normalized.mean())


def _cluster_notes(
    *,
    marker_key: str | None,
    technical_score_key: str | None,
    technical_flag_key: str | None,
    used_qc_state_final: bool,
    used_technical_flag: bool,
) -> list[str]:
    """Build context notes describing which optional evidence columns were used."""

    notes: list[str] = []
    if marker_key is not None:
        notes.append(f"marker_support derived from obs['{marker_key}'].")
    if technical_score_key is not None:
        notes.append(f"technical_score derived from obs['{technical_score_key}'].")
    elif used_qc_state_final:
        notes.append(
            "technical_score derived from obs['qc_state_final'] "
            "(quarantine + unresolved_borderline fraction)."
        )
    elif used_technical_flag and technical_flag_key is not None:
        notes.append(f"technical_score derived from obs['{technical_flag_key}'] fraction.")
    return notes


__all__ = ["build_cluster_evidence_table", "cluster_evidence_to_dataframe"]
