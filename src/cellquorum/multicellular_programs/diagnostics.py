"""Pure-math diagnostics for multicellular programs.

No I/O, no AnnData — just numpy/pandas transformations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def donor_support(
    scores: pd.DataFrame,
    donor_map: dict[str, str],
    donor_support_min: int,
) -> pd.DataFrame:
    """Compute donor support for each multicellular program.

    Active cells are those with score > that program's mean score.
    Donor support = number of distinct donors among active cells.

    Parameters
    ----------
    scores
        DataFrame with columns: cell_id, program, score
    donor_map
        Maps cell_id -> donor
    donor_support_min
        Minimum number of donors required for a program to be "supported"

    Returns
    -------
    DataFrame with columns: program, n_donors, donor_fraction, supported
    """
    results = []
    total_donors = len(set(donor_map.values()))

    for program, group in scores.groupby("program"):
        mean_score = group["score"].mean()
        # Coerce cell_id to str: donor_map is str-keyed, but scores read back from
        # CSV arrive as int when cell barcodes are purely numeric -- an int/str key
        # mismatch would silently drop every cell and report 0 donors (wrong-but-
        # plausible), so match the donor_map key type explicitly.
        active_cells = group[group["score"] > mean_score]["cell_id"].astype(str)
        active_donors = {donor_map[cell_id] for cell_id in active_cells if cell_id in donor_map}
        n_donors = len(active_donors)
        donor_fraction = n_donors / total_donors if total_donors > 0 else 0.0
        supported = n_donors >= donor_support_min

        results.append(
            {
                "program": program,
                "n_donors": n_donors,
                "donor_fraction": donor_fraction,
                "supported": supported,
            }
        )

    return pd.DataFrame(results)


def match_program_loadings(
    full: pd.DataFrame,
    resample: pd.DataFrame,
) -> dict[str, float]:
    """Match full-run programs to resample programs via Pearson correlation.

    For each program in full, find the best-matching program in resample
    by aligning their (cell_type, gene) -> loading vectors on the union
    of keys and computing Pearson r.

    Parameters
    ----------
    full
        DataFrame with columns: program, cell_type, gene, loading
    resample
        DataFrame with columns: program, cell_type, gene, loading

    Returns
    -------
    Dictionary mapping full_program -> best_r (max |Pearson r| across resample
    programs). DIALOGUE program loading signs are arbitrary between independent
    runs (like PCA/CCA components), so magnitude is matched: a reproduced-but-
    sign-flipped program still scores as reproducible.
    """
    results = {}

    for full_program in full["program"].unique():
        full_subset = full[full["program"] == full_program]
        full_subset = full_subset.set_index(["cell_type", "gene"])["loading"]

        best_r = 0.0
        for resample_program in resample["program"].unique():
            resample_subset = resample[resample["program"] == resample_program]
            resample_subset = resample_subset.set_index(["cell_type", "gene"])["loading"]

            # Align on union of indices
            all_keys = full_subset.index.union(resample_subset.index)
            full_vec = full_subset.reindex(all_keys, fill_value=0.0).values
            resample_vec = resample_subset.reindex(all_keys, fill_value=0.0).values

            # Handle constant vectors (zero variance)
            if np.std(full_vec) == 0 or np.std(resample_vec) == 0:
                r = 0.0
            else:
                corr_matrix = np.corrcoef(full_vec, resample_vec)
                r = corr_matrix[0, 1]
                # Handle potential nan from corrcoef
                if np.isnan(r):
                    r = 0.0

            # Match on |r|: MCP loading sign is arbitrary between independent
            # runs, so a sign-flipped-but-reproducible program must not score 0.
            best_r = max(best_r, abs(r))

        results[full_program] = best_r

    return results


def program_stability(
    matches: list[dict[str, float]],
) -> pd.DataFrame:
    """Average best-match correlations per program across resamples.

    Parameters
    ----------
    matches
        List of match dictionaries (one per resample), each mapping program -> best_r

    Returns
    -------
    DataFrame with columns: program, mean_stability, n_resamples
    """
    # Collect all programs
    all_programs = set()
    for match_dict in matches:
        all_programs.update(match_dict.keys())

    results = []
    n_resamples = len(matches)

    for program in sorted(all_programs):
        r_values = [match_dict.get(program, 0.0) for match_dict in matches]
        mean_stability = np.mean(r_values)

        results.append(
            {
                "program": program,
                "mean_stability": mean_stability,
                "n_resamples": n_resamples,
            }
        )

    return pd.DataFrame(results)
