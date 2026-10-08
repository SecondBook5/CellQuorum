"""Deterministic synthetic 6-subject / 12-specimen split-plot data for pseudobulk_dream tests.

Generic labels (``arm`` between subjects, ``site`` within subjects); no study-specific
names. Cells are Poisson draws from a per-specimen gene rate, so a pseudobulk is a sum of
cell counts and each effect is a known log-rate shift.
"""

from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.config.design import ContrastsConfig, DesignConfig
from cellquorum.core.contracts.layer_tags import set_layer_tag

SUBJECTS = {"C1": "ctl", "C2": "ctl", "C3": "ctl", "P2": "trt", "P3": "trt", "P4": "trt"}
SITES = {"colon": "C", "jej": "J"}
N_GENES = 120
EFFECT_GENES = slice(0, 24)  # the genes carrying the scenario's effect

FIVE_CONTRASTS = [
    {"kind": "simple_effect", "name": "trt_vs_ctl_in_colon", "factor": "arm",
     "numerator": "trt", "denominator": "ctl", "at": {"site": "colon"}},
    {"kind": "simple_effect", "name": "trt_vs_ctl_in_jej", "factor": "arm",
     "numerator": "trt", "denominator": "ctl", "at": {"site": "jej"}},
    {"kind": "simple_effect", "name": "jej_vs_colon_in_ctl", "factor": "site",
     "numerator": "jej", "denominator": "colon", "at": {"arm": "ctl"}},
    {"kind": "simple_effect", "name": "jej_vs_colon_in_trt", "factor": "site",
     "numerator": "jej", "denominator": "colon", "at": {"arm": "trt"}},
    {"kind": "difference_of_differences", "name": "arm_x_site",
     "factor_a": "arm", "numerator_a": "trt", "denominator_a": "ctl",
     "factor_b": "site", "numerator_b": "jej", "denominator_b": "colon"},
]  # fmt: skip


def design_dict(**overrides) -> dict:
    base = {
        "subject_col": "subject",
        "sample_col": "specimen",
        "factors": {
            "arm": {"reference": "ctl", "scope": "between_subject"},
            "site": {"reference": "colon", "scope": "within_subject"},
        },
        "interactions": [["arm", "site"]],
        "random_effects": ["subject"],
    }
    base.update(overrides)
    return base


def make_config(de: dict | None = None, factorial: dict | None = None, contrasts=None):
    """A minimal config object as the DE stage reads it (design + contrasts + DE block)."""
    de_block = {
        "enabled": True,
        "method": "pseudobulk_dream",
        "layer": "counts",
        "cell_type_col": "cell_type",
        "min_count": 5,
        "min_total_count": 10,
        "pseudobulk": {"min_cells_per_sample": 20},
    }
    de_block.update(de or {})
    return SimpleNamespace(
        differential_expression=de_block,
        design=DesignConfig.model_validate({"factorial": factorial or design_dict()}),
        contrasts=ContrastsConfig.model_validate(
            {"factorial": FIVE_CONTRASTS if contrasts is None else contrasts}
        ),
        cohort=None,
    )


def make_adata(
    *,
    effect: str = "none",
    magnitude: float = 1.0,
    cells: dict[tuple[str, str], int] | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Build cells for cell types ``A`` (everywhere) and optionally others.

    Args:
        effect: ``none``, ``interaction`` (only trt x jej shifted), ``arm`` (trt shifted in
            both sites) or ``site`` (jej shifted in both arms). Natural-log shift.
        magnitude: Size of the log-rate shift on the effect genes.
        cells: ``{(cell_type, specimen): n_cells}``; unspecified A cells default to 40.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.4, 1.0, N_GENES)
    obs_rows: list[dict] = []
    blocks: list[np.ndarray] = []
    subj_off = {s: rng.normal(0, 0.15, N_GENES) for s in SUBJECTS}
    n_by = dict(cells or {})
    for subject, arm in SUBJECTS.items():
        for site, code in SITES.items():
            specimen = f"{subject}{code}"
            log_rate = base + subj_off[subject]
            shift = np.zeros(N_GENES)
            hit = {
                "interaction": arm == "trt" and site == "jej",
                "arm": arm == "trt",
                "site": site == "jej",
                "none": False,
            }[effect]
            if hit:
                shift[EFFECT_GENES] = magnitude
            rate = np.exp(log_rate + shift)
            cell_types = sorted({ct for ct, _ in n_by} | {"A"})
            for ct in cell_types:
                n = n_by.get((ct, specimen), 40 if ct == "A" else 0)
                if n == 0:
                    continue
                blocks.append(rng.poisson(rate, size=(n, N_GENES)).astype(float))
                row = {"specimen": specimen, "subject": subject, "arm": arm, "site": site}
                obs_rows += [{**row, "cell_type": ct}] * n
    obs = pd.DataFrame(obs_rows)
    obs.index = [f"cell{i}" for i in range(len(obs))]
    x = sp.csr_matrix(np.vstack(blocks))
    adata = ad.AnnData(X=x.copy(), obs=obs)
    adata.var_names = [f"g{i}" for i in range(N_GENES)]
    adata.layers["counts"] = x
    set_layer_tag(adata, "counts", kind="counts")
    return adata
