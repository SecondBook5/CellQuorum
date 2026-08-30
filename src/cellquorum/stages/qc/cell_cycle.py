"""Cell-cycle scoring (Tirosh et al. 2016 S/G2M gene sets).

Scores each cell for S-phase and G2M-phase signatures on the log-normalized
layer and assigns a coarse phase. Scoring is opt-in (config.enabled) and runs on
the normalized layer, not raw counts.
"""

from __future__ import annotations

import anndata as ad
import scanpy as sc

from cellquorum.stages.qc.config import QCCellCycleConfig

# Tirosh et al. 2016 human S-phase genes.
TIROSH_S_GENES = [
    "MCM5",
    "PCNA",
    "TYMS",
    "FEN1",
    "MCM2",
    "MCM4",
    "RRM1",
    "UNG",
    "GINS2",
    "MCM6",
    "CDCA7",
    "DTL",
    "PRIM1",
    "UHRF1",
    "MLF1IP",
    "HELLS",
    "RFC2",
    "RPA2",
    "NASP",
    "RAD51AP1",
    "GMNN",
    "WDR76",
    "SLBP",
    "CCNE2",
    "UBR7",
    "POLD3",
    "MSH2",
    "ATAD2",
    "RAD51",
    "RRM2",
    "CDC45",
    "CDC6",
    "EXO1",
    "TIPIN",
    "DSCC1",
    "BLM",
    "CASP8AP2",
    "USP1",
    "CLSPN",
    "POLA1",
    "CHAF1B",
    "BRIP1",
    "E2F8",
]

# Tirosh et al. 2016 human G2M-phase genes.
TIROSH_G2M_GENES = [
    "HMGB2",
    "CDK1",
    "NUSAP1",
    "UBE2C",
    "BIRC5",
    "TPX2",
    "TOP2A",
    "NDC80",
    "CKS2",
    "NUF2",
    "CKS1B",
    "MKI67",
    "TMPO",
    "CENPF",
    "TACC3",
    "FAM64A",
    "SMC4",
    "CCNB2",
    "CKAP2L",
    "CKAP2",
    "AURKB",
    "BUB1",
    "KIF11",
    "ANP32E",
    "TUBB4B",
    "GTSE1",
    "KIF20B",
    "HJURP",
    "CDCA3",
    "HN1",
    "CDC20",
    "TTK",
    "CDC25C",
    "KIF2C",
    "RANGAP1",
    "NCAPD2",
    "DLGAP5",
    "CDCA2",
    "CDCA8",
    "ECT2",
    "KIF23",
    "HMMR",
    "AURKA",
    "PSRC1",
    "ANLN",
    "LBR",
    "CKAP5",
    "CENPE",
    "CTCF",
    "NEK2",
    "G2E3",
    "GAS2L3",
    "CBX5",
    "CENPA",
]


def score_cell_cycle(adata: ad.AnnData, config: QCCellCycleConfig) -> dict:
    """
    Score S/G2M cell-cycle signatures and assign a phase.

    Args:
        adata: AnnData whose score layer holds log-normalized expression.
        config: Cell-cycle configuration.

    Returns:
        Metrics dict: genes used per phase and phase counts.
    """

    # Filter the configured gene lists to those present in the object.
    s_present = [g for g in config.s_genes if g in adata.var_names]
    g2m_present = [g for g in config.g2m_genes if g in adata.var_names]

    # Score on the normalized layer via a working copy (do not disturb .X).
    scored = adata.copy()
    scored.X = scored.layers[config.score_layer]
    # random_state is forwarded via **kwargs to the underlying score_genes call
    # so cell-cycle scoring is reproducible across runs with the same config.
    sc.tl.score_genes_cell_cycle(
        scored,
        s_genes=s_present,
        g2m_genes=g2m_present,
        random_state=config.random_state,
    )

    # Copy the scores/phase back onto the real object.
    adata.obs["S_score"] = scored.obs["S_score"].to_numpy()
    adata.obs["G2M_score"] = scored.obs["G2M_score"].to_numpy()
    adata.obs["phase"] = scored.obs["phase"].to_numpy()

    # Return provenance metrics.
    return {
        "n_s_genes_used": len(s_present),
        "n_g2m_genes_used": len(g2m_present),
        "phase_counts": adata.obs["phase"].value_counts().to_dict(),
    }


__all__ = ["TIROSH_G2M_GENES", "TIROSH_S_GENES", "score_cell_cycle"]
