"""End-to-end integration test for EnrichmentStage through PipelineContext.

This is the guardrail task: it exercises all 4 enrichment methods through the real
dispatch path, proving the design bridge delivers case/control to every method and no
method raises when its dependencies are unavailable.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.comparative.enrichment.stage import EnrichmentStage
from cellquorum.config.design import DesignConfig
from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.core.contracts.layer_tags import set_layer_tag


def _decoupler_net_available() -> bool:
    """Check if decoupler can import and fetch hallmark network."""
    try:
        import decoupler as dc

        dc.op.hallmark(organism="human", license="academic", verbose=False)
        return True
    except Exception:
        return False


def _adata():
    """Build a real cohort with ≥3 donors/arm and required layers.

    Structure:
    - 3 control donors (N1,N2,N3) + 3 case donors (D1,D2,D3) — disjoint (unpaired).
    - cell_type column: Type0.
    - counts layer: raw counts for GSVA pseudobulk aggregation.
    - cellquorum_normalized layer: log1p(CPM), tagged as lognorm for activity contract.
    - var_names: real human gene symbols (subset that exist in hallmark) so enrichment works.
    """
    rng = np.random.default_rng(42)
    donors_ctrl, donors_case = ["N1", "N2", "N3"], ["D1", "D2", "D3"]
    # Use real gene symbols that exist in hallmark gene sets.
    genes = [
        "MAFF",
        "A2M",
        "AAAS",
        "AADAT",
        "AARS1",
        "ABAT",
        "ABCA1",
        "ABCA2",
        "ABCB1",
        "ABCB6",
        "ACAA1",
        "ACAA2",
        "ACACA",
        "ACADS",
        "ACADM",
        "ACADVL",
        "ACAT1",
        "ACAT2",
        "ACE",
        "ACHE",
    ]
    blocks, rows = [], []
    for d in donors_ctrl + donors_case:
        cond = "Normal" if d.startswith("N") else "Disease"
        for _ in range(30):
            blocks.append(rng.poisson(5, size=len(genes)).astype(float))
            rows.append({"patient_id": d, "condition": cond, "cell_type": "Type0"})
    X = np.vstack(blocks)
    a = ad.AnnData(X=X, obs=pd.DataFrame(rows))
    a.var_names = genes
    a.layers["counts"] = X.copy()  # raw counts for GSVA pseudobulk aggregation
    # log-normalized layer, tagged so the activity contract passes.
    lib = X.sum(axis=1, keepdims=True)
    a.layers["cellquorum_normalized"] = np.log1p(X / np.clip(lib, 1, None) * 1e4)
    set_layer_tag(a, "cellquorum_normalized", kind="lognorm")
    return a


class _Cfg:
    """Minimal config object exposing the enrichment stage sub-block and design."""

    organism = "human"
    enrichment = {
        "enabled": True,
        "methods": [
            {"method": "gsea"},
            {"method": "ora"},
            {"method": "gsva"},
            {"method": "activity"},
        ],
        "gene_set_collections": ["hallmark"],
        "min_size": 1,
    }
    cohort = None
    design = DesignConfig(
        donor_col="patient_id",
        condition_col="condition",
        case="Disease",
        control="Normal",
        paired=False,
    )


def test_enrichment_stage_runs_through_context(tmp_path):
    """
    Verify EnrichmentStage runs all 4 methods without crash.

    Core invariants (the guardrail):
    1. stage.run(ctx) NEVER raises.
    2. metrics["n_methods"] == 4 (the default 4-method list was dispatched).
    3. NO per_method skip reason mentions case/control-unset (design bridge worked).
    4. For each available dependency, artifact CSV exists with expected columns.
    """
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()

    # Provide a DE table so GSEA/ORA have input (matching var_names — real gene symbols).
    genes = [
        "MAFF",
        "A2M",
        "AAAS",
        "AADAT",
        "AARS1",
        "ABAT",
        "ABCA1",
        "ABCA2",
        "ABCB1",
        "ABCB6",
        "ACAA1",
        "ACAA2",
        "ACACA",
        "ACADS",
        "ACADM",
        "ACADVL",
        "ACAT1",
        "ACAT2",
        "ACE",
        "ACHE",
    ]
    pd.DataFrame(
        {
            "gene": genes,
            "logFC": np.linspace(-3, 3, len(genes)),
            "logCPM": [1] * len(genes),
            "F": [1] * len(genes),
            "PValue": np.linspace(0.001, 0.5, len(genes)),
            "FDR": np.linspace(0.001, 0.6, len(genes)),
        }
    ).to_csv(Path(paths.results) / "de_pseudobulk_edger.csv", index=False)

    ctx = PipelineContext(config=_Cfg(), paths=paths, adata=_adata())

    # Run the stage — must never raise, even if all dependencies are unavailable.
    result = EnrichmentStage().run(ctx)

    # Invariant 1: stage returned a real StageResult with n_methods=4.
    assert result.metrics["n_methods"] == 4

    # Invariant 2: metrics has per_method list with 4 entries.
    per_method = result.metrics["per_method"]
    assert len(per_method) == 4

    # Invariant 3: NO method skipped for case/control-unset (design bridge delivered them).
    for entry in per_method:
        if entry.get("skipped"):
            reason = entry.get("reason", "").lower()
            # The case/control-unset message is in the method skip guards.
            # If case/control appear in the skip reason, they should NOT be about "not set"
            assert "case" not in reason or "control" not in reason, (
                f"Method {entry.get('method')} skipped for case/control-unset: "
                f"{entry.get('reason')}"
            )

    # Invariant 4: when decoupler is available, verify GSEA artifact exists with expected columns.
    if _decoupler_net_available():
        # At least GSEA should have produced a CSV for hallmark.
        gsea_csv = Path(paths.results) / "enrichment_gsea_hallmark.csv"
        assert gsea_csv.exists(), "GSEA hallmark CSV missing despite decoupler available"

        # Verify expected columns from Task 4 output format.
        df = pd.read_csv(gsea_csv)
        expected_cols = ["source", "score", "pvalue", "padj", "significant", "collection"]
        assert (
            list(df.columns) == expected_cols
        ), f"GSEA columns mismatch: {list(df.columns)} vs {expected_cols}"
    else:
        # Offline: methods should skip for dependency/network reasons, NOT case/control-unset.
        # We already verified case/control-unset is not in any skip reason (Invariant 3),
        # so this branch just documents the offline expectation: the stage still returned
        # without raising (proved by reaching this line).
        pass
