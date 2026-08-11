from __future__ import annotations

from cellquorum.workflow.status import build_matrix, matrix_to_csv, method_status

STAGE_RECORDS = [
    {"stage_name": "qc", "status": "success"},
    {"stage_name": "enrichment", "status": "success"},
    {"stage_name": "enrichment_viz", "status": "success"},
    {"stage_name": "subclustering", "status": "failed"},
    {"stage_name": "adjudication", "status": "skipped"},
]


def test_method_status_rolls_up_stage_status() -> None:
    status = method_status(STAGE_RECORDS, run_methods=["pathway_enrichment", "subclustering"])
    assert status["pathway_enrichment"] == "succeeded"
    # any failed stage in the method -> failed
    assert status["subclustering"] == "failed"


def test_build_matrix_includes_skip_and_blocked() -> None:
    acct = {
        "il33_axis": {
            "run": ["pathway_enrichment", "subclustering"],
            "skip": ["pseudobulk"],
            "blocked": ["rna_velocity"],
        }
    }
    rows = build_matrix(acct, {"il33_axis__KC": STAGE_RECORDS})
    by_method = {(r["method"], r["status"]) for r in rows}
    assert ("pseudobulk", "skipped") in by_method
    assert ("rna_velocity", "blocked") in by_method
    assert ("pathway_enrichment", "succeeded") in by_method
    assert ("subclustering", "failed") in by_method


def test_csv_has_header() -> None:
    rows = [{"hypothesis": "h", "cell_type": "KC", "method": "m", "status": "succeeded"}]
    csv = matrix_to_csv(rows)
    assert csv.splitlines()[0] == "hypothesis,cell_type,method,status"
