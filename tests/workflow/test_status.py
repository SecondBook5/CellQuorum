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


def test_run_method_with_no_records_after_crash_is_missing_not_skipped() -> None:
    # A method the manifest slated to RUN, but the run crashed before recording any
    # of its stages, must NOT be reported as an intentional "skipped".
    status = method_status([], run_methods=["pathway_enrichment", "subclustering"])
    assert status["pathway_enrichment"] == "missing"
    assert status["subclustering"] == "missing"
    # The dict-wrapped empty form (what a fully-crashed run is fed) behaves the same.
    wrapped = method_status({"records": []}, run_methods=["pathway_enrichment"])
    assert wrapped["pathway_enrichment"] == "missing"


def test_partial_run_is_incomplete_not_skipped() -> None:
    # Only one of pathway_enrichment's two stages recorded => partial, not skipped.
    records = [{"stage_name": "enrichment", "status": "success"}]
    status = method_status(records, run_methods=["pathway_enrichment"])
    assert status["pathway_enrichment"] == "incomplete"


def test_engine_skipped_method_is_skipped() -> None:
    # Every stage of the method explicitly engine-skipped => the method is skipped.
    records = [
        {"stage_name": "enrichment", "status": "skipped"},
        {"stage_name": "enrichment_viz", "status": "skipped"},
    ]
    status = method_status(records, run_methods=["pathway_enrichment"])
    assert status["pathway_enrichment"] == "skipped"


def test_success_method_is_succeeded() -> None:
    records = [
        {"stage_name": "enrichment", "status": "success"},
        {"stage_name": "enrichment_viz", "status": "success"},
    ]
    status = method_status(records, run_methods=["pathway_enrichment"])
    assert status["pathway_enrichment"] == "succeeded"


def test_manifest_blocked_method_is_blocked() -> None:
    acct = {"il33_axis": {"run": [], "skip": [], "blocked": ["rna_velocity"]}}
    rows = build_matrix(acct, {"il33_axis__KC": []})
    by_method = {(r["method"], r["status"]) for r in rows}
    assert ("rna_velocity", "blocked") in by_method


def test_missing_provenance_pair_reports_run_methods_as_missing() -> None:
    # A pair whose stage records are absent (crashed run) must still surface a row
    # per run-method, honestly marked missing -- not omitted, not skipped.
    acct = {"il33_axis": {"run": ["pathway_enrichment"], "skip": [], "blocked": []}}
    rows = build_matrix(acct, {"il33_axis__KC": {"records": []}})
    statuses = {(r["cell_type"], r["method"], r["status"]) for r in rows}
    assert ("KC", "pathway_enrichment", "missing") in statuses


def test_csv_has_header() -> None:
    rows = [{"hypothesis": "h", "cell_type": "KC", "method": "m", "status": "succeeded"}]
    csv = matrix_to_csv(rows)
    assert csv.splitlines()[0] == "hypothesis,cell_type,method,status"
