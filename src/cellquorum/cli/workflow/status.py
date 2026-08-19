"""Aggregate per-run stage status + manifest accounting into a status matrix."""

from __future__ import annotations

import csv
import io

from cellquorum.cli.workflow.scaffold import SCAFFOLD_METHOD_STAGES

_FAIL = "failed"
_OK = "succeeded"
_SKIP = "skipped"
_BLOCKED = "blocked"
# A method the manifest slated to RUN that produced no successful record after an
# attempted/crashed run is honestly "missing" (never reached / crashed before it)
# or "incomplete" (partially recorded), never silently "skipped".
_MISSING = "missing"
_INCOMPLETE = "incomplete"


def _records(stage_records: list[dict] | dict) -> list[dict]:
    """
    Extract records list from either a bare list or a dict wrapper.

    Real stage_execution_records.json is a bare list, but tolerate both forms.

    Args:
        stage_records: Either a bare list of records or a dict with a "records" key.

    Returns:
        List of stage execution records.
    """
    if isinstance(stage_records, dict):
        return stage_records.get("records", [])
    return list(stage_records)


def method_status(
    stage_records: list[dict] | dict,
    run_methods: list[str],
    method_stages: dict[str, list[str]] = SCAFFOLD_METHOD_STAGES,
) -> dict[str, str]:
    """
    Roll up per-stage status into per-method status.

    These are run-methods: the manifest slated them to RUN, so absence of a
    successful record is a real problem, not an intentional skip. The states are
    kept honest:

    - "failed": any of the method's stages recorded status="failed".
    - "missing": the method produced no stage records at all -- the run crashed
      or never reached it. This is NOT "skipped".
    - "incomplete": some stages recorded but others have no record -- a partial
      run (e.g. crashed mid-method).
    - "skipped": every stage was explicitly engine-skipped (status="skipped").
    - "succeeded": every stage is present and each is success (or an intentional
      engine skip) with no failures.

    Args:
        stage_records: List of stage execution records (or dict wrapper).
        run_methods: Methods that were intended to run.
        method_stages: Mapping of method name to list of stage names.

    Returns:
        Dict mapping method name to one of: succeeded, failed, missing,
        incomplete, skipped.
    """
    by_stage = {rec["stage_name"]: rec.get("status", _SKIP) for rec in _records(stage_records)}
    result: dict[str, str] = {}
    for method in run_methods:
        stages = method_stages[method]
        # ``None`` marks a stage with no execution record on disk.
        statuses = [by_stage.get(s) for s in stages]
        present = [s for s in statuses if s is not None]
        if _FAIL in present:
            result[method] = _FAIL
        elif not present:
            # Slated to run but nothing recorded: crashed before/at this method.
            result[method] = _MISSING
        elif None in statuses:
            # Some stages ran, others never recorded: a partial (crashed) run.
            result[method] = _INCOMPLETE
        elif all(s == _SKIP for s in present):
            # Every stage was an explicit, intentional engine skip.
            result[method] = _SKIP
        elif all(s in ("success", _SKIP) for s in present):
            # All stages accounted for, at least one success, no failures.
            result[method] = _OK
        else:
            # Unknown non-success statuses present: not a clean success.
            result[method] = _INCOMPLETE
    return result


def build_matrix(
    accounting: dict,
    run_records: dict[str, list[dict] | dict],
    method_stages: dict[str, list[str]] = SCAFFOLD_METHOD_STAGES,
) -> list[dict]:
    """
    Build a status matrix from accounting and per-run stage records.

    For each hypothesis × cell_type × method, produce a status row.
    Status is one of: succeeded, failed, skipped, blocked.

    Args:
        accounting: Dict mapping hypothesis_id to {"run": [...], "skip": [...], "blocked": [...]}.
        run_records: Dict mapping "<hyp>__<cell_type>" to stage_execution_records.
        method_stages: Mapping of method name to list of stage names.

    Returns:
        List of dicts with keys: hypothesis, cell_type, method, status.
    """
    rows: list[dict] = []
    for hyp_id, acct in accounting.items():
        cell_runs = {
            key.split("__", 1)[1]: recs
            for key, recs in run_records.items()
            if key.startswith(f"{hyp_id}__")
        }
        for cell_type, recs in sorted(cell_runs.items()):
            statuses = method_status(recs, acct["run"], method_stages)
            for method in acct["run"]:
                rows.append(
                    {
                        "hypothesis": hyp_id,
                        "cell_type": cell_type,
                        "method": method,
                        "status": statuses[method],
                    }
                )
            for method in acct.get("skip", []):
                rows.append(
                    {
                        "hypothesis": hyp_id,
                        "cell_type": cell_type,
                        "method": method,
                        "status": _SKIP,
                    }
                )
            for method in acct.get("blocked", []):
                rows.append(
                    {
                        "hypothesis": hyp_id,
                        "cell_type": cell_type,
                        "method": method,
                        "status": _BLOCKED,
                    }
                )
    return rows


_FIELDS = ["hypothesis", "cell_type", "method", "status"]


def matrix_to_csv(rows: list[dict]) -> str:
    """
    Convert status matrix rows to CSV format.

    Args:
        rows: List of dicts with hypothesis, cell_type, method, status keys.

    Returns:
        CSV string with header and rows.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def matrix_to_markdown(rows: list[dict]) -> str:
    """
    Convert status matrix rows to markdown table format.

    Args:
        rows: List of dicts with hypothesis, cell_type, method, status keys.

    Returns:
        Markdown table string.
    """
    lines = ["| " + " | ".join(_FIELDS) + " |", "| " + " | ".join(["---"] * len(_FIELDS)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[f]) for f in _FIELDS) + " |")
    return "\n".join(lines) + "\n"
