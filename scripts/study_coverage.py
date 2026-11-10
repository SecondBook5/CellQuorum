"""Build the release coverage matrix from the public reference runs supplied on the command line."""

import argparse
import ast
import hashlib
import json
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import pandas as pd

import cellquorum.stages
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.stages import all_stage_specs
from cellquorum.methods.registry import METHOD_REGISTRY


@lru_cache(maxsize=512)
def file_digest(path: Path, size: int, mtime_ns: int) -> str:
    """Cache shared artifacts for this invocation; stat values invalidate stale entries."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def evidence_digest(run: Path, stage: str, paths: list[str]) -> str:
    """Bind an external review to the stage receipt and inspected artifact bytes."""
    root = run.resolve()
    digest = hashlib.sha256()
    for name in sorted({*paths, f"provenance/stages/{stage}/completion.json"}):
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Invalid review artifact path: {name}")
        stat = path.stat()
        if not path.is_file() or not stat.st_size:
            raise ValueError(f"Missing review artifact: {path}")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(file_digest(path, stat.st_size, stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def catalog() -> dict[str, set[str]]:
    """Include optional methods whose runtime import is gated by an installed dependency."""
    methods = {s.name: set(METHOD_REGISTRY.names(s.category or s.name)) for s in all_stage_specs()}
    for path in Path(cellquorum.stages.__file__).parent.rglob("*.py"):
        for cls in ast.parse(path.read_text(encoding="utf8")).body:
            if not isinstance(cls, ast.ClassDef):
                continue
            names = {}
            for node in cls.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            names[target.id] = node.value.value
            if names.get("stage_category") in methods and isinstance(names.get("name"), str):
                methods[names["stage_category"]].add(names["name"])
    # Some single-method stages implement their configured backend directly (SoupX).
    defaults = CellQuorumConfig.model_validate({"project": {"name": "catalog"}})
    for spec in all_stage_specs():
        method = getattr(getattr(defaults, spec.config_field, None), "method", None)
        if spec.name in methods and isinstance(method, str):
            methods[spec.name].add(method)
    return methods


def recorded_path(value: str, root: Path) -> str | None:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if path.is_relative_to(root) and path.is_file() and path.stat().st_size:
        return str(path.relative_to(root))
    return None


def evidence(run: Path) -> Iterator[tuple[str, str, str, str, bool]]:
    root = run.resolve()
    provenance = root / "provenance"
    settings = json.loads((provenance / "resolved_config.json").read_text())
    project = settings["project"]["name"]
    planned = {
        p["name"]: p.get("methods", [])
        for p in json.loads((provenance / "pipeline_plan.json").read_text())["stages"]
    }
    check = root / "reports/reference_validation.json"
    checked = json.loads(check.read_text())["checked_artifacts"] if check.exists() else {}
    specs = {s.name: s for s in all_stage_specs()}
    for record in json.loads((provenance / "stage_execution_records.json").read_text()):
        stage = record["stage_name"]
        if stage not in specs or record["status"] != "success":
            continue
        paths = [recorded_path(a["path"], root) for a in record["output_artifacts"]]
        trace = f"provenance/stages/{stage}/completion.json"
        verified_trace = recorded_path(trace, root)
        paths = [p for p in paths if p] or ([verified_trace] if verified_trace else [])
        if not paths:
            raise ValueError(f"{project} has no on-disk execution evidence for {stage}")
        metrics = record["metrics"]
        if stage == "annotation_diagnostics" and metrics.get("n_diagnostics") == 0:
            continue
        from_record = {}
        for entry in metrics.get("per_method", []):
            if not entry.get("skipped") and entry.get("status", "success") == "success":
                files = [recorded_path(v, root) for v in entry.get("artifacts", [])]
                method_files = [v for v in files if v]
                if not method_files and not verified_trace:
                    raise ValueError(
                        f"{project} has no on-disk evidence for {stage}.{entry['method']}"
                    )
                from_record[entry["method"]] = method_files or [str(verified_trace)]
        if not metrics.get("per_method"):
            names = planned.get(stage) or [
                metrics.get("method") or settings.get(specs[stage].config_field, {}).get("method")
            ]
            if len(names) == 1 and names[0]:
                from_record[names[0]] = paths
        for method, files in [("(stage)", paths), *from_record.items()]:
            outcome = checked.get(stage if method == "(stage)" else f"{stage}.{method}")
            if outcome:
                if not recorded_path(outcome, root):
                    raise ValueError(f"{project} scientific evidence is missing: {outcome}")
                files = [outcome]
            yield (stage, method, project, files[0], bool(outcome))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero if any scientific check is outstanding.",
    )
    parser.add_argument(
        "--scientific-review",
        type=Path,
        help="Attach per-method verdicts after verifying the run and artifact digests.",
    )
    args = parser.parse_args()
    reviews = {}
    if args.scientific_review is not None:
        for row in pd.read_csv(args.scientific_review, keep_default_na=False).itertuples():
            key = (row.stage, row.method)
            run_id = Path(row.run)
            if (
                key in reviews
                or run_id.is_absolute()
                or ".." in run_id.parts
                or row.verdict not in {"supported", "limited", "contradicted"}
                or not row.criterion
                or not row.result
            ):
                raise ValueError(f"Invalid scientific review for {key}")
            matches = [
                r.resolve()
                for r in args.runs
                if r.resolve().parts[-len(run_id.parts) :] == run_id.parts
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one reviewed run for {row.run}: {key}")
            paths = row.evidence.split(" | ")
            if evidence_digest(matches[0], row.stage, paths) != row.evidence_sha256:
                raise ValueError(f"Scientific evidence changed for {row.run}: {key}")
            study_root = matches[0].parents[len(run_id.parts) - 1]
            inputs = json.loads(row.study_input_sha256)
            if not inputs or set(inputs) != set(row.study_inputs.split(" | ")):
                raise ValueError(f"Incomplete study input list for {key}")
            for name, expected in inputs.items():
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError(f"Invalid study input: {name}")
                raw = study_root / name
                stat = raw.stat()
                if file_digest(raw, stat.st_size, stat.st_mtime_ns) != expected:
                    raise ValueError(f"Study input changed for {key}: {name}")
            reviews[key] = (row, matches[0])
    observed_reviews = set()
    observations = {}
    methods = catalog()
    for run in args.runs:
        for stage, method, project, path, checked in evidence(run):
            key = (stage, method)
            review = reviews.get(key)
            if review and review[1] == run.resolve():
                observed_reviews.add(key)
                checked = True
                path = review[0].evidence.split(" | ")[0]
            observations.setdefault(key, []).append((project, path, checked))
            if method != "(stage)":
                methods[stage].add(method)
    if observed_reviews != set(reviews):
        raise ValueError(
            f"Reviewed method lacks successful execution: {set(reviews) - observed_reviews}"
        )
    rows = []
    for stage, entries in methods.items():
        for method in ["(stage)", *sorted(entries)]:
            records = observations.get((stage, method), [])
            status = (
                "scientific check recorded"
                if any(r[2] for r in records)
                else "executed"
                if records
                else "unverified"
            )
            review = reviews.get((stage, method))
            rows.append(
                dict(
                    stage=stage,
                    method=method,
                    status=status,
                    studies=" | ".join(r[0] for r in records),
                    evidence=" | ".join(r[1] for r in records),
                    checked_in=" | ".join(r[0] for r in records if r[2]),
                    **(
                        {
                            "review_verdict": review[0].verdict if review else "",
                            "review_result": review[0].result if review else "",
                        }
                        if args.scientific_review
                        else {}
                    ),
                )
            )
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    table.to_csv(args.output, index=False)
    print(
        table.groupby(
            [table.method.eq("(stage)").map({True: "stages", False: "methods"}), "status"]
        )
        .size()
        .to_string()
    )
    if args.require_complete and not table.status.eq("scientific check recorded").all():
        raise SystemExit("Public-study scientific coverage is incomplete.")


if __name__ == "__main__":
    main()
