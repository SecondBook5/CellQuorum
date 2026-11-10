"""Ensure that a scientific failure survives aggregation and stale reviews are rejected."""

import hashlib
import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_coverage_retains_negative_verdict_and_rejects_changed_inputs(tmp_path, monkeypatch):
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/study_coverage.py"))
    main = module["main"]
    run = tmp_path / "study/runs/reference"
    receipt = run / "provenance/stages/annotation/completion.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"status":"success"}')
    artifact = run / "labels.csv"
    artifact.write_text("cell,prediction\n1,T\n")
    original = tmp_path / "study/truth.csv"
    original.write_text("cell,label\n1,B\n")
    review = tmp_path / "review.csv"
    pd.DataFrame(
        [
            dict(
                stage="annotation",
                method="voting",
                run="study/runs/reference",
                verdict="contradicted",
                criterion="Match the supplied labels",
                result="0/1 match",
                evidence="labels.csv",
                study_inputs="study/truth.csv",
                study_input_sha256=json.dumps(
                    {"study/truth.csv": hashlib.sha256(original.read_bytes()).hexdigest()}
                ),
                evidence_sha256=module["evidence_digest"](run, "annotation", ["labels.csv"]),
            )
        ]
    ).to_csv(review, index=False)
    monkeypatch.setitem(main.__globals__, "catalog", lambda: {"annotation": {"voting"}})
    monkeypatch.setitem(
        main.__globals__,
        "evidence",
        lambda _: [
            ("annotation", "(stage)", "reference", "labels.csv", False),
            ("annotation", "voting", "reference", "labels.csv", False),
        ],
    )
    out = tmp_path / "coverage.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["coverage", str(run), "--output", str(out), "--scientific-review", str(review)],
    )
    main()
    rows = pd.read_csv(out).set_index("method")
    assert rows.loc["voting", "review_verdict"] == "contradicted"
    assert rows.loc["voting", "status"] == "scientific check recorded"
    assert rows.loc["(stage)", "status"] == "executed"
    original.write_text("cell,label\n1,C\n")
    with pytest.raises(ValueError, match="Study input changed"):
        main()
    artifact.write_text("cell,prediction\n1,B\n")
    with pytest.raises(ValueError, match="Scientific evidence changed"):
        main()
