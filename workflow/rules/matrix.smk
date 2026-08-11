rule gen_configs:
    input:
        manifest=str(MANIFEST),
        template=str(TEMPLATE),
    output:
        accounting=str(GEN / "accounting.json"),
        configs=[str(GEN / "configs" / f"{h}__{ct}.yaml") for h, ct in PAIRS],
    params:
        out=lambda w: str(GEN),
    shell:
        "gen-configs run --manifest {input.manifest} --template {input.template} --out-dir {params.out}"

rule run_analysis:
    input:
        config=str(GEN / "configs" / "{hyp}__{ct}.yaml"),
        accounting=str(GEN / "accounting.json"),
    output:
        manifest=str(RUNS / "{hyp}" / "{ct}" / "provenance" / "artifact_manifest.csv"),
    params:
        out=lambda w: str(RUNS / w.hyp / w.ct),
    shell:
        "cellquorum run --config {input.config} --output-dir {params.out}"

def _hyp_run_targets(wildcards):
    cts = manifest[wildcards.hyp]["cell_types"]
    return [str(RUNS / wildcards.hyp / ct / "provenance" / "artifact_manifest.csv") for ct in cts]

rule bundle_hypothesis:
    input:
        _hyp_run_targets,
    output:
        report=str(BUNDLES / "{hyp}" / "report.html"),
    run:
        from cellquorum.workflow.bundle import assemble_bundle
        entry = manifest[wildcards.hyp]
        run_dirs = {ct: RUNS / wildcards.hyp / ct for ct in entry["cell_types"]}
        assemble_bundle(wildcards.hyp, entry.get("title", wildcards.hyp),
                        run_dirs, BUNDLES / wildcards.hyp)

rule aggregate_status:
    input:
        [str(RUNS / h / ct / "provenance" / "artifact_manifest.csv") for h, ct in PAIRS],
        accounting=str(GEN / "accounting.json"),
    output:
        csv=str(RUNS / "matrix_status.csv"),
        md=str(RUNS / "matrix_status.md"),
    run:
        import json
        from cellquorum.workflow.status import build_matrix, matrix_to_csv, matrix_to_markdown
        accounting = json.loads(Path(input.accounting).read_text())
        run_records = {}
        for h, ct in PAIRS:
            rec = RUNS / h / ct / "provenance" / "stage_execution_records.json"
            run_records[f"{h}__{ct}"] = json.loads(rec.read_text()) if rec.exists() else {"records": []}
        rows = build_matrix(accounting, run_records)
        Path(output.csv).write_text(matrix_to_csv(rows))
        Path(output.md).write_text(matrix_to_markdown(rows))
