# Pipeline stages, in run order

The pipeline runs top-to-bottom in this order; each row is a subpackage in this
directory. Order and identity are defined by each stage's `@register_stage`
decorator — this table is test-pinned to the live catalog
(`cellquorum.core.stages.all_stage_specs`), so it cannot drift when stages are
added, removed, or reordered. Each implemented stage's module opens with a
matching `# Pipeline step (order=…)` header; `planned` rows are reserved orders
that have no module yet.

| Order | Stage | Status |
|---|---|---|
| 10 | `ambient_correction` | implemented |
| 20 | `qc` | implemented |
| 30 | `preprocessing` | implemented |
| 40 | `feature_selection` | implemented |
| 50 | `dimensionality` | implemented |
| 60 | `integration` | implemented |
| 70 | `integration_gate` | planned |
| 80 | `clustering` | implemented |
| 90 | `annotation` | implemented |
| 100 | `subclustering` | implemented |
| 110 | `adjudication` | implemented |
| 120 | `reference_mapping` | implemented |
| 130 | `annotation_consensus` | implemented |
| 140 | `annotation_diagnostics` | implemented |
| 150 | `population_identity` | implemented |
| 160 | `integration_benchmark` | implemented |
| 170 | `state_scoring` | implemented |
| 180 | `discovery` | implemented |
| 190 | `composition` | planned |
| 200 | `embeddings` | implemented |
| 210 | `differential_expression` | implemented |
| 220 | `differential_abundance` | implemented |
| 230 | `enrichment` | implemented |
| 240 | `enrichment_viz` | implemented |
| 250 | `de_viz` | implemented |
| 260 | `coexpression` | implemented |
| 270 | `grn` | implemented |
| 280 | `perturbation` | implemented |
| 290 | `molecular_inference` | planned |
| 300 | `trajectory` | implemented |
| 310 | `trajectory_viz` | implemented |
| 320 | `cell_cell_communication` | implemented |
| 330 | `multicellular_programs` | implemented |
| 340 | `ccc_network` | implemented |
| 350 | `ccc_viz` | implemented |
