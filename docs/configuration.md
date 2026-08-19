# Configuration reference

A CellQuorum run is described entirely by a YAML configuration validated against the
`CellQuorumConfig` Pydantic model. Validation is **strict**: every model uses
`extra = "forbid"`, so an unknown or misspelled key raises a configuration error
before any compute runs. Every top-level section has a default, so a minimal config
is small — but explicit is better than implicit for a publication analysis.

A config can be supplied as a file path, a Python `dict`, or a `CellQuorumConfig`
instance (see the [Python API](../README.md#python-api)).

## Top-level sections

| Section | Purpose | Notable fields |
|---|---|---|
| `project` | project metadata | `name`, `organism`, `species_id` |
| `paths` | filesystem roots | `data_root`, `run_root`, `scratch_root`, `manifest`, `output_dir` |
| `input` | the dataset | input `.h5ad` path, counts layer, optional subset |
| `run` | run behavior | `profile`, `run_id`, `random_seed`, `resume`, `overwrite`, `log_level`, `write_final_object`, `continue_on_stage_failure` |
| `compute` | CPU/GPU routing | `backend` (`auto`/`cpu`/`gpu`), `prefer_gpu`, `fallback_to_cpu`, `n_jobs` |
| `r` | R backend | `enabled`, `preferred_backend`, `fallback_to_rscript`, `rscript_path`, `timeout_seconds` |
| `report` | reporting | `html`, `markdown`, `pdf`, `fail_on_report_error` |
| `stages` | stage enable flags | one boolean per stage (see below) |
| `markers` | marker-gene panels | named panels used by marker-based annotation and scoring |
| `cohort` | obs-key schema | the central mapping of `obs` columns (donor, condition, ...) |
| `design` | experimental design | design specification for differential analyses |
| `contrasts` | named contrasts | case/control contrasts referenced by DE/DA |
| *per-stage blocks* | stage parameters | one block per stage, e.g. `qc:`, `integration:`, `enrichment:` |

A whole-config validator rejects contradictory combinations — for example,
`compute.backend: auto` together with `compute.fallback_to_cpu: false`.

## Enabling stages

The `stages:` block is a set of boolean flags, one per stage. All default to `true`
except `ambient_correction` and `integration_gate`, which default to `false`.
Setting a flag to `false` marks the stage disabled; the planner records it as skipped
and the `AnnData` passes through unchanged.

```yaml
stages:
  ambient_correction: false   # off by default
  qc: true
  preprocessing: true
  feature_selection: true
  dimensionality: true
  integration: true
  clustering: true
  annotation: true
  embeddings: true
  differential_expression: true
  differential_abundance: true
  enrichment: true
  coexpression: true
  grn: true
  perturbation: true
  cell_cell_communication: true
  network_analysis: true       # toggles the ccc_network topology stage
  trajectory: true
```

Notes:

- **`network_analysis` toggles the `ccc_network` stage** (topology + curvature). This
  is the one place where the config flag name differs from the stage name.
- **Reserved slots.** The flags `integration_gate`, `state_scoring`, `discovery`,
  `composition`, and `molecular_inference` are accepted for forward compatibility,
  but those stages are not yet implemented — enabling them results in a planned stage
  that is skipped as "not yet implemented", visible in the plan and provenance.

## Selecting a method per stage

Each analysis stage's config block chooses its method with a `method:` key (or
`methods:` for stages that can run several), plus that method's parameters:

```yaml
integration:
  method: harmony            # or: scvi

differential_expression:
  method: pseudobulk_edger   # donor-aware pseudobulk

differential_abundance:
  method: milo               # or: sccoda | propeller | proportion_ttest

enrichment:
  methods: [gsea, ora, gsva] # run several enrichment methods
```

If a selected method's backend is unavailable at runtime (e.g. an R package or an
isolated environment is missing), the stage **skips with a recorded reason** rather
than failing the run. Inspect what a config will actually do — without running it —
with:

```bash
cellquorum plan --config configs/config.yaml
```

which prints, per stage, whether it is enabled, skipped, or unimplemented, and the
backend it would use. See [`configs/config.yaml`](../configs/config.yaml) for a
complete worked example and [architecture.md](architecture.md) for how method
dispatch and data contracts work.

## Generating configs from a manifest

To expand a hypothesis manifest into many per-`(hypothesis, cell_type)` configs, use
the `gen-configs` command:

```bash
gen-configs run --manifest hypotheses.yaml --template template.yaml --out-dir generated/
```

It writes one `generated/configs/<key>.yaml` per combination plus an
`accounting.json` recording what was produced.
