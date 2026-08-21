# Configuration reference

A CellQuorum run is described entirely by a YAML configuration validated against the
`CellQuorumConfig` Pydantic model. Validation is **strict**: every model uses
`extra = "forbid"`, so an unknown or misspelled key raises a configuration error
before any compute runs. Every top-level section has a default, so a minimal config
is small — but explicit is better than implicit for a publication analysis.

A config can be supplied as a file path, a Python `dict`, or a `CellQuorumConfig`
instance (see the [Python API](api.md)).

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

### The `r` backend section

`r.enabled` gates whether R-backed methods (edgeR pseudobulk DE, Milo, propeller,
NicheNet, MultiNicheNet, DIALOGUE, scDiagnostics, SoupX) may run, and `r.rscript_path` selects
the `Rscript` executable used to reach them — both are honored. Set `r.rscript_path`
to an explicit path when `Rscript` is not on the CLI environment's `PATH`, as in the
layered Docker image (see [`docker.md`](docker.md)).

`r.preferred_backend` and `r.fallback_to_rscript` are **reserved and not yet
honored**: every R-backed method currently dispatches through the Rscript backend
regardless of their values. They are accepted by the schema (and may appear in
example configs) so the field is explicit rather than silently dropped, but there is
no in-process rpy2 dispatch for the bundled R scripts today. Wiring rpy2 dual-dispatch
is tracked as future work; until then, setting `preferred_backend: r` does not change
behavior.

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
  state_scoring: true
  discovery: true
  embeddings: true
  differential_expression: true
  differential_abundance: true
  enrichment: true
  coexpression: true
  grn: true
  perturbation: true
  cell_cell_communication: true
  multicellular_programs: true
  network_analysis: true       # toggles the ccc_network topology stage
  trajectory: true
```

Notes:

- **`network_analysis` toggles the `ccc_network` stage** (topology + curvature). This
  is the one place where the config flag name differs from the stage name.
- **Reserved slots.** The flags `integration_gate`, `composition`, and
  `molecular_inference` are accepted for forward compatibility, but those stages are
  not yet implemented — enabling them results in a planned stage that is skipped as
  "not yet implemented", visible in the plan and provenance.

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

state_scoring:
  methods:                   # curated cell-state programs; runs both scorers by default
    - method: score_genes    #   scanpy score_genes → obs["state_<program>"]
    - method: aucell         #   decoupler AUCell   → obsm["score_aucell"]

discovery:
  method: nmf                # de-novo consensus-NMF program discovery
  n_components: 10           # number of programs (rank k)
  n_runs: 20                 # replicate factorizations consensus-clustered into k programs
  use_hvg: true              # restrict to highly-variable genes when present

multicellular_programs:
  method: dialogue           # cross-cell-type coordinated programs via DIALOGUE
  n_programs: 5              # number of programs to infer
  n_program_genes: 200       # genes per program
  min_cell_types: 2          # minimum cell types required
  min_samples: 4             # minimum samples required
  stability_resamples: 5     # resamples for program stability diagnostic
```

If a selected method's backend is unavailable at runtime (e.g. an R package or an
isolated environment is missing), the stage **skips with a recorded reason** rather
than failing the run. Inspect what a config will actually do — without running it —
with:

```bash
cellquorum plan --config configs/config.yaml
```

which prints, per stage, whether it is enabled, skipped, or unimplemented, and the
backend it would use. See [`configs/config.yaml`](https://github.com/SecondBook5/cellquorum/blob/main/configs/config.yaml) for a
complete worked example and [architecture.md](architecture.md) for how method
dispatch and data contracts work.

## Experimental design

The biological question is declared once in the `design` block (donor/condition
columns, case/control tokens, pairing) and consumed by the statistical stages. The
differential-expression engine supports three design shapes, and validates
estimability before any fit — a non-estimable design halts with a clear
configuration error rather than reaching the backend and producing a meaningless
coefficient.

**Pairwise (the default): case vs control.** Requires both arms and at least two
donors per arm. When every donor contributes both a case and a control sample the
design is auto-promoted to **paired** (a donor block, `~ donor + condition`), which
removes inter-donor baseline variance; incomplete pairs are restricted out. Set
`design.paired: true` to require pairing explicitly.

```yaml
design:
  donor_col: patient_id
  condition_col: condition
  case: LE
  control: Normal
  paired: true
```

**Covariate-adjusted.** Additive nuisance terms enter the model as
`~ [covariates +] [donor +] condition`. A categorical covariate perfectly aliased
with the tested condition (e.g. a batch that coincides with case/control) is
rank-deficient and halts loudly.

```yaml
differential_expression:
  covariates: [sex, batch]     # additive adjustment; still tests case vs control
```

**Factorial (two-way interaction).** List `[factor_a, factor_b]` pairs under
`interactions` (each member the condition column or a declared covariate). The fit
then tests the **interaction** — a difference-of-differences F-test, "is the
condition effect modified by this factor?" — instead of the case-vs-control main
effect. An empty factorial-grid cell (e.g. no case sample in one batch) leaves the
interaction inestimable and halts before the fit. The result artifact is labelled
as an interaction test, so it is never misread as a plain contrast.

```yaml
differential_expression:
  covariates: [batch]              # the interacting factor must be a covariate too
  interactions:
    - [condition, batch]           # tests condition x batch (difference of differences)
```

A single-sample dataset (one condition, or no donor replication) has no valid
comparative statistics: the descriptive spine (QC, embedding, clustering,
annotation) runs, but DE/DA halt with an explicit error rather than reporting
confident-but-meaningless p-values.

## Generating configs from a manifest

To expand a hypothesis manifest into many per-`(hypothesis, cell_type)` configs, use
the `gen-configs` command:

```bash
gen-configs run --manifest hypotheses.yaml --template template.yaml --out-dir generated/
```

It writes one `generated/configs/<key>.yaml` per combination plus an
`accounting.json` recording what was produced.
