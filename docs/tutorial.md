# Tutorial: your first run

This walkthrough takes you from a raw `.h5ad` to a provenance-tracked CellQuorum
run and its figures, tables, and report — using the shipped, dataset-agnostic
example config so nothing here is specific to any one study. By the end you will
know how to point the engine at your own data, preview what it will do, run it,
and read every output it writes.

If you have not installed CellQuorum yet, see the
[README](https://github.com/SecondBook5/cellquorum#readme) for installation and
the [Backends & environments](backends.md) guide for the optional R/GPU
backends. Confirm the CLI is on your path first:

```bash
cellquorum --version
```

## 1. Start from the example config

CellQuorum runs are driven entirely by one validated YAML file — no code
changes per dataset. Copy the shipped generic example as your starting point:

```bash
cp configs/generic_pbmc_example.yaml configs/my_run.yaml
```

Three blocks do the important work. The **`cohort`** block declares your
structural obs keys *once*, and every stage resolves from it — so you never
repeat "which column is the sample?" per stage:

```yaml
cohort:
  sample_key: sample_id
  donor_key: donor_id
  condition_key: stim          # e.g. "control" vs "stimulated"
  batch_key: batch
  condition_levels: [control, stimulated]
```

The **`design`** block names the primary comparison for the differential
stages:

```yaml
design:
  donor_col: donor_id
  condition_col: stim
  case: stimulated
  control: control
  paired: false                # set true for matched within-donor designs
```

The **`stages`** block is the on/off switch per analysis step. The example
keeps the portable, CPU-only stages on and leaves the R/GPU-dependent ones off
so it runs anywhere:

```yaml
stages:
  qc: true
  preprocessing: true
  dimensionality: true
  integration: true            # Harmony over cohort.batch_key
  clustering: true
  population_identity: true
  differential_expression: false   # turn on once your R backend is available
  cell_cell_communication: false
```

## 2. Point it at your data

Set the input path and tell the engine which layer holds raw counts:

```yaml
input:
  h5ad: /path/to/your_data.h5ad
  counts_layer: counts
```

Your `.h5ad` needs the obs columns your `cohort` block names — in the example,
`sample_id`, `donor_id`, `stim`, and `batch` — and a `counts` layer with raw
(un-normalized) counts. The engine validates this at load time and fails loud
with the exact missing key rather than producing wrong results silently.

!!! tip "No column to group by yet?"
    Every key is resolved defensively: if a declared grouping column is absent
    from a given object, the stage falls back gracefully (for example, QC
    figures collapse to a single ungrouped distribution) instead of crashing.

## 3. Preview the plan

Before running anything heavy, ask the engine what it *will* do. `plan`
validates the config, checks which backends are actually available, and prints
the resolved stage list — marking each stage as planned, skipped (backend
unavailable), or disabled:

```bash
cellquorum plan -c configs/my_run.yaml
```

This is the fastest way to catch a config mistake or a missing backend: if a
stage you wanted shows up as *skipped*, the plan tells you which backend it
needs before you spend compute. Add `--json` for machine-readable output.

## 4. Run it

```bash
cellquorum run -c configs/my_run.yaml -o runs/my_run
```

The engine creates the standardized run directory, executes each enabled stage
in dependency order, validates the `AnnData` at every stage boundary, and
records provenance as it goes. Useful flags:

| Flag | Effect |
|---|---|
| `-o, --output-dir` | Where the run directory is written. |
| `--bootstrap-only` | Create the run structure and resolve the plan without executing stages. |
| `--json` | Print the run summary as JSON. |
| `-q, --quiet` | Suppress progress output. |

## 5. Read the outputs

Every run writes the same directory contract, so you always know where to look:

```
runs/my_run/
├── reports/       # human-readable HTML + Markdown report — start here
├── figures/       # generated figures (QC, embeddings, DE, …)
├── results/       # machine-readable result tables (CSV/TSV)
├── objects/       # intermediate AnnData objects (.h5ad checkpoints)
├── provenance/    # resolved config, execution plan, backend status,
│                  #   environment/version stamp, artifact manifest
└── logs/          # execution logs and per-stage warnings
```

Open `reports/` first for the narrative summary, then drill into `figures/` and
`results/`. `provenance/` is what makes a run reproducible and citable: it
captures the exact resolved config, the plan that ran, which backends were
present, and the software/environment versions — so a run can be re-created or
audited later.

## 6. Adapt to your own study

To take this to a real dataset, you typically only touch the config:

- **Rename the cohort keys** to match your obs columns — nothing downstream
  hardcodes them.
- **Enable more stages** in the `stages` block as their backends become
  available (differential expression and multicellular programs need the R
  env; cell–cell communication and trajectory benefit from the GPU env — see
  [Backends & environments](backends.md)).
- **Restrict to a lineage** with `input.subset` (e.g. run the whole pipeline on
  fibroblasts only) without splitting your object by hand.

## Next steps

- [Configuration](configuration.md) — every top-level section and per-stage
  block, and how a method is selected per stage.
- [Architecture](architecture.md) — how validation, planning, the executor, and
  provenance fit together.
- [Backends & environments](backends.md) — the R/Rscript bridge, the GPU env,
  and the isolated backend environments.
- [Python API](api.md) — drive the same pipeline from a notebook with
  `cellquorum.run_pipeline`.
