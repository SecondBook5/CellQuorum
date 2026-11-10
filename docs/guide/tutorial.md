# Your first study

Start with a human or mouse count-bearing AnnData file, Cell Ranger output directory, standalone 10x H5 or Matrix Market matrix, or a CSV/TSV sample sheet. For gene-family QC, use gene symbols in `var_names`. Install the base package on
Linux, macOS or WSL. [Installation guide](https://github.com/SecondBook5/cellquorum#installation).

## 1. Set up the study

```bash
cellquorum init my_study --input /path/to/study.h5ad
```

For several libraries, use one row per library. Paths are relative to the sheet. Use a biological replicate ID consistently across repeated measurements.

```csv
sample_id,path,donor,condition,metadata_path
D1_normal,D1_normal/outs,D1,normal,D1_normal/cells.csv
D1_tumor,D1_tumor/outs,D1,tumor,D1_tumor/cells.csv
```

`metadata_path` is optional. Its table has a `barcode` column and one row for every barcode in the filtered library. Add your cell labels there. Additional sheet fields are library-level metadata. For AnnData entries with transformed `X`, declare `counts_layer`. Recognized sidecars include `raw_matrix_path`, `loom_path`, `bam_path` and `gtf_path`. A matrix-only study can run expression analyses and DPT as its design allows. Velocity needs splice evidence. For a sample sheet, run `cellquorum init my_study --input samples.csv`. It writes the reusable configuration, merged counts and normalized manifest into the new study directory.

Choose the question you want to answer.

| Choice | What to supply | What to open first |
|---|---|---|
| `discovery` | Counts plus the study design, label sources, reference resources and biological questions the available evidence supports | Manuscript gallery, exclusions and study report |
| `questions` | Selected question families and their inputs. Replicated comparisons can also request leave-one-donor-out abundance sensitivity | Panels, donor tables, resolved settings and inferred program results |
| `explore` | Raw counts, organism and preparation | The cluster UMAP and QC figures in the HTML report |
| `abundance` | The same inputs plus existing cell-type labels, biological replicate IDs and two conditions | The composition figure and `results/da_proportion_ttest.csv` |

The prompts list the count matrices and observation columns in your file. Specify whole cells,
nuclei or mixed preparation so QC can interpret nuclear-retained signal appropriately. Give
a library ID column if you have one. For a mixed study with several values identifying
nuclei, add them to `qc_evidence.graded.nuclei_values` in the saved config.

For abundance, choose the biological replicate, such as a mouse or a patient. In a study measuring
the same people before and after treatment, choose the patient ID, set the reference to the
baseline condition, and choose `paired`. The setup shows the number of replicates in each arm
and how many appear in both. A paired test requires at least two complete pairs and uses the
donors with both measurements. For groups of different individuals choose `independent`.

## 2. Review the saved configuration

Open `my_study/config.yaml`. The input and output paths are absolute, so the same config works
from another directory on the same machine. The `stages` block lists the analyses selected.
The baseline QC applies floors of 200 detected genes per cell and three cells per gene.
Review those thresholds for the assay. Ambient correction and doublet detection are off in
both starters.

For abundance, check `design.donor_col`, `design.condition_col`, `design.control`, `design.case`
and `design.paired`. An effect of +5 percentage points means that the cell type accounts for
five more percentage points of the sampled cells, on average, in the comparison condition.
Check the donor counts and adjusted p-values alongside the effect. The test measures relative
composition.

## 3. Preview the plan

`plan` validates the YAML and lists the stages enabled in your configuration
alongside the available backends.

```bash
cellquorum plan -c my_study/config.yaml
```

Dataset and method checks run during execution. Inspect the report for their
outcomes. Add `--json` to the plan command for machine-readable output.

## 4. Run it

```bash
cellquorum run -c my_study/config.yaml
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
my_study/run/
├── reports/       # human-readable HTML + Markdown report — start here
├── figures/       # generated figures (QC, embeddings, DE, …)
├── results/       # machine-readable result tables (CSV/TSV)
├── objects/       # intermediate AnnData objects (.h5ad checkpoints)
├── provenance/    # resolved config, execution plan, backend status,
│                  #   environment/version stamp, artifact manifest
└── logs/          # execution logs and per-stage warnings
```

Open `reports/report.html` and expand an analysis under **Analysis outputs** to
open its registered figures and tables. Keep the run directory together when sharing it
so the links continue to work. The `provenance/` directory
captures the exact resolved config, the plan that ran, which backends were
present, and the software/environment versions — so a run can be re-created or
audited later.

## Try a public paired study

Download the [Kang AnnData file](https://scverse-exampledata.s3.eu-west-1.amazonaws.com/pertpy/kang_2018.h5ad)
and run `cellquorum init kang --input kang_2018.h5ad`. Select `abundance`, `X`, `human`
and `cells`. Leave the library field empty. Choose `cell_type` for the annotation,
`replicate` for the donor and `label` for the condition. Select `ctrl` as the reference,
`stim` as the comparison and `paired` for the design. [Dataset description](https://pertpy.scverse.org/en/latest/api/data/pertpy.data.kang_2018.html).

Run `cellquorum run -c kang/config.yaml`. The tested file contained 24,673 cells from
eight donors. With the starter settings, 24,562 cells cleared the QC floors and all eight
cell types had results from eight paired donors. This file contains no mitochondrial
gene names matching `MT-`, so mitochondrial percentage is unavailable.

## 6. Extend the analysis

Edit the saved configuration to extend a study.

- **Rename the cohort keys** to match your obs columns — nothing downstream
  hardcodes them.
- **Enable more stages** in the `stages` block as their backends become
  available (differential expression and multicellular programs need the R
  env; cell–cell communication and trajectory benefit from the GPU env — see
  [Backends & environments](backends.md)).
- **Restrict to a lineage** with `input.subset` (e.g. run the whole pipeline on
  fibroblasts only) without splitting your object by hand.
- **Drop a cluster** with `input.exclude` (e.g. an ambient-RNA artifact cluster
  found by `cellquorum.stats.cluster_artifact_audit`) — the direction a subset
  cannot express, since a subset names what to keep. Both rules compose, and
  both record what they removed in provenance. See
  [Configuration](configuration.md#the-input-section).

## Next steps

- [Configuration](configuration.md) — every top-level section and per-stage
  block, and how a method is selected per stage.
- [Architecture](../reference/architecture.md) — how validation, planning, the executor, and
  provenance fit together.
- [Backends & environments](backends.md) — the R/Rscript bridge, the GPU env,
  and the isolated backend environments.
- [Python API](../reference/api.md) — drive the same pipeline from a notebook with
  `cellquorum.run_pipeline`.
