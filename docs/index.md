# CellQuorum documentation

CellQuorum is a publication-grade, config-driven single-cell RNA-seq workflow
engine. You describe a dataset and its analysis in one validated YAML file and run
it with one command; the engine plans, validates, and executes the workflow across
Python, R/Bioconductor, and GPU backends, and records machine-readable provenance
for every step.

Start with the [README](../README.md) for an overview, installation, and a
quickstart. The guides below go deeper.

## Guides

| Guide | Contents |
|---|---|
| [Architecture](architecture.md) | The execution model — validation, planning, the registry-driven executor, data contracts, method dispatch, compute routing, and provenance; the 19-package layout. |
| [Configuration](configuration.md) | The full configuration reference — top-level sections, per-stage blocks, stage enable flags, and how a method is selected per stage. |
| [Backends & environments](backends.md) | The primary and isolated environments, runtime backend detection, and the R/Rscript bridge. |
| [Docker](docker.md) | Building and running the layered container image. |
| [Snakemake](snakemake.md) | Orchestrating runs with Snakemake. |
| [Developer smoke tests](dev_smoke.md) | Fast local checks. |
| [Roadmap](ROADMAP.md) | Current capabilities and the engineering/scientific backlog. |
| [Scientific & engineering plan](SCIENTIFIC_ENGINEERING_PLAN.md) | The full design and scientific plan. |

## Key facts

- **30 registered stages**, ~60 config-selectable methods, organized into a
  best-practices pipeline (see the workflow diagram in the README).
- **20 top-level packages** under `src/cellquorum/`.
- **Two front doors:** the `cellquorum` / `cq` CLI and the `cellquorum.run_pipeline`
  Python API.
- **Fail-loud contracts:** every stage boundary validates its `AnnData`; a method
  with missing inputs or an unavailable backend skips with a recorded reason.
- **Reproducible:** standardized run directory + provenance (resolved config, plan,
  backend status, environment/version stamp, artifact manifest).

See [`CHANGELOG.md`](../CHANGELOG.md) for release notes and [`CITATION.cff`](../CITATION.cff)
for citation metadata. CellQuorum is distributed under the BSD 3-Clause license.
