# CellQuorum documentation

CellQuorum is a publication-grade, config-driven single-cell RNA-seq workflow
engine. You describe a dataset and its analysis in one validated YAML file and run
it with one command; the engine plans, validates, and executes the workflow across
Python, R/Bioconductor, and GPU backends, and records machine-readable provenance
for every step.

Start with the [README](https://github.com/SecondBook5/cellquorum#readme) for an
overview, installation, and a quickstart. The guides below go deeper.

## Guides

| Guide | Contents |
|---|---|
| [Architecture](architecture.md) | The execution model — validation, planning, the registry-driven executor, data contracts, method dispatch, compute routing, and provenance; the 17-package layout (plus compatibility shims). |
| [Configuration](configuration.md) | The full configuration reference — top-level sections, per-stage blocks, stage enable flags, and how a method is selected per stage. |
| [Backends & environments](backends.md) | The primary and isolated environments, runtime backend detection, and the R/Rscript bridge. |
| [Python API](api.md) | The `run_pipeline` entry point and the `tl` / `pp` / `diag` / `evidence` notebook namespaces, generated from the source docstrings. |
| [Docker](docker.md) | Building and running the layered container image. |
| [Snakemake](snakemake.md) | Orchestrating runs with Snakemake. |
| [Developer smoke tests](dev_smoke.md) | Fast local checks. |
| [Roadmap](ROADMAP.md) | Current capabilities and the engineering/scientific backlog. |
| [Changelog](changelog.md) | Release notes. |

Build this site locally with `make docs-serve` (live reload) or `make docs`
(strict build, as CI runs it).

## Key facts

- **30 registered stages**, ~60 config-selectable methods, organized into a
  best-practices pipeline (see the workflow diagram in the README).
- **17 top-level packages** under `src/cellquorum/` (plus four compatibility-shim
  packages preserving pre-#187 import paths).
- **Two front doors:** the `cellquorum` / `cq` CLI and the `cellquorum.run_pipeline`
  Python API.
- **Fail-loud contracts:** every stage boundary validates its `AnnData`; a method
  with missing inputs or an unavailable backend skips with a recorded reason.
- **Reproducible:** standardized run directory + provenance (resolved config, plan,
  backend status, environment/version stamp, artifact manifest).

See the [Changelog](changelog.md) for release notes and
[`CITATION.cff`](https://github.com/SecondBook5/cellquorum/blob/main/CITATION.cff)
for citation metadata. CellQuorum is distributed under the BSD 3-Clause license.
