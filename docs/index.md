# CellQuorum documentation

CellQuorum is a publication-grade, config-driven single-cell RNA-seq workflow
engine. You describe a dataset and its analysis in one validated YAML file and run
it with one command; the engine plans, validates, and executes the workflow across
Python, R/Bioconductor, and GPU backends, and records machine-readable provenance
for every step.

Start with the [README](https://github.com/SecondBook5/cellquorum#readme) for an
overview and installation, then work through the [Tutorial](guide/tutorial.md) for a
hands-on first run. The guides below go deeper.

## Get started

- **[Tutorial](guide/tutorial.md)**: a hands-on first run, from a raw `.h5ad` to a
  provenance-tracked run. Point the config at your data, preview the plan, run it, and read every
  output.
- To see it work without any data of your own, run `cellquorum demo my_demo` (see the
  [README](https://github.com/SecondBook5/cellquorum#see-it-work-in-two-minutes)).

## User guide

| Guide | Contents |
|---|---|
| [Configuration](guide/configuration.md) | The full configuration reference: top-level sections, per-stage blocks, stage enable flags, and how a method is selected per stage. |
| [Backends & environments](guide/backends.md) | The primary and isolated environments, runtime backend detection, and the R/Rscript bridge. |
| [Docker](guide/docker.md) | Building and running the layered container image. |
| [Snakemake](guide/snakemake.md) | Orchestrating runs across many hypotheses and cell types with Snakemake. |

## Reference

| Page | Contents |
|---|---|
| [Python API](reference/api.md) | The `run_pipeline` entry point and the `tl` / `pp` / `diag` / `evidence` notebook namespaces, generated from the source docstrings. |
| [Architecture](reference/architecture.md) | The execution model: validation, planning, the registry-driven executor, data contracts, method dispatch, compute routing, and provenance. |
| [How it works](reference/how-it-works.md) | A file-by-file walkthrough of one run. |
| [Figure gallery](gallery/index.md) | Every figure CellQuorum draws, from synthetic data, pinned by pixel-exact tests. |
| [Third-party software](reference/third-party-software.md) | The software CellQuorum calls: licenses and citations. |

## Project

| Page | Contents |
|---|---|
| [Roadmap](ROADMAP.md) | Current capabilities and the engineering and scientific backlog. |
| [Changelog](changelog.md) | Release notes. |
| [Developer smoke tests](dev_smoke.md) | Fast local checks. |
| [Deployment requirements](design/deployment-goal.md) and [readiness assessment](design/production-readiness-audit.md) | How the package is validated for real deployments. |
| [GSE140819 benchmark](benchmarks/gse140819.md) | The real-data benchmark. |

Build this site locally with `make docs-serve` (live reload) or `make docs`
(strict build, as CI runs it).

## Key facts

- **40 registered stages** and more than 70 config-selectable methods, organized into a
  best-practices pipeline (see the workflow diagram in the README).
- **Two front doors:** the `cellquorum` / `cq` CLI and the `cellquorum.run_pipeline`
  Python API.
- **Fail-loud contracts:** every stage boundary validates its `AnnData`; a method
  with missing inputs or an unavailable backend skips with a recorded reason.
- **Reproducible:** standardized run directory + provenance (resolved config, plan,
  backend status, environment/version stamp, artifact manifest).

See the [Changelog](changelog.md) for release notes and
[`CITATION.cff`](https://github.com/SecondBook5/cellquorum/blob/main/CITATION.cff)
for citation metadata. CellQuorum is distributed under the BSD 3-Clause license.
