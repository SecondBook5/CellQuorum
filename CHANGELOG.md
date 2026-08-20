# Changelog

All notable changes to CellQuorum are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
CellQuorum is pre-1.0 and under active development: minor versions may change the
public API and the configuration schema.

## [Unreleased]

### Changed

- **Package consolidation (#167).** Reorganized the source tree from ~40
  top-level packages into **18** cohesive packages, and introduced a single
  shared R-method abstraction (`cellquorum.methods.r_method.RAnalysisMethod`)
  used by the edgeR, Milo, propeller, NicheNet, MultiNicheNet, DIALOGUE, and
  scDiagnostics adapters. Each `*_viz` module now lives inside its parent stage
  package. This
  is a pure legibility refactor: behavior, the CLI, the configuration schema, the
  public Python API, and all analysis outputs are unchanged.
- **Version single-sourcing (#156).** The package version is now defined only in
  `cellquorum.version.__version__`; `pyproject.toml` resolves it dynamically, so
  there is one canonical version string.
- **License metadata reconciled.** The declared license is **BSD-3-Clause**
  across `LICENSE`, `pyproject.toml`, `README.md`, and `CITATION.cff` (previously
  `pyproject`/`README` disagreed with the `LICENSE` file).
- **Documentation rewrite (#156).** The `README` was rewritten with a rendered
  workflow diagram, and the `docs/` guides (index, architecture, configuration,
  backends) were filled in.

### Fixed

Correctness defects that could silently produce wrong output (#168):

- **QC no-drop mode (#168b / #181).** The QC "report only" behavior silently kept
  all cells. The mode is now named `flag_no_drop`, annotates cells instead of
  dropping them, and the legacy `report_only` value is **loudly rejected** with a
  clear configuration error rather than silently aliased.
- **Doublet threshold never fired (#168a / #180).** The doublet-score cutoff was
  computed but never applied; flagged doublets are now removed as configured.
- **`dimensionality` auto-`n_pcs` (#168c / #182).** `n_pcs: auto` now logs the
  selected dimensionality and warns when the knee heuristic under-selects, so an
  under-powered embedding is visible rather than silent.
- **`r.rscript_path` now honored (#183).** The documented `r.rscript_path`
  configuration field was dead — never threaded to the Rscript backend, and R
  methods gated availability on a hardcoded bare `Rscript` on `PATH`. It is now
  threaded into the default backend registry, and R methods check the backend's
  *configured* path, so an R install outside the default `PATH` (as in the
  layered container/HPC image) is reachable instead of silently skipped.
- **Reserved R preference fields documented (#183).** `r.preferred_backend` and
  `r.fallback_to_rscript` express an rpy2-vs-Rscript preference that is not yet
  wired — every R-backed method dispatches through the Rscript backend. The
  `RConfig` docstrings and `docs/configuration.md` now state plainly that these
  fields are reserved and currently no-ops, rather than silently ignoring them.
  Wiring rpy2 dual-dispatch is tracked as future work.

### Added

Recent engineering and reproducibility hardening:

- **`multicellular_programs` stage:** DIALOGUE-based inference of cross-cell-type
  coordinated programs, with donor-support and program-stability diagnostics.
- **CI (#155):** ruff, the fast pytest tier, a build check, and a Docker smoke
  test, plus pytest markers (`gpu`, `slow`, `r`, `integration`) and coverage.
- **Provenance bundling (#154):** run-level environment and version stamps are
  written alongside each run.
- **Repro-honest status reporting (#153):** the status matrix survives stage
  failures and distinguishes *failed* from *never-ran* from *skipped*.
- **Packaged R scripts (#150):** bundled `*.R` backend scripts ship with the
  wheel; `LICENSE` filled.
- **Docker entrypoint fix (#151):** the container runs the CLI inside the correct
  environment so the CCC, trajectory, and GPU stages are reachable in-image.
- **Environment locking:** `make lock` generates `conda-lock` files from the
  `envs/*.yml` recipes (see `envs/README.md`).

## [0.1.0] — initial development version

The execution spine and config-driven analysis backbone.

- Strict YAML/Pydantic configuration with fail-loud data contracts (structural,
  layer-provenance, and statistical validation at every stage boundary).
- Backend registry (Python / R / Rscript / GPU / RAPIDS availability) and an
  execution planner that reports enabled, skipped, and unimplemented stages.
- Registry-driven executor with **30 stages** covering QC, preprocessing,
  dimensionality reduction, integration, clustering, annotation, embeddings,
  differential expression, differential abundance, enrichment, co-expression,
  gene-regulatory networks, in-silico perturbation, cell-cell communication, and
  RNA-velocity trajectory analysis.
- Standardized run directory layout and machine-readable provenance artifacts.
- Public Python API (`cellquorum.run_pipeline`) and CLI entry points
  (`cellquorum`, `cq`, `gen-configs`).
- GPU acceleration via `rapids-singlecell` / `cupy` with automatic CPU fallback,
  gated on real device capability.

[Unreleased]: https://github.com/SecondBook5/cellquorum/compare/main...HEAD
