# Changelog

All notable changes to CellQuorum are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
CellQuorum is pre-1.0 and under active development: minor versions may change the
public API and the configuration schema.

## [Unreleased]

### Added

- **Public reusable-utility surface `cellquorum.utils` (consolidation Move 5).**
  The analytical building blocks that analysis scripts already reach for are now
  a first-class, versioned public surface instead of deep-internal reach-in:
  `cq.utils.de_table_to_ranking` (DE table → preranked GSEA vector),
  `cq.utils.get_net` (long-format prior-knowledge net via decoupler/OmniPath), and
  `cq.utils.aggregate_pseudobulk` (cells → donor × condition pseudobulk), plus the
  companion types `PseudobulkResult` and `PriorFetchError`. These are **re-exports
  of the canonical `cellquorum.comparative` implementations, not copies** — a fix
  to the engine is a fix here — and the pre-consolidation deep-import paths still
  resolve to the same objects, so existing scripts keep working unchanged.
  Importing `cellquorum.utils` pulls in no heavy optional dependency (`get_net`
  lazy-imports `decoupler` only when called), preserving the skip-not-crash
  invariant. The surface is frozen by `test_public_api_contract` and
  `test_public_utils_surface`, and documented in `docs/api.md`.
- **Tensor-cell2cell decomposition cost guardrail (#140).** The non-negative
  CP factorization cost scales with ``runs x prod(tensor.shape)``, and the
  sender/receiver axes are the cell-type group count — so a fine-grained
  (many-subcluster) tensor at the ``robust`` default (100 runs) could silently
  run for many hours. The stage now always records the tensor shape and run
  count (in the stage notes and metrics) so the cost is visible rather than
  silent, and adds two opt-in knobs: ``tf_optimization: auto`` scales the run
  count down to fit a new ``max_decomposition_cost`` budget (the ``runs x
  tensor-elements`` proxy), never below one run; an explicit ``robust`` /
  ``regular`` run that would exceed a set budget is *honored* but logs a loud
  over-budget warning pointing at the ``auto`` escape hatch, a coarser
  resolution, or GPU. With no budget set (the default) behavior is unchanged,
  and ``tf_optimization`` is now a validated enum (``robust`` / ``regular`` /
  ``auto``) that rejects typos at config-parse time.
- **Factorial / interaction differential expression (#192).** The pseudobulk
  edgeR DE stage now supports two-way *interaction* testing. A new
  `differential_expression.interactions` config field takes `[factor_a, factor_b]`
  pairs (each member the condition column or a declared covariate); when set, the
  fit tests the interaction — a difference-of-differences quasi-likelihood F-test
  over the interaction coefficients ("is the condition effect modified by this
  factor?") — instead of the case-vs-control main effect, and the result artifact
  is labelled accordingly so an interaction table is never misread as a contrast.
  This is backed by a general multi-factor **design-estimability layer** in
  `cellquorum.config.design` (`build_design_matrix` / `analyze_design` /
  `validate_design_matrix`): it treatment-codes an arbitrary set of factors plus
  interactions exactly as R's `model.matrix` does, then halts loudly on a
  rank-deficient design — a covariate perfectly aliased with the tested condition,
  two confounded (nested) factors, or an empty factorial-grid cell that leaves an
  interaction inestimable. The DE stage runs this gate before the fit, so a
  confounded covariate or an empty crossed cell fails with a clear configuration
  error rather than reaching edgeR and producing a meaningless coefficient.
- **Cell-state program scoring (`state_scoring`, #190).** A new analysis stage
  that scores curated cell-state programs (stress/HSP, hypoxia/HIF, interferon,
  senescence/SASP, fibrosis/ECM) on the annotated object. It runs two scorers by
  default — scanpy `score_genes` (per-program scores into `obs`) and decoupler
  AUCell (into `obsm`) — and resolves programs from the built-in curated set,
  user-supplied programs, the markers config, and/or a `.gmt` file.
- **De-novo program discovery (`discovery`, #191).** A new consensus-NMF stage
  that discovers data-driven expression programs without prior curation: it runs
  scikit-learn NMF at rank *k* across `n_runs` seeds, consensus-clusters the
  replicate gene spectra (the cNMF idea, in-process), and projects every cell
  onto the consensus spectra for a non-negative usage matrix (`obsm["X_cnmf"]`),
  writing per-program top-gene loadings and per-cell-type mean usage. Together
  these fill the two remaining phase-4 (state) slots; three planner slots
  (`integration_gate`, `composition`, `molecular_inference`) remain reserved.

### Changed

- **Consolidation round 2 (#187).** Continued the source-tree consolidation
  begun in #167. Stage registration is now single-sited on each stage class
  through a `@register_stage` decorator; a shared `StageArtifactWriter`
  centralizes run-directory artifact writing; SoupX ambient-RNA correction
  moved into its own top-level ``ambient_correction`` package (it runs before
  QC, so it is no longer nested under ``qc``); and the QC figure builders moved
  under ``visualization.qc``. The four comparative analyses — differential expression, differential abundance,
  enrichment, and multicellular programs — are now submodules of a single
  ``comparative`` package (their former top-level packages remain as thin
  compatibility shims). The user-facing Python surface — the
  :func:`run_pipeline` entry point and the notebook namespaces ``tl`` / ``pp``
  / ``diag`` / ``evidence`` plus the ``_notebook`` adapter — now lives in a
  single ``cellquorum.api`` package (formerly the ``cellquorum.api`` *module*
  and five scattered top-level modules); the top-level package re-exports it so
  ``cq.run_pipeline``, ``cq.tl`` … stay canonical, and the old module paths
  (``cellquorum.tl``, ``cellquorum.pp``, ``cellquorum.diag``,
  ``cellquorum.evidence``, ``cellquorum._notebook``) remain as thin re-export
  shims. The tree now has **20** top-level packages plus four
  compatibility-shim packages. The QC configuration module
  (``cellquorum.qc.config``) is now config-only: the dozen near-identical
  Pydantic field-coercion validators it carried were extracted into a new
  standard-library-only leaf module ``cellquorum.qc.config_validators`` (six
  parametrized ``coerce_*`` helpers), so each coercion pattern lives once and is
  directly unit-tested; the config models delegate to them and stay declarative.
  The seam is a new leaf module rather than a fold into ``qc.thresholds`` because
  ``qc.thresholds`` already imports ``qc.config`` — the validators' error
  messages and coerced values are byte-identical to the per-model versions they
  replace. Like
  #167 this is a pure legibility refactor — behavior, the CLI, the configuration
  schema, the public Python API, and all analysis outputs are unchanged, and the
  pre-move import paths keep working through thin re-export shims.
- **Package consolidation (#167).** Reorganized the source tree from ~40
  top-level packages into **18** cohesive packages (**19** after adding the
  ``multicellular_programs`` stage; later **20** engine packages, see #187 above), and
  introduced a single
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

- **`make lock` targeted impossible platforms (#189).** The conda-lock recipe
  ran `conda-lock lock` with no `--platform`, so it attempted a multi-platform
  solve (osx-64/win-64 included). The GPU env carries linux-only CUDA packages
  (`pytorch-cuda`, the `nvidia` channel) with no osx/win build, so the solve
  aborted before any lock file was written. The target now pins `-p linux-64` —
  the image's only build platform and the only one we ship — matching the docs
  in `envs/README.md` and `docs/backends.md`.

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
- **Hypothesis bundling survived neither a failed pair nor the #187 move (#161).**
  Two defects in the Snakemake hypothesis-bundling path: (1) the `bundle_hypothesis`
  and `aggregate_status` rules imported `cellquorum.workflow.*`, but the modules
  moved to `cellquorum.cli.workflow.*` in #187 — so both rules crashed with
  `ModuleNotFoundError` on first real use; the rule imports now point at the
  canonical location. (2) `assemble_bundle` silently degraded a crashed or
  never-run cell type to an empty "no artifacts" section indistinguishable from a
  successful run with no figures. It now classifies each pair as
  `completed`/`failed`/`missing` (completion signal = `provenance/artifact_manifest.csv`,
  the same marker the status matrix uses), flags failures loudly in the HTML report
  with a one-glance "N of M completed" summary, and writes a machine-readable
  `bundle_status.json` alongside `report.html` so a caller or CI can act on partial
  failure without scraping HTML.

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
- **DIALOGUE provisioned reproducibly (#189):** the `multicellular_programs`
  stage depends on `livnatje/DIALOGUE`, a GitHub-only R package with no
  conda/CRAN release, so it silently skipped in every image. Its CRAN `Depends:`
  are now solved from conda-forge in `cellquorum-r.yml` (ABI-matched to
  `r-base`); the one dependency absent from conda-forge (`unikn`) is
  source-installed from a date-pinned CRAN snapshot, and DIALOGUE itself is
  installed at a pinned commit (`dependencies=FALSE`). A build-time
  `library(DIALOGUE)` check fails the image loudly if the package cannot load,
  rather than letting the stage skip at runtime. `envs/README.md` documents the
  same pins for a local `cellquorum-r` env.

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
