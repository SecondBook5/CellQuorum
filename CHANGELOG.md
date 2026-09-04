# Changelog

All notable changes to CellQuorum are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
CellQuorum is pre-1.0 and under active development: minor versions may change the
public API and the configuration schema.

## [Unreleased]

### Added

- **`tests/test_no_hardcoded_machine_paths.py`** keeps machine-specific paths from
  coming back (see *Changed* below for the removal itself). It scans
  every git-tracked file under `configs/`, `src/`, `tests/`, and `scripts/` for absolute
  paths rooted at `/mnt/`, `/home/`, `/Users/`, `/media/`, `/Volumes/`, or a Windows drive
  letter and reports them all at once; a `machine-path-ok` line comment opts out. It also
  asserts `configs/config.yaml` stays portable, and walks every shipped config to check it
  still validates — so a field renamed in `config/models.py` fails there rather than in a
  user's first run. Writing it immediately surfaced four hardcoded paths missed by hand
  (a second `cellranger_root` in two configs, and two external atlas paths).
- **`tests/_external_data.py`** — the external-dataset opt-in (`require_external_dir`,
  `require_external_file`, `require_cellranger_library`, `stub_config_env`) plus a cached
  `r_package_available`. Eighteen test modules each spawn their own `Rscript` probe and
  several do it at module scope, where the cost lands on collection for every session
  regardless of what was selected; the cached helper is now available to replace them.
  Only the two SoupX modules are converted so far.
- **The `integration`, `r`, and `slow` markers are now actually applied** to the
  real-data tests, so `pytest -m "not integration"` deselects them even on a machine that
  *does* have the data — which `skipif` alone cannot do. The markers were declared in
  `pyproject.toml` but used by nothing.
- **`tests/test_backend_probe_cache.py`** covers the new probe cache: the probe runs
  once per distinct question, distinct questions never share an answer (a narrowed
  cache key would misreport one environment's availability as another's), a missing
  launcher or timeout still reads as "unavailable" rather than raising, and module-name
  validation still happens in the backend method ahead of the cache.
- **`tests/test_import_cost.py` pins the lazy-import invariant.** Asserts that a bare
  `import cellquorum` pulls in none of torch, scvi, lightning, celltypist, or
  decoupler, stays under a module ceiling, that importing the CLI does not pull in
  the engine, and that every lazily bound public name still resolves. The invariant
  was previously documented in prose and silently false.
- **PEP 561 `py.typed` marker.** The package ships full annotations (ruff's `ANN`
  rules are enabled) but omitted the marker, so type checkers in downstream projects
  treated every `cellquorum` import as `Any`. CI now asserts the built wheel contains
  it, since an editable install cannot catch a missing package-data entry.
- **mypy configuration and a CI typecheck gate on the engine spine.** mypy was
  declared in the `[dev]` extra but never configured or run. A baseline found 436
  errors across 102 modules; the gate covers `core`, `config`, `io`, `methods`, and
  `cli`, which are green. The `[[tool.mypy.overrides]]` `ignore_errors` list is a
  backlog of 14 spine modules, not a policy — see `CONTRIBUTING.md`.
- **Coverage measurement.** `pytest-cov` and `pytest-xdist` are now dev
  dependencies with `[tool.coverage]` configuration; the project previously had no
  coverage number at all. Subprocess-invoked backend scripts are omitted because the
  in-process tracer cannot see them and would report a false 0%.
- **`CONTRIBUTING.md`, `SECURITY.md`, issue/PR templates, and `dependabot.yml`.**
- **`.github/workflows/release.yml`** — tag-triggered release via PyPI Trusted
  Publishing, gated on a protected environment. Verifies the tag matches
  `version.py`, requires a CHANGELOG heading, re-runs the wheel content assertions,
  runs `twine check --strict`, and smoke-tests the wheel in a clean venv.
- **Makefile development targets**: `check`, `lint`, `format`, `typecheck`,
  `test-fast`, `test`, `test-cov`, plus a `help` default goal.
- **Public reusable-utility surface `cellquorum.utils` (consolidation Move 5).**
  The analytical building blocks that analysis scripts already reach for are now
  a first-class, versioned public surface instead of deep-internal reach-in:
  `cq.utils.de_table_to_ranking` (DE table → preranked GSEA vector),
  `cq.utils.get_net` (long-format prior-knowledge net via decoupler/OmniPath), and
  `cq.utils.aggregate_pseudobulk` (cells → donor × condition pseudobulk), plus the
  companion types `PseudobulkResult` and `PriorFetchError`. These are **re-exports
  of the canonical implementations in `cellquorum.stages.comparative`, not copies** —
  a fix to the engine is a fix here. (The pre-consolidation deep-import paths have
  since been removed in the API clean break — see *Removed* below — so
  `cellquorum.utils` is now the single supported import path for these helpers.)
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

- **A donor-consistency verdict on every differential-abundance call**
  (`cq.stats.paired_abundance_concordance`, `qualify_abundance_calls`). A compositional
  method reports a cohort mean, which cannot distinguish an effect every donor shares
  from one two large movers carry — and reviewers ask. Each called cell type now gets
  the per-donor direction count, an exact sign test, a leave-one-donor-out check, and a
  `consistent` / `heterogeneous` verdict written into the result table, with a run-summary
  note when a called effect is not donor-consistent. The `differential_abundance` stage
  annotates its own output with it.
- **A depth-confounding audit (`cq.stats.depth_confound_audit`,
  `depth_stratified_abundance`).** Library complexity confounds every continuous per-cell
  metric — pseudotime, stemness scores, module indices — and it is routinely correlated
  with condition. The audit tests all three legs (does depth track condition, does the
  metric track depth *within sample*, and does adjusting change the answer) rather than
  flagging correlation alone, which would manufacture alarm on depth-balanced cohorts. It
  adjusts on the within-sample slope, not the pooled one: with a depth-imbalanced cohort
  the pooled slope subtracts part of the biology. On a real cohort it graded two published
  pseudotime metrics `depth_driven` — one of them reversing sign under adjustment — and
  left a module score at 101% of its raw effect.
- **A compositional reference chosen on a scale-free criterion**
  (`select_compositional_reference`). scCODA's `reference_cell_type="automatic"` minimizes
  `var(p)/mean(p)`, which scales with the mean and so ranks cell types largely by how
  *rare* they are. The engine now picks the steadiest abundant cell type by centred-log-ratio
  variance, requires a real abundance floor rather than mere presence, and emits the
  criterion table with scCODA's own metric beside its own so the divergence is visible per
  dataset instead of taken on trust.
- **One canonical way to render a label (`cellquorum.core.labels.as_label_strings`).**
  Cluster labels arrive as ints, floats, categoricals or strings depending on which stage
  wrote them, and `"1"` vs `"1.0"` is enough to empty a join silently.

### Changed

- **No tracked file hardcodes a machine-specific path any more.** Five of the six shipped
  configs named the author's filesystem — including `configs/config.yaml`, which the CLI
  loads when no `--config` is given, so a new user's first `cellquorum run` tried to write
  to an external drive that did not exist on their machine. Three configs pointed
  `output_dir` at an absolute path inside one particular checkout, making them unusable
  from any other clone.
  - `configs/config.yaml` is now portable with no edits: `run_root: runs` (relative, and
    `runs/` is gitignored), everything else `null`. It loads and resolves an output
    directory from any working directory.
  - Study configs take their external inputs from the environment via
    `${oc.env:...}` — `CELLQUORUM_CELLRANGER_ROOT`, `CELLQUORUM_KC_H5AD`,
    `CELLQUORUM_AD_ATLAS_H5AD`, `CELLQUORUM_KC_ATLAS_H5AD`, `CELLQUORUM_SMOKE_H5AD` — and
    use repo-relative `output_dir`s. An unset variable fails at config-load time naming
    both the variable and the config key. **Existing invocations need those exports set;**
    see the table in `CONTRIBUTING.md`.
  - Integration tests resolve their data through `CELLQUORUM_TEST_CELLRANGER_ROOT` and
    `CELLQUORUM_TEST_KC_H5AD` and skip with a message naming the variable, instead of
    silently skipping forever on every machine but one.
  - The `/mnt/e/...` strings in `test_config.py` were purely cosmetic — that test never
    touches the filesystem — and are now obviously synthetic.
- **Dev tooling is pinned to exact versions; runtime dependencies remain floors.**
  `ruff>=0.6` meant CI pinned 0.8.6 while a fresh `pip install -e .[dev]` resolved to
  0.15.5, which reported 169 findings and wanted to reformat 71 files that CI
  considered clean — the two enforced different rules silently. CI now derives the
  ruff version from `pyproject.toml` and fails if `.pre-commit-config.yaml` disagrees,
  so the pin can only move in lockstep across all three declaration sites.
- **CI additions**: `concurrency` cancels superseded runs; the test job runs a fast
  tier first as a tripwire, then the full suite sharded with `-n auto` and coverage;
  the build job additionally asserts `py.typed` is bundled, runs `twine check
  --strict`, and installs the built wheel in a clean venv.
- `_restrict_plan_from_stage` in `core/pipeline.py` is annotated
  `PipelinePlan -> PipelinePlan` instead of `object -> object`. `PipelinePlan` was
  already imported at module scope, so the vague annotation bought nothing and
  disabled type checking on the resume path.
- `cli/app.py` gained an `if __name__ == "__main__"` guard;
  `python -m cellquorum.cli.app` previously imported and exited silently, which is
  indistinguishable from a CLI that ran and printed nothing.
- **API clean break + stage-package regroup (Adoptability A+B).** Established
  **one canonical import path per public thing** and grouped the pipeline-step
  packages under a single namespace, so a newcomer can navigate the tree without
  the author. The 12 pipeline-step packages (`ambient_correction`, `qc`,
  `preprocessing`, `clustering`, `integration`, `annotation`, `state_scoring`,
  `discovery`, `comparative`, `gene_regulation`, `cell_cell_communication`,
  `trajectory`) now live under `cellquorum.stages.*`, leaving `src/cellquorum/`
  with **10** top-level packages (`core`, `config`, `methods`, `backends`,
  `stages`, `io`, `visualization`, `api`, `cli`, `utils`). Every retired
  re-export shim is deleted (see *Removed*), so each stage, config, and utility
  has exactly one import path — frozen by `tests/test_old_paths_removed.py`. The
  convenience top-level re-exports on the `cellquorum` package are unchanged:
  `cq.run_pipeline`, `cq.tl` / `cq.pp` / `cq.diag` / `cq.evidence`, and
  `cq.utils` stay canonical. New legibility docs land alongside: per-stage
  `# Pipeline step (order=…)` module headers, a catalog-pinned ordered stage map
  (`src/cellquorum/stages/README.md`), a file-level run walkthrough
  (`docs/how-it-works.md`), and a README START-HERE section. **Breaking**
  (permitted pre-1.0): code importing a retired deep path must move to the
  canonical one — `cellquorum.stages.<stage>` for stages, `cellquorum.utils` for
  the reusable helpers, `cellquorum.api.<ns>` for the notebook namespaces (or the
  unchanged `cq.<ns>` top-level alias).
- **Consolidation round 2 (#187).** *(Superseded in part by the API clean break +
  stage-package regroup above: the package counts and import-compatibility claims
  in this entry describe the tree as of #187 — every compatibility shim has since
  been removed, and the 12 pipeline-step packages regrouped under
  `cellquorum.stages.*`.)* Continued the source-tree consolidation
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

### Removed

- **All 12 backward-compatibility re-export shims (API clean break).** The
  pre-consolidation import paths no longer resolve — each raises
  `ModuleNotFoundError`, frozen gone by `tests/test_old_paths_removed.py`: the
  four comparative-analysis package shims `cellquorum.differential_expression`,
  `cellquorum.differential_abundance`, `cellquorum.enrichment`,
  `cellquorum.multicellular_programs`; the five notebook/API standalone-module
  shims `cellquorum.tl`, `cellquorum.pp`, `cellquorum.diag`, `cellquorum.evidence`,
  `cellquorum._notebook`; and the three QC submodule shims `cellquorum.qc.ambient`,
  `cellquorum.qc.publication`, `cellquorum.qc.visualization`. Canonical
  replacements: import comparative analyses from `cellquorum.stages.comparative.*`,
  the reusable helpers from `cellquorum.utils`, and the notebook namespaces from
  `cellquorum.api.*`. This removes the *module* paths `cellquorum.tl` etc.; the
  `cq.tl` / `cq.pp` / `cq.diag` / `cq.evidence` attribute re-exports on the
  top-level package are retained.
- **Old top-level paths of the 12 relocated stage packages.** Importing a
  pipeline step as `cellquorum.<stage>` (e.g. `cellquorum.qc`,
  `cellquorum.comparative`) now raises `ModuleNotFoundError`; the canonical path
  is `cellquorum.stages.<stage>`.

### Fixed

- **SoupX wrote `rho_per_cell.csv` before anything created the output directory.**
  `soupx_per_library.R` relied on `write_mtx()` to create `out_dir`, but that runs *after*
  the per-cell rho table is written, so a fresh output directory failed with R's "cannot
  open the connection". `out_dir` is now created up front, before any write. This was the
  cause of both real-data ambient-correction test failures.
- **`make lock` targeted impossible platforms (#189).** The conda-lock recipe
  ran `conda-lock lock` with no `--platform`, so it attempted a multi-platform
  solve (osx-64/win-64 included). The GPU env carries linux-only CUDA packages
  (`pytorch-cuda`, the `nvidia` channel) with no osx/win build, so the solve
  aborted before any lock file was written. The target now pins `-p linux-64` —
  the image's only build platform and the only one we ship — matching the docs
  in `envs/README.md` and `docs/backends.md`.

- **scCODA counted every credible effect twice.** The fitting helper returns two fits
  stacked in one table whenever a reference is resolved — one at the engine's reference,
  one at scCODA's own automatic pick — told apart only by a `reference` column, and
  resolving a reference is the default. `n_credible` was summed over the stacked frame, so
  a cohort with three credible effects was reported as having six, on every scCODA run the
  engine had produced. The reported fit is now split out before anything is measured
  (`split_reference_fits`, the single place that decides which stacked fit is the reported
  one — the figure delegates to it so the plot and the metrics cannot disagree), and the
  written table marks which rows are the result. The second fit is no longer wasted: it is
  reported as a denominator-sensitivity check (`credible_set_reference_stable`), since
  every compositional effect is relative to its reference and "unchanged under a different
  reference" is the robustness statement a reader of a compositional result needs.
- **Adding one optional config field invalidated every checkpoint on disk.** Fingerprints
  hash the *resolved* config, which carries every field a model declares — so a new
  optional setting arrived as `None` in runs that never mentioned it and changed their
  hash. A finished run's checkpoints went stale on upgrade, and the guard reported that "a
  setting changed" for a setting that did not exist when the checkpoint was written.
  `None` is now treated as absent (as is a sub-block holding nothing), while a field
  holding an actual value — including `0` or `false` — still invalidates. The fingerprint
  schema version is recorded in each checkpoint sidecar, and a checkpoint from an older
  schema is refused with a message naming the engine upgrade as the cause instead of
  accusing the config.
- **Every R-backed method could not find its script after the #167 consolidation.** The
  methods located `r_scripts/` by walking `__file__.parent` a fixed number of times, and
  the consolidation changed how deep each module sits — so all eleven R-backed methods
  reported themselves unavailable and skipped, which looks exactly like a missing R
  install. Script paths now resolve from the package root
  (`cellquorum.backends.script_paths.r_script_path`), which cannot drift with module
  depth, and `tests/test_r_script_paths.py` asserts every referenced script resolves to a
  real file.
- **Paired concordance rendered cell-type labels its own way.** It is callable on any
  count matrix, not only one from `aggregate_celltype_counts`, so a raw numeric state
  column came out `"1.0"` there and `"1"` everywhere else — a silently empty join on the
  qualification merge. It now canonicalizes through the shared label helper.

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

### Performance

- **Per-sample scDblFinder runs in one R process instead of one per capture, and is
  now reproducible.** Per-capture doublet detection was driven from Python: subset the
  object, launch `Rscript`, `library(scDblFinder)`, score ~120 cells, exit — once per
  sample. On the 18-capture LEC arm of the mechanotransduction run that was 136s of a
  150s QC stage, of which ~95s was loading the same three R libraries eighteen times.
  scDblFinder has taken the sample assignment itself via `samples=` for years, with the
  same default `multiSampleMode="split"` semantics, so the split now happens inside the
  one session. Measured on that arm: **149.6s → 40.7s serial, → 18.3s across 8 workers
  (8.2×)**, and the whole QC stage 150s → 56s.

  Agreement with the old path, on the same 2,125 cells: per-cell score Pearson **0.978**
  (Spearman 0.974), per-capture mean-score correlation **0.998** across the 18 captures,
  68 vs 73 cells called doublets (0.24% of cells). The residual is RNG, and it now falls
  on the better side of it — the R adapter pins a per-CAPTURE L'Ecuyer stream via
  BiocParallel's `RNGseed`, so scores are **bit-identical at 1, 4 and 8 workers**, where
  the old path's own doublet count wandered 73 → 76 between two seeds because each
  subprocess reseeded the whole pipeline for its one sample. `compute.n_jobs` is what
  chooses the worker count, which makes QC the first stage to honour that field; it
  defaults to 1, so an untouched config keeps serial behaviour.

  Scrublet is unaffected and still splits in Python — it has no `samples=` equivalent,
  and handing it the pooled object would quietly turn per-capture detection into pooled
  detection. `tests/test_qc_doublets.py` pins which detectors are allowed to skip the
  Python-side loop, that the sample column really reaches R, and (under real R) that the
  scores come back in input cell order rather than in whatever order scDblFinder
  rebinds its splits.
- **Backend availability probes are cached for the lifetime of the process.**
  Asking "is celloracle importable in `celloracle_env`?" runs
  `micromamba run -n celloracle_env python -c 'import celloracle'`, and importing
  celloracle costs ~7.7s. Nothing cached the answer, so
  `BackendRegistry.to_status_table()` measured 8.5s / 7.3s / 7.0s across three calls
  in one process, and every planner or CLI test paid the same toll despite running no
  analysis. Repeat calls now cost ~0.2s (37× faster), including across the freshly
  constructed backend instances that `build_default_backend_registry()` returns —
  which is why the cache lives at module scope in `backends/_probe.py` rather than on
  the backend objects. Availability cannot change mid-process, so this is safe.

  This is a win for the test suite and for any long-lived process (the notebook API,
  a future server); it does **not** speed up a single `cellquorum plan`, which probes
  once either way. Making that faster needs lazy probing — only checking backends that
  the enabled stages actually require — which is not done here.
- **`import cellquorum` is ~300× cheaper, and the CLI starts ~45× faster.** A bare
  import went from 3.71s / 7312 modules to 0.002s / 4 modules, and
  `cellquorum --version` from 4.75s to 0.10s. Two causes, both fixed:
  - `stages/annotation/reference_mapping/__init__.py` probed for scvi with
    `import scvi` inside `try/except ImportError`. The guard prevented a crash but
    still *executed* the module whenever scvi was installed, dragging in torch,
    lightning, and jax. Because `config/models.py` imports that stage's config,
    the cost landed on every entry point — merely validating a YAML file loaded
    PyTorch. It now probes with `importlib.util.find_spec`, which answers the same
    question without executing the module and preserves the existing
    method-registration behavior exactly.
  - The public surfaces in `cellquorum/__init__.py` and `cellquorum/utils/__init__.py`
    now resolve lazily via PEP 562 `__getattr__`, and `cli/app.py` imports the engine
    inside each command body rather than at module scope. `from cellquorum import pp`
    and `from cellquorum.utils import get_net` work unchanged.

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
