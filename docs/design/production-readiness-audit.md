# Production readiness assessment

Assessment date: 2026-09-10. Scope: current working tree, including the ongoing QC
refactor and SampleQC integration. This is a sampled engineering assessment, not
a certification, a code-coverage percentage, or the percentage of work completed.

## Addendum, 2026-10-02: what has changed since this assessment

The assessment below is kept as written. These findings no longer hold, or have been closed:

- **Donor-count defect (issue 1):** fixed. `two_group_test_on_donor_medians` now returns `None` for
  three donor labels with one missing and `min_donors=3`; paired versus unpaired must be declared.
- **Image-baseline figure tests:** 72 pixel-exact baselines (`scripts/gallery_*.py`,
  `docs/gallery/assets`, `tests/test_figure_gallery.py`) run in CI and cover every tuned figure
  family. Two figures were order-dependent through third-party import side effects
  (`pycirclize`, `decoupler`) and are now deterministic.
- **Installed-wheel acceptance (issue 2):** `tests/acceptance` runs from a wheel-only virtual
  environment in CI: the Python API (analysis, resume, stale-config rejection) and the command
  line (plan, run, re-run, and the usual user mistakes giving one readable error).
- **Release gating:** `release.yml` checks tag against version, requires a changelog entry, and
  runs the whole CI workflow (including the acceptance job) before it builds or publishes.
- **Type checking:** the CI mypy gate now has no exempted modules and covers `core`, `config`,
  `io`, `methods`, `cli`, `stats`, `datasets` and five small stage packages. Whole-package errors
  fell from 619 to 379.
- **Coverage:** measured at 86.1% (85.9% with the R/GPU tiers excluded) and gated at 84%.
- **Silent failures:** thirteen handlers that swallowed an error without a trace now report it.
- **Style entry points:** one `figstyle.apply_theme(style)` replaces three setters and seven
  per-stage wrappers, with every figure's look unchanged.

Still open from this assessment: provisioned R and GPU CI jobs, fully resolved runtime environment
locks, scientific validation on independent datasets, and the remaining type errors and long
functions.

## Estimate and scoring basis

**Approximately 60/100; a reasonable judgment range is 55–65.** Define production
as a documented set of supported workflows that installs reproducibly, executes
with validated scientific semantics, records failures and partial results honestly,
resumes safely, and produces reproducible publication outputs within tested resource
limits. Supporting every single-cell assay automatically is not an acceptance test.

Scores describe demonstrated maturity: 0 absent; 25 prototype; 50 functional with
material gaps; 75 consistently tested on supported workflows; 100 release acceptance
criteria demonstrated. Weights reflect this project's scientific and operational
priorities. They are judgment calls, not measured probabilities.

| Area | Weight | Score | Evidence and remaining gap |
| --- | ---: | ---: | --- |
| Architecture and readability | 20% | 60 | StageResult, configuration models, backend boundaries and public API exist. Large modules still mix responsibilities; style, rendering and statistical inference coexist. |
| Runtime reliability and verification | 20% | 65 | Checkpoints, fingerprints, atomic figure writes and numerous tests exist. Routine CI has no provisioned R/GPU test jobs; backend tests can skip. Full-scale resource and failure-recovery acceptance remain unproven by this audit. |
| Scientific validation and generalization | 20% | 45 | Data/label/statistical contracts and a generic CPU cohort smoke test exist. Scientific equivalence across assays and independently labeled datasets is not established; a donor-validation defect was reproduced. |
| Packaging, reproducibility and releases | 15% | 65 | Wheel builds, metadata checks, clean-venv import/version smoke checks and a publishing workflow exist. Full installed-wheel analyses are not release gates; release publishing does not depend on test/typecheck jobs. Runtime environment recipes are not complete locks. |
| Publication figures | 15% | 65 | Shared palettes, vector-font settings, atomic export and measured layouts exist. Multiple style entry points remain; no image-baseline comparison tests were found in the searched test tree. Visual equivalence to the reference collection is not yet demonstrated. |
| Documentation and developer experience | 10% | 70 | Architecture/tutorial/backend docs, contribution guidance, changelog and citation metadata exist. An explicit supported-workflow matrix and a compact consistent module map need further work. |

Weighted result: 60.5/100, rounded to approximately 60. Critical scientific or
release failures must block a release regardless of the weighted score.

## Evidence collected

- 340 Python modules under `src/cellquorum`, about 89,493 physical lines including
  comments/docstrings; 364 test files. Counts are inventory, not quality metrics.
- Repository-wide `ruff check src tests`: passed.
- CI's mypy command: passed for 54 source files, with the existing configured
  exclusions. `pyproject.toml` explicitly ignores errors in 14 named modules;
  this is not evidence that the whole package is type-safe.
- Targeted public API, generic CPU workflow, checkpoint, resume, fingerprint,
  figure-style and measured-layout tests: **155 passed, 31 warnings**.
- Earlier in this session: 631 QC regression tests and 13 SampleQC adapter tests
  passed, including real upstream R runs for one, two and three samples.
- Full non-QC scientific workflows, all backend environments, clean wheel execution
  of a complete analysis, sustained atlas-scale loads, and current remote CI/release
  status were not verified in this audit. The old `.coverage` file was not treated
  as current coverage evidence.

Key source locations: `core/stage.py`, `core/executor.py`, `core/checkpoint.py`,
`core/fingerprint.py`, `pyproject.toml`, `.github/workflows/ci.yml`,
`.github/workflows/release.yml`, `visualization/figstyle.py`,
`visualization/figio.py`, and `visualization/measured_layout.py`.

## Concrete issues and priorities

### 1. Statistical validity must be independent of rendering

A reproduced defect in `visualization/figstyle.py::two_group_test_on_donor_medians`:
with donor IDs `[A, B, missing]` in each condition and `min_donors=3`, the helper
returns a paired test labeled `n = 3 paired`. There are only two known donors.
Converting donor IDs to strings before dropping missing rows creates a fictitious
third donor. This is a confirmed behavior, not merely an inspection concern.

The helper also infers pairing from complete donor overlap; partly overlapping
cohorts fall into an unpaired test. Study design should be explicit and validated.
These findings were recorded, not patched, during this assessment.

Move inference into the existing `stats/` boundary. Renderers should consume
validated estimates, uncertainty, p-values/FDR, tested population, donor counts,
and contrast direction. They should not choose a statistical design themselves.
Add donor-missingness, incomplete-pair, contrast-reversal and table/figure-agreement
regressions before migrating callers. Preserve intentional statistical differences
between methods; consolidate common validation and result schemas.

### 2. Establish one complete release acceptance workflow

Start with a small, openly distributable fixture and an explicit CPU profile. Install
the actual wheel outside the checkout, run QC through the supported downstream
analysis and figure generation, verify tables and AnnData round trips, then rerender
from saved results. Test interruption, resume and configuration-change invalidation.

Gate publishing on this acceptance job and the existing lint/type/unit jobs. The
current release workflow verifies metadata and importability but can publish without
waiting for the separate CI test workflow. Add provisioned R acceptance jobs for
advertised R capabilities; a skipped backend test must not count as validation.
Either test Python 3.14 or narrow the advertised range until it is validated.

### 3. Consolidate responsibilities rather than minimizing file counts

Keep the existing top-level boundaries: `core`, `config`, `io`, `backends`, `stats`,
`stages`, and `visualization`. Avoid replacing the engine wholesale.

For each stage, establish a readable sequence: validate inputs, perform computation,
return a typed result, write artifacts, render figures. Keep one owner for each
algorithm and schema. Public StageResult already provides a useful outer contract;
method-specific results should carry scientific details without ad hoc dictionaries.

The 1,690-line `figstyle.py` currently contains several style setters, saving helpers,
plot primitives and statistical testing. Consolidation should remove competing
implementations while separating genuinely different responsibilities. A small,
predictable set of modules is preferable to either many tiny wrappers or one giant
module. Historical implementation commentary belongs in design notes and tests.

### 4. Make environments and failure states reproducible

Maintain reasonable library dependency ranges alongside fully resolved supported
runtime environments. These serve different purposes, as described in the
[PyPA dependency guidance](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/).

Record package/backend versions, reference versions, seeds and resolved configuration.
Declare required versus optional outputs and distinguish failure, intentional skip,
and degraded completion. Audit broad exception handlers at stage boundaries; the
205 occurrences found under `stages/` are review targets, not 205 proven defects.

Define CPU/R/GPU capability profiles and memory/runtime budgets. Existing memory
invariant tests are useful foundations; extend them with representative workloads
and process-level peak-memory measurements. Do not infer atlas-scale reliability
from successful small fixtures.

## Figure reference assessment

The exact supplied Windows path was not present. Two related collections were found:

- `C:\Users\ajboo\BookAbraham\mehrara_analyses\figures` (114 PNG files).
- `C:\Users\ajboo\BookAbraham\mehrara_analyses\Mehrara Lab analysis\figures`
  (104 PNG files).

Pending confirmation of the intended collection, five representative figures were
inspected across both directories: the all-cell-type volcano grid, mechanosensory
program panel, PROGENy LEC contrast, annotated cell atlas, and LEC GSEA leading-edge
panels. These are visual references, not evidence validating their statistical results.

The desired visual language is concrete:

- White backgrounds, clear sans-serif typography and bold hierarchical titles.
- Consistent control-blue/case-red direction and centered diverging color scales.
- Saturated categorical colors, readable cell-type labels and population counts.
- Donor points and pairing lines, effect sizes, explicit significance legends.
- Compact multi-panel compositions and pathway/gene groupings.
- Aligned legends, panel labels, minimal spines and informative method captions.

The package already includes much of this in its figure primitives. The next step
is a single supported visual contract and reusable figure templates, not another
plotting framework. Preserve the aesthetic while improving label collisions,
legend placement and excessive empty space visible in some reference panels.
Biological names and case/control mapping must remain configuration inputs.

Proposed structure within the existing visualization package: a single theme API,
measured layout utilities, reusable primitives, figure-family templates, and one
export/manifest API. Statistical inference belongs in `stats/`. Consolidate existing
code into these responsibilities; do not retain parallel legacy rendering paths.

Build a deterministic reference gallery using synthetic/public data: UMAP, QC
summary, volcano grid, paired donor comparison, enrichment dot plot, and GSEA panel.
Save PNG previews and editable PDF/SVG outputs, with the source table and rendering
configuration linked through the artifact manifest. Dense point clouds can be
rasterized while text/axes remain vector graphics.

Add image comparisons with pinned fonts/rendering dependencies and appropriate
numeric tolerance, plus structural checks for labels, legend semantics, bounds,
contrast direction and source-table agreement. Matplotlib provides
[image-comparison testing utilities](https://matplotlib.org/stable/api/testing_api.html).
Existing style and layout tests should be retained. Do not commit private study data
or its figures as public test fixtures.

## Recommended implementation order and exit criteria

1. **Correct statistical boundaries:** fix the reproduced donor issue; require
   explicit study design; prove figure statistics match saved analysis results.
2. **Freeze a small figure contract/gallery:** extract the desired reference style;
   consolidate its setters and export path; verify adaptive labels and legends.
3. **Complete one installed-wheel workflow:** require real outputs, safe resume,
   deterministic rerendering and release gating.
4. **Migrate stages incrementally:** use the same result and artifact contracts,
   beginning with preprocessing/integration and donor-level comparative analyses.
5. **Expand supported profiles through evidence:** provision backend CI, freeze
   environments, benchmark biological behavior and memory, document supported limits.

Do not score progress by script deletion or new method count. Reassess maturity when
these acceptance criteria are demonstrated on the intended deployment environments.
