# Deployment goal and acceptance sequence

The target is a deployable single-cell analysis package for individual workstations,
with CPU and NVIDIA GPU capabilities. A wet-lab scientist interface is the final
phase. Scientific correctness, reproducibility, reliable execution, intuitive
organization, and publication figures precede UI work.

## Required outcomes

1. Statistical results have explicit biological replicate identity, study design,
   contrasts, excluded observations, and provenance. Rendering consumes results;
   it cannot silently choose statistical tests or treat cells as donors.
2. Every supported execution profile runs from an installed wheel, writes reusable
   results and figures, and has documented backend/environment requirements.
3. GPU execution is measured, not inferred from a configuration flag. Strict GPU
   requests fail if GPU compute is unavailable; permitted fallback is reported.
4. Checkpoint continuation preserves results within documented numerical tolerance,
   and changed upstream settings cannot reuse stale state.
5. Releases require the existing quality jobs and complete installed-workflow
   acceptance. Backend capability claims require corresponding actual backend tests.
6. One figure contract supports the reference aesthetic, deterministic gallery
   fixtures, source-table agreement, layout validation, and editable vector exports.
7. Supported environments, memory limits, and workflow limitations are explicit.
   Study-specific scripts and parallel implementations are consolidated by purpose.
8. The interface exposes these verified workflows last, on the same engine. Browser
   state must not determine job correctness or persistence.

## Current work

The real-data QC evidence is maintained in the
[GSE140819 benchmark](../benchmarks/gse140819.md), including verified preparation
pairings, reference-label limitations, CLL baselines, and detector comparisons.
These runs exposed and drove fixes for ignored detector seeds, ignored explicit
thresholds, and the missing scDblFinder expected-rate argument. Detector outputs
now validate identity, dimensions, values, and sample labels, and report scoring
coverage. This is correctness and observability progress, not demonstrated
biological accuracy. The remaining matched specimens, raw-droplet recovery,
ambient correction, miQC/SampleQC evaluation, and independent truth sets remain
required scientific work. UI remains last.

- Moved donor-level comparison into `stats/donor_comparison.py`; pairing is explicit.
- Removed automatic inference from violin and QC dumbbell renderers. QC orchestration
  supplies precomputed paired results and saves their table.
- Donor comparison objects validate finite probabilities/effects, unique donor
  identities, replicate counts, pairing consistency, and exclusion counts at
  construction, including for results supplied directly to renderers.
- Added installed-workflow acceptance for normalization, QC, PCA, clustering,
  embeddings, outputs, checkpoint continuation, and stale-config rejection.
- Made CI reusable and made the release build depend on those quality jobs.
- Enforced GPU fallback policy in the compute router and surfaced fallback warnings.
- Expanded run-level dependency provenance to include rendering, clustering,
  PyTorch/scVI, CuPy, RAPIDS, and CUDA runtime distributions. Versions come from
  installed metadata without importing optional stacks; they do not prove runtime
  availability and are not a complete environment lockfile.
- QC artifact writers use unique sibling temporary paths before replacement.
  Tests verify that failed CSV writes preserve previous results and concurrent
  writers leave one complete CSV without temporary-file collisions. This is
  per-file atomic replacement, not a transaction across all run artifacts or
  a guarantee of power-loss durability.

Artifact failure policies are not yet uniform. QC CSV/JSON replacement failures
preserve an earlier complete file; the shared H5AD writer deliberately removes
an earlier target after failure because existing consumers discover outputs by
filename and could otherwise consume stale data. Before unifying these policies,
consumers must validate output provenance/status rather than file existence alone.
The current H5AD writer also uses a fixed temporary name and is not safe for
direct concurrent writes to one target. Pipeline execution and bootstrap now hold
an operating-system-backed output-directory lock before context setup and through
final artifact writing. A competing cooperating run fails immediately. Tests cover
both entry points, setup failure, and release after an owning process is killed;
the execution, context, and checkpoint acceptance tests passed, as did the engine
type checks. Direct stage/writer calls and external programs do not acquire this
lock. Network filesystem lock semantics are not validated. The lock prevents
overlapping runs; it does not make multiple artifacts a transaction. JSON
replacement-failure and existing H5AD failure tests cover the current behavior;
whole-run transactional recovery remains open.

The wheel rebuilt after the locking change passed the copied installed-workflow
test outside the checkout on CPU (13.24 seconds) and on the working RAPIDS GPU
environment (11.27 seconds). CPU `pip check` reported no broken requirements.
The GPU environment remains the previously repaired local profile, not a freshly
provisioned deployment recipe; its pass does not close that provisioning gap.

These changes do not complete the objective. Cross-environment figure validation, remaining style
consolidation, provisioned backend CI, locked runtime profiles, broader scientific
validation, resource-scale benchmarks, and the final interface remain open.

The broader QC regression run completed with 626 passed and three skipped
SampleQC R tests in the current environment. Real Partipy tests ran. This suite
checks implemented contracts and synthetic cases, not general biological accuracy.
The SampleQC suite was then run with the isolated R library containing upstream
SampleQC 0.6.6: all 13 tests passed with no skips, including real adapter runs
with one, two, and three samples. This resolves those three skips for local
validation; automated provisioning of that R profile remains unfinished.
The run exposed an all-missing heatmap scaling warning; color-scale selection now
uses finite observations and a defined minimum when none are available.

Publication style validation found and fixed seaborn overwriting explicit font
sizes and axis widths. The shared renderer now restores Matplotlib rcParams after
successful and failed callbacks, preventing style leakage between sequential
panels. The [figure gallery](../gallery/index.md) now contains six visually inspected
synthetic examples: enrichment, categorical embedding, paired donors, GSEA running
score, volcano, and QC summary. Each has source data and PNG/PDF/SVG exports.
Automated tests regenerate all 18 exports, check editable SVG text and selected
source-data invariants. Reference-image comparisons and arbitrary-data layout
validation remain outstanding; current checks do not establish visual equivalence
to the supplied examples.

Additional renderer fixes prevent volcano effect clipping at the former ±3.2 cap
and reject invalid probability thresholds. The shared embedding helper rejects
orders that omit populations. The production PAGA renderer displays missing-label
cells as an explicit gray category without modifying source annotations or PAGA
category indexing. These are figure correctness guarantees, not biological
annotation or rare-population detection validation.

The shared renderer also closes newly opened pyplot figures after success or
failure while preserving pre-existing figures. A repeated-failure regression
checks that 25 failed callbacks leave the open-figure count unchanged; saved
paths and failure warnings remain available. This limits figure accumulation in
long runs using this helper, but does not establish whole-pipeline memory bounds.

## Validation commands

The configured CI lint/format scope is `src tests`, and the type-check gate covers
the engine spine (54 source files), not the entire package. Both passed locally
after the GPU, precision, and rendering changes; strict documentation also built.
An additional `ruff check src tests scripts` run reported 57 errors in ignored,
local study-specific `scripts/le_*.py` files. Inspection with `git ls-files`
confirmed these are not tracked package tooling. The seven tracked scripts pass
lint and formatting, and CI now checks `scripts` separately as well. Local study
scripts remain outside the release scope; passing current CI must not be
described as whole-workspace lint or whole-package type completeness.

The wheel acceptance test is self-contained. Copy
`tests/acceptance/test_installed_workflow.py` outside the checkout, install the wheel
and pytest into a fresh environment, and run the copied test with
`CELLQUORUM_REQUIRE_WHEEL=1`. It asserts the imported package is in site-packages.

After the result-invariant and renderer-lifecycle changes, rebuilt-wheel acceptance
passed on CPU (13.11 seconds) and RAPIDS GPU (11.15 seconds). It now also compares
retained raw counts to the source file and resumed counts to the completed run,
checks finite PCA/UMAP coordinates, and verifies recorded float64 PCA arithmetic
in the GPU profile. These checks remain limited to the synthetic fixture below.

For GPU acceptance also set `CELLQUORUM_ACCEPTANCE_BACKEND=gpu` in an environment
with working CuPy, rapids-singlecell, and CUDA. This mode refuses missing GPU
capabilities and verifies GPU compute in both PCA and clustering provenance. R and
PyTorch/scVI capability tests are separate obligations; RAPIDS does not validate
all GPU methods.

Acceptance accepts only `cpu` or `gpu` (default `cpu`); unknown values fail during
test collection instead of silently selecting CPU. CI declares its CPU profile
explicitly. This prevents a mistyped GPU selector from yielding a misleading
successful CPU acceptance result.

CPU acceptance does not claim doublet or ambient correction: those capabilities are
explicitly disabled in this small portable fixture. It tests the listed execution
contract. Broader assay and QC-method acceptance must be added separately.

### Release artifact acceptance

Gallery reproducibility was checked separately from workflow correctness. All six
PNG figures regenerated with identical pixels in the development renderer
(Matplotlib 3.10.8, NumPy 2.3.5, Pillow 11.0.0, FreeType 2.6.1). Rendering through
the installed-wheel environment (Matplotlib 3.11.1, NumPy 2.5.3, Pillow 12.3.0)
changed five image dimensions and produced RMS difference 27.75 for the remaining
QC image. These results require a controlled rendering environment for a pixel
comparison gate; no broad image tolerance has been introduced to mask the drift.
Comparison artifacts are local under `/tmp/cellquorum-gallery-wheel-repeat`.
Pinning those three packages in an isolated target directory restored identical
pixels for all six figures through the installed-wheel environment. The gallery
test now supports exact reference comparisons, and a separate CI job enables
them with `envs/figure-constraints.txt`. The local baseline test passed; the
GitHub job has not yet run. A new venv at `/tmp/cellquorum-figure-ci`, without
system-site packages, installed `.[dev]` with the same constraints and passed the
baseline-enabled gallery test for all six figures. `pip check` reported no broken
requirements. Font availability and remaining transitive packages are not fully
locked.

The repository-wide regression run (`python -m pytest -n 2 -q`, with numerical
thread counts capped at one) completed with exit code zero. Optional tests skipped
and warnings were reported; xdist disabled performance benchmarks. This validates
the available local test environment, not every backend or production-scale load.
The captured output is `/tmp/cellquorum-full-regression.log` (local, ephemeral).

The release job now runs the installed analysis and checkpoint acceptance test
against its own rebuilt wheel before uploading the distributions for publication.
This closes the gap between the wheel tested by reusable CI and the separately
built release artifact. Both workflows explicitly register the integration marker
and enforce strict marker validation when running outside the checkout.

Local validation rebuilt the current wheel, installed it into the existing clean
CPU acceptance environment, and ran the copied test outside the repository:
one test passed in 12.78 seconds. Workflow YAML parsed successfully. The GitHub
release workflow itself has not run, and this CPU gate does not establish GPU
release coverage; GPU provisioning and a hardware-backed release gate remain open.

## GPU validation status

Lock generation with conda-lock 4.0.2 is not yet successful. Recipe parsing was
fixed by selecting exact PyTorch wheel URLs with upstream SHA-256 fragments.
Both subsequent lock attempts failed during wheel selection: CPU on jaxlib
0.11.1 and GPU on nvidia-cublas-cu12 12.9.1.4. The tool's static manylinux tag
list omits `manylinux_2_27`; the published Python 3.12 Linux jaxlib wheel uses
exactly that tag. This is a lock-tool compatibility gap, not evidence that the
already tested environments failed. A maintainable lock-generation solution
and installation from those locks are still required.

The recipe-created CPU environment now passed all nine copied CPU workflow,
checkpoint, CellRank estimator, and PETSc signal-handling checks (23.27 seconds,
no skips). It uses PyTorch 2.8.0+cpu, with no compiled CUDA runtime, and `pip check`
is clean after installing the latest wheel and Cython. Cython has also been
installed into the GPU test environment, satisfying BBKNN/Scrublet metadata;
the separate RAPIDS metadata discrepancies remain. Both source recipes now
declare Cython. These are local provisioning checks, not remote CI or fully
locked release environments.

Single-command GPU recipe creation completed successfully, followed by installation
of the latest built wheel and pytest. All 14 copied checks passed in that
environment (34.95 seconds, no skips). The recipe subsequently gained Cython to
satisfy BBKNN/Scrublet dependency metadata; that addition has not yet been
installed in the test environment. RAPIDS metadata inconsistencies remain.
The CPU recipe now selects PyTorch 2.8.0's CPU wheel explicitly, and its first
single-command installation is running under `/tmp/cellquorum-cpu-recipe-final`.

The full fresh environment passed all 14 copied acceptance, CUDA model-training,
CellRank estimator, and PETSc signal-handling tests (24.98 seconds, no skips).
This required CellRank 2.3.2 and AnnData 0.12.19: both CellRank 2.2 and 2.3.2
failed to import against AnnData 0.13. The MPI negative-control test now checks
nonzero exit and the abort diagnostic without assuming MPI's capitalization.
The GPU recipe has been updated with the compatible accelerator versions,
AnnData bound, CellRank pin, and upstream pygpcca commit. A new single-command
creation directly from that checked-in recipe is running under
`/tmp/cellquorum-gpu-recipe-final`; earlier full-environment validation included
separate corrective installation steps. Clean dependency metadata and full
transitive locks remain unresolved.

Fresh provisioning investigation isolated a concrete dependency conflict:
RAPIDS 26.4's CUDA-12 libcudf requires `libnvjitlink >=12.9.86`, while
`pytorch-cuda=12.4` requires a version below 12.5.40. The accelerator-only solve
failed with that conflict. The obsolete full CUDA 12.1 and 12.4 solves were
intentionally terminated after this evidence; they did not complete successfully.
A Python 3.12/RAPIDS 26.4/CuPy 14.1.1/CUDA 12.9 solve succeeded (262 conda
packages), and installation from its explicit URLs and MD5 hashes completed
under `/tmp/cellquorum-rapids129-check`. A CuPy allocation/kernel/reduction passed
through `micromamba run -p /tmp/cellquorum-rapids129-check`; invoking its Python
directly without activation initially failed to discover the installed CUDA
headers. The matching PyTorch 2.8.0 CUDA-12.9 wheel is being installed from the
official PyTorch index. That installation completed, and both PyTorch and CuPy
executed CUDA reductions in the same process; cuml/cudf/cuvs also imported.
Installing the full analysis stack then failed dependency resolution because
pygpcca 1.0.4 requires `jinja2==3.0.3`, conflicting with CellQuorum's `jinja2>=3.1`.
The currently released CellRank 2.3.2 still depends on pygpcca >=1.0.4, so merely
upgrading CellRank does not resolve this. Upstream pygpcca commit
`773c463cc5043a5d40e6f31bbbc20f4e49587008` removes that runtime dependency;
installing this exact commit allowed the wheel and pinned scVI, RAPIDS single-cell,
communication, and trajectory packages to install while preserving accelerator
versions. Both `verify_gpu.py --profile all` probes passed. `pip check` still
reports conda RAPIDS distribution metadata requirements for CUDA-13 Python wheels
and native libraries supplied by conda; this inconsistency remains unresolved.
Installed-workflow acceptance in this fresh profile passed in 21.98 seconds,
including GPU PCA/clustering provenance, saved outputs, and checkpoint recovery.
All five copied scVI/scANVI tests also passed (4.83 seconds), with assertions that
models actually trained on CUDA. A complete conda recipe solve, including the
remaining core dependencies and sparse trajectory solvers, succeeded with 467
conda packages. Installation from the explicit resolved package URLs/hashes
completed under `/tmp/cellquorum-full-gpu129`; its pip portion is installing,
and workflow tests remain to be run. The candidate uses CUDA 12.9 and the pinned upstream pygpcca
source. The earlier tested environment is a partial environment, not a
validated replacement for the full GPU recipe. CUDA-compatible PyTorch, scVI,
remaining analysis dependencies, and actual workflow validation are still required.

Container verification now has a `make smoke-gpu` target that requests GPU access,
checks the core/R/GPU environments, plans in the GPU runtime, and executes both
accelerator probes. Make expansion was checked locally; container execution is
unverified because Docker Desktop is not integrated with this WSL distribution.
The GPU recipe's CUDA compatibility correction is still under dependency-solve
investigation, so this smoke target is not evidence of a working GPU image.

Local installed-wheel acceptance now passes with explicit float64 PCA. The existing `cellquorum-gpu`
environment cannot import RAPIDS because `cuvs` is missing. In an isolated wheel
environment based on `rapids_sc`, adding CUDA 12.9 runtime/toolkit dependencies
resolved the kernel compilation failure. This environment uses Python 3.14.6,
CuPy 14.1.1, and rapids-singlecell 0.15.2 on the RTX 4000 Ada; it is diagnostic
evidence, not a cleanly provisioned supported profile.

The initial float32 workflow produced results and figures with GPU PCA and clustering
confirmed in execution records. Resume produces identical normalized matrices and
cluster labels, but PCA coordinates differ by up to approximately 0.00051, failing
the existing `rtol=1e-5, atol=1e-6` check. Three direct PCA calls on identical
normalized input reproduced variation in float32 (maximum differences around
0.00030); the float64 diagnostic produced identical coordinates. This small
fixture does not establish float64 scalability or general determinism. Precision,
memory cost, and justified numerical tolerances remain to be resolved; the
acceptance tolerance has not been relaxed.

The standard PCA configuration now accepts `dimensionality.precision` values
`input` (default), `float32`, and `float64`. The normalized source layer is restored
even if PCA fails; arithmetic dtype is saved in `uns['pca']['params']`. Explicit
precision is rejected for scclr, whose external implementation does not expose
this contract. Float64 increases working memory; it is an explicit choice rather
than an unconditional GPU default.

With `precision: float64`, the installed GPU fixture passed in 15.22 seconds,
including figures, result serialization, checkpoint continuation at the unchanged
numerical tolerance, identical cluster labels, and stale-config rejection. This
does not establish performance at production scale or validate scVI, ambient RNA,
doublet detection, or every GPU algorithm.

`python scripts/verify_gpu.py --profile rapids` now verifies allocation, kernel
compilation, and reduction as well as imports. `--profile scvi` checks PyTorch CUDA
and requires scVI importability; `--profile all` checks both independently. These
probes do not validate model training or replace workflow acceptance.

Five installed-wheel scVI/scANVI GPU tests passed on the separate Python 3.12
environment with PyTorch 2.5.1/CUDA 12.1. They exercise short training runs,
encoding excluded cells, invariance of core-cell coordinates to excluded cells,
and the disclosed all-cell training behavior when core cells omit a batch. They
do not establish biological embedding quality or large-cohort performance.

Normalization currently uses CPU matrix recipes or the separate scclr backend.
The preprocessing stage no longer probes RAPIDS for these implementations and
records its actual compute implementation. GPU-enabled workflows can therefore
use CPU normalization without requiring an unrelated RAPIDS installation solely
for that stage. GPU fallback policy still applies to stages that actually route
between GPU and CPU implementations.
