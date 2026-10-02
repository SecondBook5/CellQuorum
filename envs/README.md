# CellQuorum Environment Architecture

CellQuorum uses a layered environment strategy to isolate incompatible dependencies. This document describes the primary environments and the isolated backend environments.

## Primary Environments

### `cellquorum-core`
The main runtime environment containing CellQuorum's core dependencies. Install
the CellQuorum wheel separately after creating the environment. Both CPU and GPU
recipes bound AnnData to 0.12 for CellRank compatibility and use the same pinned
upstream pygpcca source to avoid its released Jinja2 dependency conflict. PyTorch
2.8.0's CPU wheel avoids downloading CUDA libraries for CPU workstations.
Both recipes select Python 3.12/Linux x86_64 PyTorch wheels using direct URLs
and SHA-256 hashes published in the official PyTorch wheel index. This avoids
pip index-option lines that `conda-lock` cannot parse. These recipes are not
native Windows, macOS, or ARM profiles.
SciPy is pinned to 1.16.3 in both recipes, matching the validated runtime and
preventing the mixed conda/pip solver from assigning it to both package managers
at different versions.

A fresh recipe-created CPU environment, followed by wheel/pytest/Cython
installation, passed all nine installed-analysis, checkpoint, CellRank estimator,
and PETSc signal-handling checks. `pip check` was clean. Cython is now included in
the recipe itself. This is local Linux/WSL validation; it does not cover every
optional method or establish a fully locked environment.

### `cellquorum-dev`
Development environment extending `cellquorum-core` with testing, linting, and build tools (pytest, ruff, mypy, sphinx, etc.).

### `cellquorum-gpu`
GPU-enabled environment for GPU-accelerated analyses, extending `cellquorum-core` with CUDA-enabled PyTorch and scvi-tools.

The recipe now selects CUDA 12.9, RAPIDS 26.4, CuPy 14.1.1, and the PyTorch
2.8.0 CUDA-12.9 wheel. CellRank 2.3.2 uses AnnData 0.12 because its import fails
with AnnData 0.13. The exact upstream pygpcca commit removes the obsolete Jinja2
pin in its PyPI release. Git is included for that source installation.

A fresh Linux/WSL environment assembled with these corrections passed 14 tests:
installed GPU analysis and checkpoint recovery, scVI/scANVI CUDA training, and
CellRank/PETSc execution checks. Both GPU probes passed. This does not validate
every optional method, hardware model, or production-scale dataset. A generated
lockfile now records the resolved dependencies; fresh installation and workflow
validation of that exact repository lock remain outstanding. Conda RAPIDS packages also expose Python
metadata inconsistent with their CUDA-12/native conda dependencies, so `pip check`
is not clean. That packaging issue remains unresolved.

Create using flexible channel priority and the declared channels, then install
the built CellQuorum wheel separately:

```bash
micromamba create --override-channels --channel-priority flexible -f envs/cellquorum-gpu.yml
micromamba run -n cellquorum-gpu python -m pip install dist/cellquorum-*.whl
micromamba run -n cellquorum-gpu python scripts/verify_gpu.py --profile all
```

Activate the environment or use `micromamba run`; direct interpreter invocation
without activation initially failed CUDA header discovery during validation.

Validate both accelerator families with `python scripts/verify_gpu.py --profile all`,
then run installed-wheel acceptance with `CELLQUORUM_ACCEPTANCE_BACKEND=gpu`.
The probes exercise device operations; the acceptance test exercises PCA,
clustering, saved outputs, and checkpoint continuation. Neither establishes
production-scale memory limits or validates every optional method.

### `cellquorum-r`
R bridge environment for the R/Bioconductor methods dispatched over the Rscript
(and rpy2) bridge: pseudobulk DE (edgeR/limma/DESeq2), SingleR, scran/scater,
scDblFinder, batchelor, and the regulatory-network tools (AUCell, GENIE3,
dorothea, viper). It also carries the DIALOGUE `Depends:` from conda-forge.

**DIALOGUE (multicellular_programs):** DIALOGUE is a GitHub-only R package with
no conda/CRAN release, so it is not created by the env solve. Its CRAN
`Depends:` are provisioned from conda-forge in `cellquorum-r.yml`; the one
`Depends:` absent from conda-forge (`unikn`) and DIALOGUE itself are installed
after the env is created. To run the `multicellular_programs` stage in a local
`cellquorum-r` env, install them at the same pins the Docker image uses:

```bash
micromamba run -n cellquorum-r Rscript -e '
  options(repos = c(CRAN = "https://packagemanager.posit.co/cran/2025-06-02"));
  install.packages("unikn");
  remotes::install_github(
    "livnatje/DIALOGUE@9c146ccf28d7706aaa60d00947a9126b4e75fd69",
    dependencies = FALSE, upgrade = "never");
  suppressPackageStartupMessages(library(DIALOGUE))'
```

## Isolated Backend Environments

Five analysis backends live in isolated environments because they cannot coexist with the core environment's dependency stack. CellQuorum invokes these backends as subprocesses via `micromamba run -n <env_name> python <script> [args...]` (or `Rscript` for R backends), exchanging data through temporary files. **The environment names below are hardcoded in the backend subprocess calls — you must use these exact names when creating the environments.**

### `pyscenic_env` (pySCENIC GRN backend)
**Isolation rationale:** pySCENIC (classic GRNBoost2 → cisTarget → AUCell pipeline) has a version-brittle dependency stack that pins:
- `numpy=1.23.5`
- `pandas=1.5.3`
- `setuptools<81`

These pins conflict with the modern dependency stack required by cellquorum-core (anndata>=0.11, scanpy, scvi-tools).

**Install:**
```bash
micromamba create -n pyscenic_env -c conda-forge -c bioconda \
  'python=3.10' 'numpy=1.23.5' 'pandas=1.5.3' 'setuptools<81' \
  pyscenic loompy
```

Download cisTarget databases (TFs, motifs, rankings) separately and configure via `grn.tfs_path`, `motifs_path`, and `rankings_glob` in your analysis config.

### `scclr` (sparse PFlog1pPF normalization + sparse PCA)
**Isolation rationale:** scclr (a Python shim over the Rust crates runorm/rupca) has two incompatibilities:
- Pins `anndata<0.10.9`, while cellquorum-core requires `anndata>=0.11`
- PyO3 build caps at Python 3.13

**Install:**
```bash
micromamba create -n scclr python=3.13 rust maturin pip
micromamba run -n scclr pip install -e /path/to/scclr
```

**Docker note:** The Dockerfile bakes only the scclr toolchain (Rust, maturin) by default. To bake scclr itself, pass `--build-arg SCCLR_SRC=/path/to/scclr` to `docker build`.

### `sccoda_env` (scCODA compositional differential abundance)
**Isolation rationale:** scCODA pins an older `scipy` version that includes `scipy.signal.gaussian`, which was removed in newer scipy versions required by cellquorum-core. The pertpy integration is also broken in the main environment.

**Install:**
```bash
micromamba create -n sccoda_env python=3.10 pip
micromamba run -n sccoda_env pip install sccoda tensorflow
```

### `celloracle_env` (CellOracle in-silico knockout)
**Isolation rationale:** CellOracle has specific dependency requirements that make env isolation desirable for reproducibility and to avoid conflicts with the main environment's stack.

**Install:**
```bash
micromamba create -n celloracle_env -c conda-forge celloracle
```

The promoter base GRN (hg38/mm10) ships with CellOracle.

### `hdwgcna_env` (hdWGCNA hierarchical co-expression)
**Isolation rationale:** hdWGCNA is an R package with Seurat/WGCNA dependencies that cannot be direct dependencies of the Python-centric cellquorum-core environment.

**Install:** hdWGCNA has no conda package (`r-hdwgcna` does not exist on
conda-forge/bioconda) -- it ships only via GitHub, so creation is two steps:
```bash
micromamba create -f envs/hdwgcna_env.yml
micromamba run -n hdwgcna_env Rscript -e \
  'remotes::install_github("smorabit/hdWGCNA", upgrade = "never")'
```

### `metacells_env`
Isolated env for MetaCell2 (the `metacells` package, Baran & Tanay) — has its own
C-extension build requirements that would conflict with `cellquorum-core`'s pins.
Invoked exclusively as a subprocess (`MetacellsBackend`), never imported in-process.

## Locking Environments

Use `make lock` to generate conda-lock files from the environment YAML recipes:

```bash
make lock
```

This scans environment recipes in `envs/*.yml`, excludes existing
`*.conda-lock.yml` outputs, and generates corresponding lockfiles. A failed solve
stops the command. The wrapper requires conda-lock 4.0.2 in the selected Python:

```bash
python -m pip install conda-lock==4.0.2
make lock LOCK_PYTHON=python
```

`scripts/lock_environments.py` fills missing intermediate manylinux tags in
conda-lock 4.0.2, including 2.27 used by JAX and NVIDIA wheels. It retains the
tool's glibc ceiling and target-version filtering. Changing conda-lock versions
requires reviewing this workaround; the wrapper refuses an unreviewed version.
The repository contains Linux CPU (417 package records) and GPU (528 package
records) locks. CI checks package ownership across conda/pip, artifact hashes,
immutable Git revisions, direct pip requirements, and CPU/GPU profile selection.
These structural checks do not establish successful installation, complete recipe
freshness, runtime compatibility, or scientific accuracy.

Install a generated lock into a new environment without re-solving:

```bash
conda-lock install --micromamba -n cellquorum-core-locked envs/cellquorum-core.conda-lock.yml
conda-lock install --micromamba -n cellquorum-gpu-locked envs/cellquorum-gpu.conda-lock.yml
```

The previous temporary validation environments and unfinished test logs are no
longer available. Fresh installation and installed-wheel acceptance of these
exact repository locks must therefore be repeated before calling them validated
deployment profiles. The earlier recipe-based test results above remain separate
evidence.

Locks are solved for **linux-64 only** — the platform the Docker image builds
on. This is deliberate: the GPU env selects Linux CUDA/RAPIDS packages and a
Linux x86_64 PyTorch wheel, so an
unpinned multi-platform solve would fail on those envs. Run `make lock` on a
linux-64 host (or CI) with `conda-lock` installed; the solve needs network
access to the conda channels.

## Docker Integration

The CPU image contains the core and R environments; the GPU target adds the GPU
environment. Other backends require their separate build targets, and those
targets still have unresolved build issues. See `docs/docker.md` for the actual
target contents and validation limits.

## Optional SampleQC backend

The current `cellquorum-r.yml` does not install SampleQC. The adapter was tested
with upstream SampleQC 0.6.6 at commit
`adf0d97fa6e06a617e85fc4d95a8441175b0788e`. Pin this commit rather than installing
an unspecified default branch. Its R dependencies must be installed first; they
are listed in the upstream `DESCRIPTION` file. `R CMD INSTALL` fails if required
dependencies are missing.

In the intended R environment, with Git and an R compilation toolchain available:

```bash
git clone https://github.com/wmacnair/SampleQC.git SampleQC-source
git -C SampleQC-source checkout --detach adf0d97fa6e06a617e85fc4d95a8441175b0788e
R CMD INSTALL SampleQC-source
Rscript --vanilla -e 'stopifnot(requireNamespace("SampleQC", quietly=TRUE)); print(packageVersion("SampleQC"))'
```

Ensure the Python pipeline's Rscript backend resolves this same R environment.
If using a separate R library directory, expose it through `R_LIBS` to both the
availability check and the Python test process. Validate the adapter with:

```bash
python -m pytest tests/stages/qc/test_qc_sampleqc.py -rA
```

All 13 tests must pass, including the three `test_upstream_sampleqc_smoke` cases;
skips do not validate installation. These tests exercise one-, two-, and
three-sample inputs. They do not establish biological accuracy or justify changing
SampleQC from assessment-only to automatic cell removal.

This pins SampleQC's source, not its full transitive R environment. A freshly
provisioned and locked R profile remains a deployment requirement.
