# CellQuorum Environment Architecture

CellQuorum uses a layered environment strategy to isolate incompatible dependencies. This document describes the primary environments and the isolated backend environments.

## Primary Environments

### `cellquorum-core`
The main runtime environment containing cellquorum and its core dependencies (scanpy, anndata>=0.11, scvi-tools, etc.). This is the environment from which `cellquorum` CLI commands are invoked.

### `cellquorum-dev`
Development environment extending `cellquorum-core` with testing, linting, and build tools (pytest, ruff, mypy, sphinx, etc.).

### `cellquorum-gpu`
GPU-enabled environment for GPU-accelerated analyses, extending `cellquorum-core` with CUDA-enabled PyTorch and scvi-tools.

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

**Install:**
```bash
micromamba create -n hdwgcna_env -c conda-forge -c bioconda \
  r-seurat r-hdwgcna r-wgcna bioconductor-zellkonverter
```

## Locking Environments

Use `make lock` to generate conda-lock files from the environment YAML recipes:

```bash
make lock
```

This scans `envs/*.yml` and generates corresponding `envs/*.conda-lock.yml` files. Requires `conda-lock` on PATH:

```bash
pip install conda-lock
```

Locks are solved for **linux-64 only** — the platform the Docker image builds
on. This is deliberate: the GPU env carries linux-only CUDA packages
(`pytorch-cuda`, the `nvidia` channel) with no osx-64/win-64 build, so an
unpinned multi-platform solve would fail on those envs. Run `make lock` on a
linux-64 host (or CI) with `conda-lock` installed; the solve needs network
access to the conda channels.

## Docker Integration

All environments (primary + isolated backends) are baked into the `cellquorum` Docker image. See `docs/docker.md` for details on building and using the containerized environments.
