# Backends & environments

CellQuorum runs methods across Python, R/Bioconductor, and GPU stacks. Because some
of these dependency stacks are mutually incompatible, the engine uses a **layered
environment strategy**: a core runtime plus dedicated environments for the heavyweight
and GPU backends. Backend availability is **detected at runtime** and reported in
provenance (`backend_status.json`/`.csv`); a backend that is absent causes the
affected method to skip with a recorded reason — it never crashes the run.

This page summarizes the environments and how dispatch works. For the exact install
commands, dependency-isolation rationales, and `make lock` usage, see
[`envs/README.md`](../envs/README.md).

## Primary environments

| Environment | Role |
|---|---|
| `cellquorum-core` | The main runtime (scanpy, anndata ≥ 0.11, scvi-tools, …). The `cellquorum` CLI is invoked from here. |
| `cellquorum-dev` | `cellquorum-core` plus testing/linting/build tooling (pytest, ruff, mypy, docs). |
| `cellquorum-gpu` | `cellquorum-core` plus CUDA PyTorch, scvi-tools, and the RAPIDS stack for GPU acceleration. |
| `cellquorum-r` | The R bridge: Seurat, zellkonverter, rpy2, anndata2ri for R/Bioconductor methods. |

## Isolated backend environments

Five backends live in isolated environments because their pins conflict with the
core stack. CellQuorum invokes them as subprocesses (`micromamba run -n <env> …`),
exchanging data through temporary files. **The environment names are hardcoded in
the dispatch code — create them with exactly these names.**

| Environment | Powers | Isolation rationale |
|---|---|---|
| `pyscenic_env` | `grn` — pySCENIC regulon inference (GRNBoost2 → cisTarget → AUCell) | pins `numpy=1.23.5`, `pandas=1.5.3`, `setuptools<81` |
| `hdwgcna_env` | `coexpression` — hdWGCNA co-expression modules | R Seurat/WGCNA stack |
| `sccoda_env` | `differential_abundance` — scCODA compositional testing | older scipy/tensorflow pins |
| `celloracle_env` | `perturbation` — CellOracle in-silico TF knockouts | dependency isolation for reproducibility |
| `scclr` | `preprocessing`/`dimensionality` — sparse PFlog1pPF normalization + sparse PCA | pins `anndata<0.11`; Python ≤ 3.13 |

pySCENIC also requires external cisTarget databases (TFs, motifs, rankings),
configured via `grn.tfs_path`, `grn.motifs_path`, and `grn.rankings_glob`.

## The R / Rscript bridge

R/Bioconductor methods run over the Rscript or rpy2 bridge (configured under the
`r:` config section). Six adapters share one abstraction,
`cellquorum.methods.r_method.RAnalysisMethod`, which centralizes Rscript resolution
and R-package availability checks:

| Method | Stage |
|---|---|
| edgeR pseudobulk (`PseudobulkEdgeRMethod`) | `differential_expression` |
| Milo (`MiloMethod`) | `differential_abundance` |
| propeller (`PropellerMethod`) | `differential_abundance` |
| NicheNet (`NicheNetMethod`) | `cell_cell_communication` |
| MultiNicheNet (`MultiNicheNetMethod`) | `cell_cell_communication` |
| DIALOGUE (`MulticellularProgramsMethod`) | `multicellular_programs` |
| scDiagnostics (`ScdiagnosticsMethod`) | `annotation_diagnostics` |

Two further R usages sit outside that abstraction: **SoupX** (`ambient_correction`)
runs a bundled R script (`cellquorum/backends/r_scripts/soupx_per_library.R`) through
the Rscript adapter, and **hdWGCNA** (`coexpression`) runs in the isolated
`hdwgcna_env` rather than the shared Rscript backend.

## Locking and Docker

- **Locking.** `make lock` generates `conda-lock` files from the `envs/*.yml`
  recipes for reproducible solves (requires `conda-lock` on `PATH`).
- **Docker.** A layered image bakes the primary environments and, on the optional
  `backends`/`celloracle` targets, the isolated backends. The CLI runs inside the
  correct environment (`cellquorum-core` for the CPU image, `cellquorum-gpu` for the
  GPU image), so the GPU and CCC/trajectory stages are reachable in-image. R is the
  one exception: R/Bioconductor lives in the separate `cellquorum-r` environment and
  is invoked as a direct `Rscript` subprocess, so `Rscript` is **not** on the CLI
  env's `PATH`. To use R-backed methods (pseudobulk edgeR DE, Milo, propeller,
  NicheNet, DIALOGUE, SoupX) in-image, point `r.rscript_path` at the `cellquorum-r` env's
  Rscript — see the R-methods section of [`docker.md`](docker.md).
