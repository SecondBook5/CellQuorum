# Docker Image Build and Usage

The CPU target contains the core and R environments. The GPU target adds the GPU
environment. Optional subprocess backends use separate build targets; neither
default image contains every backend. Environment recipes are not yet fully locked.

## Building the Image

### CPU Image

Build the CPU-only image:

```bash
make image
```

This runs:
```bash
docker build --target cpu -t cellquorum:<version> -f docker/Dockerfile .
```

The version is read from `cellquorum.__version__`.

### GPU Image

The GPU recipe is undergoing compatibility validation; see the
[deployment status](../design/deployment-goal.md). A successful image build alone does not
prove that GPU kernels or complete analyses run.

Build the GPU-enabled image (includes CUDA-enabled PyTorch and scvi-tools):

```bash
make image-gpu
```

This runs:
```bash
docker build --target gpu -t cellquorum:<version>-gpu -f docker/Dockerfile .
```

### Build Arguments

#### `SCCLR_SRC`

The optional `backends` target creates the scclr toolchain environment. It installs
scclr only when `SCCLR_SRC` names a source location available inside that build:

```bash
docker build --target backends -t cellquorum:<version>-backends \
  --build-arg SCCLR_SRC=/opt/cellquorum/vendor/scclr \
  -f docker/Dockerfile .
```

For this example, place the source in `vendor/scclr` in the build context and
ensure it is not excluded by `.dockerignore`. The argument itself does not copy a
host path. The Dockerfile uses `pip install` in the scclr environment. The
`backends` target has an unresolved hdWGCNA solve issue and is not validated for
deployment.

## Validating the Image

### Smoke Test

Run the three-part smoke test to verify the image is functional:

```bash
make smoke
```

This performs:

1. **Version check:** Confirms `cellquorum --version` runs.
2. **Plan dry-run:** Confirms `cellquorum plan --config docker/smoke/smoke.yaml --json` runs without error on the bundled smoke config.
3. **Environment inventory:** Confirms the CPU target's required environments are present:
   - `cellquorum-core`
   - `cellquorum-r`

This checks startup and planning. Installed analysis, checkpoint recovery, and
backend execution still need separate validation.

### GPU device checks

After building the GPU target, run `make smoke-gpu`. It checks the three required
environments, runs the CLI in `cellquorum-gpu`, and probes both RAPIDS and
PyTorch/scVI with real device operations. Missing capabilities fail the command.
This target requires a Docker installation with NVIDIA GPU access; on Windows,
Docker Desktop must also expose its engine to the WSL distribution.

For a GPU analysis, pass GPU access explicitly:

```bash
docker run --rm --gpus all \
  -v /path/to/data:/data \
  cellquorum:<version>-gpu \
  run --config /data/my_config.yaml --output-dir /data/results
```

The analysis configuration must select the GPU backend for supported methods.
Device probes do not replace installed-workflow acceptance or memory benchmarks.

## Running Analyses in the Image

### Interactive Shell

Enter an interactive shell inside the image:

```bash
docker run --rm -it --entrypoint micromamba cellquorum:<version> run -n cellquorum-core /bin/bash
```

Once inside, the `cellquorum` CLI is available:

```bash
cellquorum --version
cellquorum plan --help
```

### One-off Commands

Run a single cellquorum command:

```bash
docker run --rm cellquorum:<version> --version
```

### Mounting Data

To analyze data on the host, mount the data directory into the container:

```bash
docker run --rm \
  -v /path/to/data:/data \
  cellquorum:<version> \
  run --config /data/my_config.yaml --output-dir /data/results
```

### Running R-backed methods (pseudobulk DE, Milo, propeller, NicheNet, SoupX)

The image runs the CLI inside `cellquorum-core` (CPU image) or `cellquorum-gpu`
(GPU image). R/Bioconductor lives in the **separate** `cellquorum-r` environment,
and CellQuorum invokes R by running `Rscript` as a direct subprocess — *not* via
`micromamba run -n cellquorum-r`. Because `Rscript` is therefore not on the CLI
env's `PATH`, R-backed methods would otherwise record a clean
`Rscript unavailable` skip.

To make them run, point `r.rscript_path` at the `cellquorum-r` env's Rscript in
your config:

```yaml
r:
  enabled: true
  rscript_path: /opt/conda/envs/cellquorum-r/bin/Rscript
```

`/opt/conda` is the `MAMBA_ROOT_PREFIX` of the `mambaorg/micromamba` base image;
confirm the exact path for your build with:

```bash
docker run --rm --entrypoint micromamba cellquorum:<version> run -n cellquorum-r which Rscript
```

Invoking that env's `Rscript` binary directly loads the `cellquorum-r` R library,
so edgeR/limma/DESeq2/Milo/propeller resolve without any further activation. The
engine honors this configured path (it checks the *configured* Rscript, not a bare
`Rscript` on `PATH`); a missing or wrong path still yields a recorded skip rather
than a crash.

### Using Snakemake Workflow

Run the multi-hypothesis workflow matrix (see `docs/guide/snakemake.md` for details):

```bash
make matrix
```

Or equivalently:

```bash
docker run --rm \
  -v $(PWD):/work -w /work \
  --entrypoint micromamba \
  cellquorum:<version> \
  run -n cellquorum-core snakemake --snakefile workflow/Snakefile --cores 4 --keep-going
```

## Notes

- All isolated backend environments are pre-installed and available inside the image. CellQuorum will automatically invoke the isolated *Python* backends (pySCENIC, hdWGCNA, scCODA, scclr, CellOracle) via `micromamba run -n <env>` as needed. The `cellquorum-r` environment is the exception: R is reached as a direct `Rscript` subprocess, so set `r.rscript_path` as described in "Running R-backed methods" above.
- The GPU image requires a host with NVIDIA drivers and `nvidia-docker` runtime configured.
- See `envs/README.md` for details on the environment architecture and `docs/guide/snakemake.md` for the Snakemake workflow structure.
