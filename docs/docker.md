# Docker Image Build and Usage

The CellQuorum Docker image packages the full analysis stack (cellquorum-core + all isolated backend environments) into a reproducible container. This document covers building, validating, and running analyses inside the image.

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

By default, the Dockerfile bakes only the scclr toolchain (Rust, maturin, Python 3.13) into the `scclr` environment, but does NOT install scclr itself. To bake scclr from source:

```bash
docker build --target cpu -t cellquorum:<version> \
  --build-arg SCCLR_SRC=/path/to/scclr \
  -f docker/Dockerfile .
```

This copies the scclr source tree into the image and installs it via `pip install -e` inside the `scclr` environment.

## Validating the Image

### Smoke Test

Run the three-part smoke test to verify the image is functional:

```bash
make smoke
```

This performs:

1. **Version check:** Confirms `cellquorum --version` runs.
2. **Plan dry-run:** Confirms `cellquorum plan --config docker/smoke/smoke.yaml --json` runs without error on the bundled smoke config.
3. **Environment inventory:** Confirms all seven required environments are present in the image:
   - `cellquorum-core`
   - `cellquorum-r`
   - `celloracle_env`
   - `pyscenic_env`
   - `hdwgcna_env`
   - `scclr`
   - `sccoda_env`

If the smoke test passes, the image is ready for use.

## Running Analyses in the Image

### Interactive Shell

Enter an interactive shell inside the image:

```bash
docker run --rm -it cellquorum:<version> /bin/bash
```

Once inside, the `cellquorum` CLI is available:

```bash
cellquorum --version
cellquorum plan --help
```

### One-off Commands

Run a single cellquorum command:

```bash
docker run --rm cellquorum:<version> cellquorum --version
```

### Mounting Data

To analyze data on the host, mount the data directory into the container:

```bash
docker run --rm \
  -v /path/to/data:/data \
  cellquorum:<version> \
  cellquorum run --config /data/my_config.yaml --output /data/results
```

### Using Snakemake Workflow

Run the multi-hypothesis workflow matrix (see `docs/snakemake.md` for details):

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

- All isolated backend environments are pre-installed and available inside the image. CellQuorum will automatically invoke them via `micromamba run -n <env>` as needed.
- The GPU image requires a host with NVIDIA drivers and `nvidia-docker` runtime configured.
- See `envs/README.md` for details on the environment architecture and `docs/snakemake.md` for the Snakemake workflow structure.
