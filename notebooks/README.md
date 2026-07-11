# CellQuorum Validation Notebooks

## Prerequisites

To run the mast-cell validation notebook, you need:

1. **CellRanger data**: The 10x CellRanger outputs must be accessible at `/mnt/e/lymphedema_cellranger` (or adjust `paths.data_root` in the config).
2. **R + SoupX**: The SoupX R package must be installed and accessible via `Rscript`. Install with `install.packages("SoupX")` in R.
3. **Python environment**: Use `cellquorum-gpu` for GPU-accelerated Harmony integration, or the base CellQuorum environment for CPU-only execution. Install with `pip install -e .` from the repo root.

## Running the Notebook

Launch Jupyter from the repo root:

```bash
jupyter lab
```

Open `notebooks/mast_validation.ipynb` and run all cells.

## Validation Criterion

The notebook runs a 2-library smoke test and reports the **SoupX litmus test**: the fraction of mast cells with detected mast-cell markers (TPSAB1, CPA3, KIT) should remain high (~80-85%), while ECM contaminant genes (COL1A1, COL1A2, COL3A1, LUM, DCN) should drop low (~2-6%). This confirms that ambient RNA correction preserves biological signal while removing contamination.
