# CellQuorum Real Data Smoke Test

This document describes how to run smoke tests on real biological h5ad files to verify that CellQuorum's preprocessing pipeline works correctly.

## Quick Smoke Test

The fastest way to verify the pipeline is to use the `smoke_real_h5ad.py` script:

```bash
cd /home/ajbook/projects/cellquorum

mamba run -n cellquorum-dev python scripts/smoke_real_h5ad.py \
  --input-h5ad /mnt/e/mg_thymoma_tolerance_data/raw/mg_yasumizu_figshare/processed_scRNAseq_all_rmdoublet.h5ad \
  --output-dir /mnt/e/cellquorum_smoke_runs/mg_thymoma_processed_smoke_script \
  --n-cells 200 \
  --n-genes 500 \
  --counts-layer counts \
  --recipe cellquorum_log1p_cp10k_v1 \
  --overwrite-output
```

Expected output (JSON summary):
```json
{
  "status": "success",
  "input_h5ad": "/mnt/e/mg_thymoma_tolerance_data/raw/mg_yasumizu_figshare/processed_scRNAseq_all_rmdoublet.h5ad",
  "input_shape": [113948, 27631],
  "subset_shape": [200, 500],
  "successful_stages": ["qc", "preprocessing"],
  "failed_stages": [],
  "has_qc_annotations": true,
  "has_normalized_layer": true,
  "has_counts_layer": true,
  "recipe": "cellquorum_log1p_cp10k_v1"
}
```

## What the Script Does

1. **Loads** the input h5ad file
2. **Subsets** to the specified number of cells and genes (deterministically: first N cells, first N genes)
3. **Optionally sets X** to a specified counts layer if `--counts-layer` is provided
4. **Writes** the subset as `_subset_input.h5ad` in the output directory
5. **Runs** CellQuorum with:
   - QC in report-only mode (minimal thresholds)
   - Preprocessing normalization with the specified recipe
6. **Validates** that:
   - QC and preprocessing stages succeeded
   - No stages failed
   - `counts` layer is preserved
   - `cellquorum_normalized` layer is created
   - Artifacts are written (preprocessing_summary.json, stage_execution_records.json)
7. **Prints** a JSON summary to stdout

## Script Options

```
--input-h5ad PATH          Input h5ad file path (required)
--output-dir PATH          Output directory for smoke test (required)
--n-cells INT              Number of cells to subset (default: 200)
--n-genes INT              Number of genes to subset (default: 500)
--counts-layer TEXT        Layer to use as counts (optional, sets X to this layer)
--recipe TEXT              Normalization recipe (default: cellquorum_log1p_cp10k_v1)
--overwrite-output         Overwrite existing output directory
```

## Normalization Recipes

CellQuorum supports multiple normalization recipes:

- `none` - Pass-through (no normalization)
- `cellquorum_pf_v1` - Proportional fractions: `x / depth`
- `cellquorum_log1p_cp10k_v1` - Log1p counts per 10k: `log1p((x / depth) * 10000)` **(recommended for smoke tests)**
- `cellquorum_log1p_pf_v1` - Log1p proportional fractions: `log1p(x / depth)`
- `cellquorum_pf_log1p_pf_v1` - Shifted CLR-like (default, may densify sparse matrices)

## Expected Test Dataset Properties

The smoke test works best with h5ad files that have:

- **Raw integer counts** in X or a layer
- **Reasonable size** (subset should have at least 100 cells × 100 genes)
- **Non-zero data** (avoid all-zero subsets)
- **Valid var_names** (gene names should be unique or uniqueified automatically)

## Troubleshooting

### "Layer 'counts' not found"

If the h5ad doesn't have a `counts` layer, either:
- Omit `--counts-layer` to use X directly
- Specify a different layer name that exists in the file

### "QC stage did not succeed"

Check that:
- The subset has non-zero cells
- The matrix contains numeric data
- There are no NaN or Inf values

### "Preprocessing stage did not succeed"

Check that:
- Input counts are non-negative
- The recipe name is valid
- No cells have zero total counts (or use a recipe that handles them)

## Running Tests

The smoke script has its own test suite:

```bash
mamba run -n cellquorum-dev pytest tests/test_smoke_real_h5ad_script.py -v
```

This creates tiny synthetic h5ad files and verifies the script works correctly.

## Integration with CI/CD

The smoke test can be integrated into CI/CD pipelines:

```bash
# Run smoke test and capture exit code
if mamba run -n cellquorum-dev python scripts/smoke_real_h5ad.py \
    --input-h5ad "$INPUT_H5AD" \
    --output-dir "$OUTPUT_DIR" \
    --overwrite-output; then
  echo "Smoke test passed"
else
  echo "Smoke test failed"
  exit 1
fi
```

## Manual Verification

After running the smoke script, you can manually inspect the outputs:

```bash
# View preprocessing summary
cat $OUTPUT_DIR/run/results/preprocessing/preprocessing_summary.json | jq

# View QC summary
cat $OUTPUT_DIR/run/results/qc/qc_summary.json | jq

# View stage execution records
cat $OUTPUT_DIR/run/provenance/stage_execution_records.json | jq

# Load the final AnnData in Python
python -c "
import anndata as ad
adata = ad.read_h5ad('$OUTPUT_DIR/_subset_input.h5ad')
print('Input shape:', adata.shape)
print('Layers:', list(adata.layers.keys()))
"
```

## Real Dataset Example

The MG Thymoma dataset used in the example above:
- **Source**: Yasumizu et al. Figshare
- **Size**: 113,948 cells × 27,631 genes
- **Data type**: scRNA-seq after doublet removal
- **Layers**: `counts` (raw integer counts)
- **Location**: `/mnt/e/mg_thymoma_tolerance_data/raw/mg_yasumizu_figshare/processed_scRNAseq_all_rmdoublet.h5ad`

This is a good smoke test dataset because:
- It has a large cell count (robust subsetting)
- It has a preserved counts layer
- It represents real biological data
- It has been preprocessed to remove doublets
