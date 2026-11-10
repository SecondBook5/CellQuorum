# A complete paired PBMC study

Which cell types respond to interferon stimulation, which genes change within them,
and how consistent is the pathway response across cell types? This workflow starts
with the Kang PBMC count matrix and the authors’ eight cell-type labels. It fits a
paired comparison across eight donors. One run produces marker-based annotations,
population and QC audits, embeddings, state scores, differential expression and
abundance, enrichment, figures and an HTML report.

## Reproduce the run

Use Python 3.12 and R on Linux or WSL. From a CellQuorum checkout, install CellQuorum
and check the R packages used for expression and figures.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
Rscript -e 'stopifnot(requireNamespace("edgeR"), requireNamespace("ComplexHeatmap"), requireNamespace("circlize"))'
```

If the R check names a missing package, install it via Bioconductor.

```r
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("edgeR", "ComplexHeatmap", "circlize"))
```

Download the public count matrix and matching human Hallmark gene sets. Run the
commands from the same directory so that the input paths in the recipe resolve.

```bash
mkdir -p data
curl --fail --location https://scverse-exampledata.s3.eu-west-1.amazonaws.com/pertpy/kang_2018.h5ad -o data/kang_2018.h5ad
curl --fail --location https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/h.all.v2025.1.Hs.symbols.gmt -o data/h.all.v2025.1.Hs.symbols.gmt
sha256sum -c <<'SUMS'
e6a5adac64dcdeb36eaba27db49b63e0c64bb0ed4a64c6705971506b41c39830  data/kang_2018.h5ad
f22066af72e215ccb7b89d88e492c07e1eef17534c2ca7b0f9902cfecbbdd8e9  data/h.all.v2025.1.Hs.symbols.gmt
SUMS
cellquorum run -c configs/kang_paired_analysis.yaml
```

Open `runs/kang_paired_analysis/reports/report.html`. Its **Analysis outputs** section
links the figures and tables. The standard wheel includes the same recipe under
`share/cellquorum/workflows` in the installation prefix. Cite the original study and
the gene set collection when reusing this example. [Kang et al.](https://doi.org/10.1038/nbt.4042) · [MSigDB Hallmarks](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#H)

## Read the outputs

| Question | Figure under `figures/` | Supporting data |
|---|---|---|
| How are annotated cells distributed? | `embeddings/categorical_umap_cell_type.pdf` | The updated object in `objects/final_annotated.h5ad` includes the coordinates and labels. |
| Do the marker calls agree with the published labels? | `annotation/marker_dotplot.pdf` | `results/population_identity/tables/population_by_annotation.csv` gives the cell counts and fractions for every combination of published label and marker vote. The dot plot's means and detection percentages are in `results/annotation/`. |
| Does the interferon program shift across donors? | `state_scoring/state_score_violin_aucell_interferon_isg.pdf` | The violin plots show per-cell AUCell scores within each published type. Their marks use paired donor tests. `results/differential_state_results.csv` separately tests donor medians of Scanpy program scores across all PBMCs. |
| How do proportions vary across donors? | `differential_abundance/composition_per_patient_stacked.pdf` | `results/da_proportion_ttest.csv` reports the paired effect in percentage points, bootstrap interval, donor support and FDR. |
| Which genes respond within each population? | The cell-type volcano PDFs and `differential_expression/de_heatmap.pdf` | `results/de_pseudobulk_edger_by_subcluster.csv` gives cell-type-specific edgeR estimates. The heatmap displays mean normalized expression, scaled within each gene. |
| Which pathways respond across populations? | `enrichment/gsea_nes_heatmap_hallmark.pdf` and running enrichment curves | `results/enrichment_gsea_hallmark_by_cell_type.csv` gives normalized enrichment scores, adjusted p-values and leading-edge genes. |

The expression model aggregates raw counts by donor and condition and fits
`donor + condition` with `stim` compared to `ctrl`. The workflow also fits this
contrast within each supplied cell type. All eight cell types returned DE estimates.
Six qualified for pathway scoring at the 1,000-ranked-gene threshold. CD8 T cells
had 862 and megakaryocytes had 66. The figure labels these omissions. The
interferon-alpha response had positive enrichment at FDR < 0.05 in all six eligible
cell types. Cells marked `n.t.` denote untested combinations.

The matrix contains filtered PBMCs and lacks mitochondrial gene symbols. Its QC
outputs record that measurement as unavailable. The recipe uses existing cell-type
labels, 30 PCs for the embedding, and 1,000 GSEA permutations. Other studies need
their own design, annotation and QC decisions.

## What was selected

The recipe executes 18 of the 41 catalog stages. The original labels in `cell_type`
define the expression, abundance and enrichment comparisons. The annotation stage
keeps its initial Leiden vote in `marker_vote_label_cluster_vote` and its RNA-only
cell call in `marker_vote_label`. In T- or NK-labeled clusters, the opt-in
`annotation.lymphocyte_refinement: human_pbmc_v1` compares local T-lineage,
innate, cytotoxic, CD8 and helper-marker expression. It uses the log1p CP10k layer
and the RNA neighbors graph. Calls with weak or conflicting evidence are
`Unassigned`. The HTML report gives their counts and reasons and links the
barcode-level scores. Other cluster labels retain the original vote, including
its known limitations for myeloid and rare cells. See the [blinded PBMC benchmark](../benchmarks/pbmc-identity.md) for the independent CD8 and NK results.

Expand **Stages not selected and why** in the report to see each omission.
The filtered matrix contains no empty droplets or splice layers. Atlas mapping
requires a reference, and trajectory inference needs a specified lineage.
Other advanced analyses, including gene regulatory networks, de novo programs
and cell communication, have implementations and need their own end-to-end
validation.

## Verify a release or adapt the recipe

The recipe declares the outputs each analysis must produce. A full run fails if a
required file is absent, empty or missing from its producing stage’s artifact list.
For this known positive-control dataset, the declaration includes a heatmap of
significant expression changes. When adapting the recipe, use outputs that also
make sense for a null result. Per-method eligibility and skips remain in
`provenance/stage_execution_records.json`.

The opt-in acceptance test starts a fresh analysis and checks the paired design,
gene and cell-type coverage, the interferon positive control, figure files and
report links. With the test dependencies installed and the two inputs in `data/`,
run it from the checkout.

```bash
CELLQUORUM_PAIRED_STUDY_DIR="$PWD" python -m pytest tests/acceptance/test_paired_study.py
```
