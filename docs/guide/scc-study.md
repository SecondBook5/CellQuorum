# Paired carcinoma signaling and regulation

Start with gene counts and the published population labels from 9,916 cells across ten patients. The study compares each patient's tumor and normal skin. LIANA ranks expression-based interactions by library, Tensor-cell2cell models their variation, NicheNet links ligands to receiver targets, and MultiNicheNet prioritizes condition-dependent interactions. CollecTRI and pySCENIC supply two different estimates of transcription-factor activity. [Paired SCC reference analysis](https://github.com/saeyslab/multinichenetr/blob/main/vignettes/paired_analysis_SCC.knit.md)

## Prepare the environments

Use Linux or WSL, Python 3.12 and an R environment on PATH. From the checkout, set up the isolated pySCENIC backend and install CellQuorum into the active analysis environment. The pinned direct dependencies below reproduce the versions used for this benchmark. The core and GPU environment locks are under `envs/`.

```bash
micromamba create -y -f envs/pyscenic_env.yml
python -m pip install '.[signaling]' -c envs/scc-signaling-constraints.txt
Rscript -e 'stopifnot(requireNamespace("edgeR"), requireNamespace("DropletUtils"), requireNamespace("SingleCellExperiment"), requireNamespace("nichenetr"), requireNamespace("multinichenetr"))'
```

The reference run used edgeR 4.6.3, DropletUtils 1.28.1 and SingleCellExperiment 1.30.1. These are Bioconductor packages. Install the two signaling packages with their reference revisions in the same R library. Ensure `micromamba` is on PATH in the shell running CellQuorum.

```r
remotes::install_github("saeyslab/nichenetr@66f90d5eeafef280b2b2f339b3fd70ffec1781dd")
remotes::install_github("saeyslab/multinichenetr@f1db92cf1ef72fee6edfa570d0bd6055d943a064")
```

## Prepare the study and run

The installed wheel includes the recipes and preparation scripts. Run the following commands from an empty study directory. The export script checks the published RDS checksum, writes one H5 count matrix per library, and matches cell metadata by barcode. The resource helper checks all six NicheNet and cisTarget files against the SHA256 digests used here. CollecTRI also needs access to the decoupler prior service or a populated cache.

```bash
recipes="$(python -c 'import sysconfig; print(sysconfig.get_path("data"))')/share/cellquorum/workflows"
curl -fL https://zenodo.org/records/8010790/files/sce_subset_scc.rds -o sce_subset_scc.rds
Rscript "$recipes/helpers/prepare_scc.R" sce_subset_scc.rds libraries
cellquorum import-counts libraries/samples.csv scc.h5ad
export CELLQUORUM_RESOURCES="$PWD/resources"
python "$recipes/helpers/fetch_scc_resources.py" "$CELLQUORUM_RESOURCES"
cellquorum plan -c "$recipes/scc_paired_signaling.yaml" --json > plan.json
cellquorum run -c "$recipes/scc_paired_signaling.yaml"
python "$recipes/helpers/verify_reference_study.py" runs/scc_paired_signaling --data .
```

Open `runs/scc_paired_signaling/reports/report.html`. The manuscript gallery links to the assembled figures, individual editable SVGs, exact plotted values and legends. Its JSON index links each panel to the original stage tables and resolved configuration. To regenerate figures from an existing analysis, run `cellquorum publication runs/scc_paired_signaling`.

## Read the figures

- The SCC input is an annotated, filtered public subset. All four published populations are available to LIANA and CollecTRI. LIANA scores 18 of 20 libraries. The three-factor tensor therefore contains 18 contexts. The reference uses one regular optimization start. Factors and their numbering can change across fits.
- MultiNicheNet compares fibroblasts and KC_other keratinocytes. Requiring ten cells of each type in both samples leaves P2, P6, P8, P9 and P10. Open `results/mnn_pairing_audit.csv` to see cell counts and exclusions. The all-four-population comparison has one eligible patient.
- The paired edgeR model supplies fibroblast differential expression for the NicheNet receiver analysis. The communication figure follows the same sender, receiver, ligand and receptor across rankings and paired donor scores. The target panel shows prior weights. Use the full tables to inspect other edges.
- pySCENIC fits a seeded 1,000-cell subsample and scored 274 regulons in the first run. CollecTRI scores the larger cohort. The regulatory figure selects a shared TF by variation of donor- and condition-level CollecTRI means and shows how many patients support each panel. Treat these scores as inferred regulation.
- The verifier checks the positive fibroblast MMP1 response and four overlapping sample-level CD63 values against the published vignette. The first run recovered an MMP1 log2 fold change of 11.59 and absolute differences of 0.14–0.18 for CD63. Its 35 minutes of wall time and 7.84 GiB peak resident memory describe this reference configuration on the validation workstation.
