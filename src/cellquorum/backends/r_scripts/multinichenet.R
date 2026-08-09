# src/cellquorum/backends/r_scripts/multinichenet.R
# Tissue-wide differential CCC via multinichenetr. All settings via argv; no
# study-specific biology baked in. Reads mtx+csv, builds a SingleCellExperiment,
# runs multi_nichenet_analysis, writes group_prioritization_tbl to out_csv.
suppressPackageStartupMessages({
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
counts_mtx <- args[[1]]; genes_csv <- args[[2]]; barcodes_csv <- args[[3]]
obs_csv <- args[[4]]; out_csv <- args[[5]]
celltype_col <- args[[6]]; sample_col <- args[[7]]; group_col <- args[[8]]
case <- args[[9]]; control <- args[[10]]
ligand_target_rds <- args[[11]]; lr_network_rds <- args[[12]]
fraction_cutoff <- as.numeric(args[[13]]); min_sample_prop <- as.numeric(args[[14]])
logfc_threshold <- as.numeric(args[[15]]); p_val_threshold <- as.numeric(args[[16]])
p_val_adj <- as.logical(args[[17]]); top_n_target <- as.integer(args[[18]])
scenario <- args[[19]]; n_cores <- as.integer(args[[20]]); seed <- as.integer(args[[21]])

set.seed(seed)

# Fail fast (non-zero exit) when priors are absent -> Python turns this into MethodSkip.
stopifnot(file.exists(ligand_target_rds), file.exists(lr_network_rds))

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(multinichenetr)
})

counts <- as(Matrix::readMM(counts_mtx), "CsparseMatrix")  # genes x cells
genes <- read.csv(genes_csv, stringsAsFactors = FALSE)$gene
barcodes <- read.csv(barcodes_csv, stringsAsFactors = FALSE)$barcode
rownames(counts) <- genes; colnames(counts) <- barcodes
obs <- read.csv(obs_csv, stringsAsFactors = FALSE)
rownames(obs) <- obs$barcode
obs <- obs[barcodes, , drop = FALSE]

sce <- SingleCellExperiment(assays = list(counts = counts), colData = obs)
# multinichenetr expects a normalized "logcounts" assay.
logcounts(sce) <- log1p(counts)

ligand_target_matrix <- readRDS(ligand_target_rds)
lr_network <- readRDS(lr_network_rds)

contrasts_oi <- sprintf("'%s-%s','%s-%s'", case, control, control, case)
contrast_tbl <- data.frame(
  contrast = c(sprintf("%s-%s", case, control), sprintf("%s-%s", control, case)),
  group = c(case, control), stringsAsFactors = FALSE
)

output <- multi_nichenet_analysis(
  sce = sce,
  celltype_id = celltype_col, sample_id = sample_col, group_id = group_col,
  lr_network = lr_network, ligand_target_matrix = ligand_target_matrix,
  contrasts_oi = contrasts_oi, contrast_tbl = contrast_tbl,
  fraction_cutoff = fraction_cutoff, min_sample_prop = min_sample_prop,
  logFC_threshold = logfc_threshold, p_val_threshold = p_val_threshold,
  p_val_adj = p_val_adj, empirical_pval = FALSE,
  top_n_target = top_n_target, scenario = scenario, n.cores = n_cores,
  verbose = FALSE
)

tbl <- output$prioritization_tables$group_prioritization_tbl
write.csv(tbl, out_csv, row.names = FALSE)
