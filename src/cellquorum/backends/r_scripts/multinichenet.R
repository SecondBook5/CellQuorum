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
p_val_adj <- as.logical(args[[17]])
# multinichenetr checks is.double() on both, not is.numeric() -- an
# as.integer() value fails that check even though it's conceptually a
# number ("top_n_target should be numeric" / "n.cores should be numeric").
top_n_target <- as.numeric(args[[18]])
scenario <- args[[19]]; n_cores <- as.numeric(args[[20]]); seed <- as.integer(args[[21]])
# Optional: per-sample and pre-trim (celltype-specificity) tables --
# multi_nichenet_analysis() already computes both internally
# (prioritization_tables$sample_prioritization_tbl and
# $group_prioritization_table_source), but until now nothing extracted
# them, so the only output was the fully-aggregated group table -- no
# per-sample values, no ligand/receptor celltype-specificity scores.
out_sample_csv <- if (length(args) >= 22) args[[22]] else NA_character_
out_source_csv <- if (length(args) >= 23) args[[23]] else NA_character_

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

# multinichenetr requires cell type labels to be syntactically valid R names
# (they end up as list/column names internally) -- a bare numeric label like
# "0" is not ("All the cell type labels ... should be syntactically valid R
# names"). Sanitize here, and map the output back to the real labels below,
# so nothing downstream of this script has to know about the substitution.
#
# Forced to character first: read.csv parses a numeric-looking column (e.g.
# cluster ids "0".."8") as integer, and indexing a named vector by an
# integer does R's POSITIONAL lookup, not its NAME lookup -- silently wrong
# (and index 0 drops the row entirely, which is why this once produced
# 61775 rows instead of 68148).
obs[[celltype_col]] <- as.character(obs[[celltype_col]])
original_celltypes <- unique(obs[[celltype_col]])
sanitized_celltypes <- make.names(original_celltypes)
celltype_lookup <- setNames(original_celltypes, sanitized_celltypes)
obs[[celltype_col]] <- setNames(sanitized_celltypes, original_celltypes)[obs[[celltype_col]]]

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
  # No batch or covariate column in this design; NA is multinichenetr's own
  # documented value for "correct for nothing" (required, no default as of
  # multinichenetr 2.1.0 -- an older/different version this script was
  # originally written against apparently didn't require them).
  batches = NA, covariates = NA,
  lr_network = lr_network, ligand_target_matrix = ligand_target_matrix,
  contrasts_oi = contrasts_oi, contrast_tbl = contrast_tbl,
  fraction_cutoff = fraction_cutoff, min_sample_prop = min_sample_prop,
  logFC_threshold = logfc_threshold, p_val_threshold = p_val_threshold,
  p_val_adj = p_val_adj, empirical_pval = FALSE,
  top_n_target = top_n_target, scenario = scenario, n.cores = n_cores,
  verbose = FALSE
)

# Undo the make.names() sanitization: any column whose values are entirely
# drawn from the sanitized cell type labels (sender/receiver, in the current
# schema, but this stays correct if multinichenetr renames them) gets mapped
# back to the real, original labels. Shared across every output table, not
# just group_prioritization_tbl, since sample_prioritization_tbl and
# group_prioritization_table_source carry the same sanitized sender/receiver
# columns.
unsanitize <- function(tbl) {
  for (col in names(tbl)) {
    values <- as.character(tbl[[col]])
    if (all(values %in% names(celltype_lookup))) {
      tbl[[col]] <- unname(celltype_lookup[values])
    }
  }
  tbl
}

write.csv(unsanitize(output$prioritization_tables$group_prioritization_tbl), out_csv, row.names = FALSE)

if (!is.na(out_sample_csv)) {
  write.csv(
    unsanitize(output$prioritization_tables$sample_prioritization_tbl),
    out_sample_csv, row.names = FALSE
  )
}
if (!is.na(out_source_csv)) {
  write.csv(
    unsanitize(output$prioritization_tables$group_prioritization_table_source),
    out_source_csv, row.names = FALSE
  )
}
