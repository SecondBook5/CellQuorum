# Milo neighborhood-level differential-abundance test.
# Usage: Rscript milo.R <rep.csv> <meta.csv> <out.csv> \
#        <condition_col> <case> <control> <donor_col> <k> <prop> [celltype_col]
# rep.csv:  first column 'cell' id, remaining columns reduced-dim embedding (cells x dims).
# meta.csv: first column cell id, columns include condition_col, donor_col, optional cell_type.
# out.csv:  nhood, logFC, PValue, SpatialFDR, nhood_size, majority_celltype, celltype_fraction.
suppressPackageStartupMessages({ library(miloR); library(SingleCellExperiment); library(edgeR) })

args <- commandArgs(trailingOnly = TRUE)
rep_csv <- args[1]; meta_csv <- args[2]; out_csv <- args[3]
condition_col <- args[4]; case <- args[5]; control <- args[6]; donor_col <- args[7]
k <- as.integer(args[8]); prop <- as.numeric(args[9])
celltype_col <- if (length(args) >= 10 && nzchar(args[10])) args[10] else NA

# Read embedding (cells x dims) and metadata (cells x columns).
emb <- read.csv(rep_csv, row.names = 1, check.names = FALSE)          # cells x dims
meta <- read.csv(meta_csv, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
meta <- meta[rownames(emb), , drop = FALSE]

# Restrict to case/control levels.
keep <- meta[[condition_col]] %in% c(case, control)
emb <- emb[keep, , drop = FALSE]
meta <- meta[keep, , drop = FALSE]

# Build SingleCellExperiment with placeholder assay and reduced-dim embedding.
# Attach cell_type to colData if provided so annotateNhoods can read it.
n_cells <- nrow(emb)
assays_list <- list(logcounts = matrix(0, nrow = 1, ncol = n_cells))
colnames(assays_list$logcounts) <- rownames(emb)

if (!is.na(celltype_col)) {
  # Attach cell_type to colData so annotateNhoods can find it
  coldata <- meta[, celltype_col, drop = FALSE]
  sce <- SingleCellExperiment(assays = assays_list, colData = coldata)
} else {
  sce <- SingleCellExperiment(assays = assays_list)
}
reducedDim(sce, "PCA") <- as.matrix(emb)

# Build Milo object and construct neighborhood graph.
milo <- Milo(sce)
d <- ncol(emb)
milo <- buildGraph(milo, k = k, d = d, reduced.dim = "PCA")
milo <- makeNhoods(milo, prop = prop, k = k, d = d, refined = TRUE, reduced_dims = "PCA")

# Count cells per neighborhood, aggregated by donor (sample).
milo <- countCells(milo, samples = donor_col, meta.data = meta)

# Build per-sample design.df: one row per donor with its condition.
# Must reorder to match colnames(nhoodCounts(milo)).
sm <- unique(meta[, c(donor_col, condition_col), drop = FALSE])
rownames(sm) <- sm[[donor_col]]
sm[[condition_col]] <- factor(sm[[condition_col]], levels = c(control, case))
design_df <- sm[colnames(nhoodCounts(milo)), , drop = FALSE]

# Run testNhoods with the design formula and per-sample data.
res <- testNhoods(
  milo,
  design = as.formula(paste("~", condition_col)),
  design.df = design_df,
  reduced.dim = "PCA"
)

# Compute nhood_size from the nhoods matrix (testNhoods output does NOT include it).
nhood_sizes <- colSums(nhoods(milo))
res$nhood_size <- nhood_sizes[res$Nhood]

# Annotate neighborhoods with majority cell type if celltype_col is provided.
if (!is.na(celltype_col)) {
  # annotateNhoods returns the data.frame with added columns: <celltype_col> and <celltype_col>_fraction
  res <- annotateNhoods(milo, res, coldata_col = celltype_col)
  # Rename annotation columns to the contract names
  res$majority_celltype <- res[[celltype_col]]
  res$celltype_fraction <- res[[paste0(celltype_col, "_fraction")]]
} else {
  res$majority_celltype <- NA
  res$celltype_fraction <- NA
}

# Emit the output CSV with the contract columns.
out <- data.frame(
  nhood = res$Nhood,
  logFC = res$logFC,
  PValue = res$PValue,
  SpatialFDR = res$SpatialFDR,
  nhood_size = res$nhood_size,
  majority_celltype = res$majority_celltype,
  celltype_fraction = res$celltype_fraction,
  row.names = NULL
)
write.csv(out, out_csv, row.names = FALSE)
