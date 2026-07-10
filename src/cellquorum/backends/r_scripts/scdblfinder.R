# Read a genes-x-cells Matrix Market file, run scDblFinder, write per-cell scores.
# Usage: Rscript scdblfinder.R <counts.mtx> <out.csv> <seed>
suppressPackageStartupMessages({
  library(Matrix); library(scDblFinder); library(SingleCellExperiment)
})
args <- commandArgs(trailingOnly = TRUE)
mtx_path <- args[1]; out_path <- args[2]; seed <- as.integer(args[3])
set.seed(seed)
counts <- as(Matrix::readMM(mtx_path), "CsparseMatrix")   # genes x cells
sce <- SingleCellExperiment(assays = list(counts = counts))
sce <- scDblFinder(sce)
scores <- sce$scDblFinder.score
write.csv(data.frame(score = scores), out_path, row.names = FALSE)
