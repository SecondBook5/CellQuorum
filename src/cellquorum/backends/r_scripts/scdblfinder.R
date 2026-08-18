# Read a genes-x-cells Matrix Market file, run scDblFinder, write per-cell
# score AND scDblFinder's own singlet/doublet class call.
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
# Emit both the score and the native class call. The Python adapter uses the
# class column (scDblFinder's calibrated threshold) as the doublet call, rather
# than re-thresholding the score at an arbitrary cut.
write.csv(
  data.frame(
    score = sce$scDblFinder.score,
    class = as.character(sce$scDblFinder.class)
  ),
  out_path,
  row.names = FALSE
)
