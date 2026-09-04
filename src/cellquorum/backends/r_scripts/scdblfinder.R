# Read a genes-x-cells Matrix Market file, run scDblFinder, write per-cell
# score AND scDblFinder's own singlet/doublet class call.
# Usage: Rscript scdblfinder.R <counts.mtx> <out.csv> <seed> [samples.csv] [threads]
#
# The optional 4th argument is a one-column CSV (header `sample`) with one row
# per cell, in the same order as the matrix columns. When given it is passed to
# scDblFinder's own `samples=` argument, which searches for doublets
# independently within each capture -- the correct treatment, since a doublet
# cannot form across two libraries. Doing it here rather than by calling this
# script once per sample matters for wall time and nothing else: R startup plus
# library(scDblFinder) costs ~5s, which on a cohort of 18 samples was 95s of the
# QC stage spent loading the same packages 18 times.
suppressPackageStartupMessages({
  library(Matrix); library(scDblFinder); library(SingleCellExperiment)
})
args <- commandArgs(trailingOnly = TRUE)
mtx_path <- args[1]; out_path <- args[2]; seed <- as.integer(args[3])
samples_path <- if (length(args) >= 4 && nzchar(args[4])) args[4] else NULL
threads <- if (length(args) >= 5 && nzchar(args[5])) as.integer(args[5]) else 1L
set.seed(seed)
counts <- as(Matrix::readMM(mtx_path), "CsparseMatrix")   # genes x cells
sce <- SingleCellExperiment(assays = list(counts = counts))
# Name the cells by their input position. With `samples=` scDblFinder splits the
# object, scores each part and rebinds, so the returned column order is its
# business, not ours -- and a silently permuted score vector would attach one
# donor's doublet calls to another's cells. Restored by name below.
colnames(sce) <- as.character(seq_len(ncol(sce)))
n_cells_in <- ncol(sce)

samples <- NULL
if (!is.null(samples_path)) {
  samples <- as.character(read.csv(samples_path)$sample)
  if (length(samples) != ncol(sce)) {
    stop(sprintf(
      "samples file has %d rows but the matrix has %d cells",
      length(samples), ncol(sce)
    ))
  }
}

# multiSampleMode is left at its default "split": doublets are searched for
# independently per sample, which is what calling this script per sample did.
#
# RNGseed is set on BOTH branches, and that is the point of writing it this way.
# BiocParallel with RNGseed gives each TASK -- here, each capture -- its own
# L'Ecuyer stream keyed to the task, not to the worker that ran it, so the scores
# come out the same at any thread count. Left to the defaults instead, serial and
# parallel draw from different streams and the doublet calls quietly depend on how
# many cores the machine happened to have. Verified on the 2,125-cell LEC arm:
# bit-identical scores at threads 1, 4 and 8.
bpparam <- if (threads > 1L) {
  BiocParallel::MulticoreParam(threads, RNGseed = seed)
} else {
  BiocParallel::SerialParam(RNGseed = seed)
}
sce <- scDblFinder(sce, samples = samples, BPPARAM = bpparam)

# Back to input order, and refuse to write a short table: the Python adapter
# assigns these rows to cells positionally.
if (ncol(sce) != n_cells_in) {
  stop(sprintf("scDblFinder returned %d of %d cells", ncol(sce), n_cells_in))
}
sce <- sce[, order(as.integer(colnames(sce)))]

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
