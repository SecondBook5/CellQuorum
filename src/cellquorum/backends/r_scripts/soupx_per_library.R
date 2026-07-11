# src/cellquorum/backends/r_scripts/soupx_per_library.R
# SoupX ambient-RNA correction for ONE 10x library.
# Ported from the validated le_kc scripts/31_soupx_per_library.R.
# Usage: Rscript soupx_per_library.R <raw_h5> <filtered_h5> <out_dir> <resolution> <round_to_int>
# Reads raw + filtered CellRanger h5, quick-clusters filtered cells for autoEstCont,
# estimates contamination (rho), adjusts counts, writes corrected mtx. Prints RHO=<v>.
suppressPackageStartupMessages({
  library(SoupX); library(Seurat); library(Matrix)
})

# Minimal 10x-style writer (avoids the DropletUtils dependency).
write_mtx <- function(mat, dir) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  Matrix::writeMM(mat, file.path(dir, "matrix.mtx"))
  writeLines(rownames(mat), file.path(dir, "features.tsv"))
  writeLines(colnames(mat), file.path(dir, "barcodes.tsv"))
  for (f in c("matrix.mtx", "features.tsv", "barcodes.tsv"))
    system2("gzip", c("-f", shQuote(file.path(dir, f))))
}

args <- commandArgs(trailingOnly = TRUE)
raw_h5 <- args[1]; filt_h5 <- args[2]; out_dir <- args[3]
resolution <- as.numeric(args[4]); round_to_int <- as.logical(args[5])

raw <- Seurat::Read10X_h5(raw_h5)
filt <- Seurat::Read10X_h5(filt_h5)
if (is.list(raw)) raw <- raw[["Gene Expression"]]
if (is.list(filt)) filt <- filt[["Gene Expression"]]

# Quick clusters on filtered cells for autoEstCont.
so <- CreateSeuratObject(filt)
so <- NormalizeData(so, verbose = FALSE)
so <- FindVariableFeatures(so, verbose = FALSE)
so <- ScaleData(so, verbose = FALSE)
so <- RunPCA(so, npcs = 30, verbose = FALSE)
so <- FindNeighbors(so, dims = 1:30, verbose = FALSE)
so <- FindClusters(so, resolution = resolution, verbose = FALSE)

sc <- SoupChannel(raw, filt)
sc <- setClusters(sc, setNames(as.character(Idents(so)), colnames(so)))
sc <- autoEstCont(sc, doPlot = FALSE)
rho <- mean(sc$metaData$rho)
adj <- adjustCounts(sc, roundToInt = round_to_int)
write_mtx(adj, out_dir)
cat(sprintf("RHO=%.6f\n", rho))
