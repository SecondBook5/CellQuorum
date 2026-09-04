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

# Create the output directory up front, before ANY write. write_mtx() also creates it,
# but it runs after rho_per_cell.csv is written, so relying on it made the first write
# fail with "cannot open the connection" on a fresh out_dir. Creating it here keeps the
# guarantee independent of the order the outputs happen to be written in.
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

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

# SoupX estimates rho PER CELL. Collapsing it to a mean here (which is all this
# script used to keep) discards the distribution, and a single scalar per library
# cannot show whether the correction removed the RIGHT counts — only that it
# removed some. Persist the per-cell values so downstream QC can plot them.
rho_per_cell <- sc$metaData$rho
names(rho_per_cell) <- rownames(sc$metaData)
utils::write.csv(
  data.frame(barcode = names(rho_per_cell), rho = as.numeric(rho_per_cell)),
  file.path(out_dir, "rho_per_cell.csv"),
  row.names = FALSE
)
rho <- mean(rho_per_cell)

adj <- adjustCounts(sc, roundToInt = round_to_int)
write_mtx(adj, out_dir)

# Per-gene soup fraction: how much of each gene's total counts the correction
# removed. This is what identifies WHICH genes were ambient — the input to a
# before/after marker-specificity check.
removed <- Matrix::rowSums(filt[rownames(adj), colnames(adj), drop = FALSE]) -
  Matrix::rowSums(adj)
observed <- Matrix::rowSums(filt[rownames(adj), colnames(adj), drop = FALSE])
utils::write.csv(
  data.frame(
    gene = rownames(adj),
    counts_observed = as.numeric(observed),
    counts_removed = as.numeric(removed),
    frac_removed = as.numeric(ifelse(observed > 0, removed / observed, 0))
  ),
  file.path(out_dir, "soup_fraction_per_gene.csv"),
  row.names = FALSE
)

cat(sprintf("RHO=%.6f\n", rho))
