# Speckle propeller proportion test for cell-type differential abundance.
# Usage: Rscript propeller.R <counts.csv> <meta.csv> <out.csv> \
#        <condition_col> <case> <control> <transform>
# counts.csv: first column 'sample', remaining columns cell types (samples x celltypes, integer counts).
# meta.csv:   first column sample id, columns include condition_col.
# transform:  "asin" (default) or "logit" for transformed proportions.
suppressPackageStartupMessages({ library(speckle); library(limma) })

args <- commandArgs(trailingOnly = TRUE)
counts_csv <- args[1]; meta_csv <- args[2]; out_csv <- args[3]
condition_col <- args[4]; case <- args[5]; control <- args[6]; transform <- args[7]

# Read aggregated counts (samples x celltypes).
counts <- read.csv(counts_csv, check.names = FALSE, stringsAsFactors = FALSE)
rownames(counts) <- counts[["sample"]]; counts[["sample"]] <- NULL
counts <- as.matrix(counts)
storage.mode(counts) <- "integer"

# Read sample metadata and align to count rows.
meta <- read.csv(meta_csv, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
meta <- meta[rownames(counts), , drop = FALSE]

# Restrict to the two condition levels.
keep <- meta[[condition_col]] %in% c(case, control)
counts <- counts[keep, , drop = FALSE]
meta <- meta[keep, , drop = FALSE]

# getTransformedProps expects long-form per-cell vectors.
# Reconstruct from aggregated counts by expanding each (sample, celltype) count into that many rows.
clusters_vec <- c()
sample_vec <- c()
for (i in seq_len(nrow(counts))) {
  sample_id <- rownames(counts)[i]
  for (j in seq_len(ncol(counts))) {
    celltype <- colnames(counts)[j]
    count <- counts[i, j]
    if (count > 0) {
      clusters_vec <- c(clusters_vec, rep(celltype, count))
      sample_vec <- c(sample_vec, rep(sample_id, count))
    }
  }
}

# Get transformed proportions (returns celltypes x samples matrices).
tp <- getTransformedProps(clusters = clusters_vec, sample = sample_vec, transform = transform)

# Build group factor from the column order of tp$Proportions (map each sample → condition).
sample_ids <- colnames(tp$Proportions)
grp <- factor(meta[sample_ids, condition_col], levels = c(control, case))

# Run propeller.ttest with the transformed proportions and design matrix.
design <- model.matrix(~ grp)
res <- propeller.ttest(
  prop.list = tp,
  design = design,
  contrasts = c(0, 1),
  robust = TRUE,
  trend = FALSE,
  sort = TRUE
)

# Emit CSV with the contract columns: cell_type, PropRatio, Tstatistic, PValue, FDR.
# Note: speckle returns P.Value (with dot); rename to PValue.
out <- data.frame(
  cell_type = rownames(res),
  PropRatio = res$PropRatio,
  Tstatistic = res$Tstatistic,
  PValue = res$P.Value,
  FDR = res$FDR,
  row.names = NULL
)
write.csv(out, out_csv, row.names = FALSE)
