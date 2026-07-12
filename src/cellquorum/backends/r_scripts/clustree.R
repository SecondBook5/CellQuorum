# Clustree resolution tree plot for leiden grid search.
# Usage: Rscript clustree.R <in.h5ad> <out.png> <prefix>
suppressPackageStartupMessages({
  library(zellkonverter)
  library(clustree)
  library(SingleCellExperiment)
  library(ggplot2)
})

# Parse command-line arguments.
args <- commandArgs(trailingOnly = TRUE)
in_h5ad <- args[1]
out_png <- args[2]
prefix <- args[3]

# Wrap in tryCatch for fail-loud behavior.
tryCatch(
  {
    # Read h5ad → SingleCellExperiment.
    sce <- zellkonverter::readH5AD(in_h5ad)

    # Find cluster columns matching prefix (e.g., "leiden_0.1", "leiden_0.5").
    # clustree needs at least 2 cluster columns with prefix + numeric suffix.
    cluster_cols <- colnames(colData(sce))
    matching_cols <- grep(paste0("^", prefix), cluster_cols, value = TRUE)

    if (length(matching_cols) < 2) {
      # Skip gracefully (clustree needs >=2 resolutions).
      cat("Clustree skipped: < 2 cluster columns with prefix '", prefix,
          "' found.\n", sep = "")
      # Write empty plot or skip file creation.
      quit(status = 0)
    }

    # Run clustree.
    p <- clustree::clustree(sce, prefix = prefix, node_colour = "sc3_stability")

    # Save plot.
    ggsave(out_png, plot = p, width = 10, height = 8, dpi = 300)

    cat("Clustree plot saved to:", out_png, "\n")
  },
  error = function(e) {
    cat("ERROR in clustree.R:", e$message, "\n", file = stderr())
    quit(status = 1)
  }
)
