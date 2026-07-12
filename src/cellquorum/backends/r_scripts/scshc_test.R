# sc-SHC formal significance test for supplied cluster labels.
# Usage: Rscript scshc_test.R <in.h5ad> <clusters.csv> <out.csv> <alpha> <batch_key_or_NONE>
suppressPackageStartupMessages({
  library(zellkonverter)
  library(scSHC)
  library(SingleCellExperiment)
  library(Matrix)
})

# Parse command-line arguments.
args <- commandArgs(trailingOnly = TRUE)
in_h5ad <- args[1]
clusters_csv <- args[2]
out_csv <- args[3]
alpha <- as.numeric(args[4])
batch_arg <- args[5]

# Wrap everything in tryCatch for fail-loud behavior.
tryCatch(
  {
    # Read h5ad → SingleCellExperiment.
    sce <- zellkonverter::readH5AD(in_h5ad)

    # Ensure counts assay exists (zellkonverter maps X → first assay).
    if (!"counts" %in% assayNames(sce)) {
      if (length(assayNames(sce)) > 0) {
        assayNames(sce)[1] <- "counts"
      } else {
        stop("Input h5ad has no assays to use as counts.")
      }
    }

    # Extract counts matrix (genes × cells).
    counts_matrix <- assay(sce, "counts")

    # Read cluster labels CSV (barcode, cluster).
    clusters_df <- read.csv(clusters_csv, stringsAsFactors = FALSE)

    # Ensure barcode column exists (case-insensitive).
    barcode_col <- NULL
    for (col in colnames(clusters_df)) {
      if (tolower(col) %in% c("barcode", "cell")) {
        barcode_col <- col
        break
      }
    }

    if (is.null(barcode_col)) {
      stop("Cluster CSV missing barcode column (expected 'barcode' or 'cell').")
    }

    # Ensure cluster column exists.
    if (!"cluster" %in% colnames(clusters_df)) {
      stop("Cluster CSV missing 'cluster' column.")
    }

    # Align cluster labels to SCE cell order.
    barcodes <- colnames(sce)
    clusters_df <- clusters_df[match(barcodes, clusters_df[[barcode_col]]), ]

    if (any(is.na(clusters_df$cluster))) {
      stop("Cluster CSV barcodes do not match h5ad cell names; misalignment detected.")
    }

    cluster_ids <- clusters_df$cluster

    # Resolve batch labels (colData column or NULL).
    batch_labels <- NULL
    if (batch_arg != "NONE" && batch_arg %in% colnames(colData(sce))) {
      batch_labels <- colData(sce)[[batch_arg]]
    }

    # Run testClusters (sc-SHC formal split significance test).
    # testClusters returns a list with per-split significance results.
    test_result <- scSHC::testClusters(
      data = counts_matrix,
      cluster_ids = cluster_ids,
      batch = batch_labels,
      alpha = alpha
    )

    # Extract per-split significance results.
    # testClusters returns a list with $p_norm (p-values) and $idx (split indices).
    # Build a per-split significance data frame.
    if (!is.null(test_result$p_norm)) {
      n_splits <- length(test_result$p_norm)
      results <- data.frame(
        split_index = seq_len(n_splits),
        p_value = test_result$p_norm,
        significant = test_result$p_norm < alpha,
        stringsAsFactors = FALSE
      )
    } else {
      # No splits tested (single cluster or test failed).
      results <- data.frame(
        split_index = integer(0),
        p_value = numeric(0),
        significant = logical(0),
        stringsAsFactors = FALSE
      )
    }

    # Write per-split significance CSV.
    write.csv(results, out_csv, row.names = FALSE)
  },
  error = function(e) {
    cat(paste("ERROR:", e$message, "\n"), file = stderr())
    quit(status = 1)
  }
)
