# CHOIR permutation-tested clustering with significance pruning.
# Usage: Rscript choir.R <in.h5ad> <out.csv> <key> <alpha> <n_iterations> <n_trees> <batch_key_or_NONE> <seed>
suppressPackageStartupMessages({
  library(zellkonverter)
  library(CHOIR)
  library(SingleCellExperiment)
})

# Parse command-line arguments.
args <- commandArgs(trailingOnly = TRUE)
in_h5ad <- args[1]
out_csv <- args[2]
key <- args[3]
alpha <- as.numeric(args[4])
n_iterations <- as.integer(args[5])
n_trees <- as.integer(args[6])
batch_arg <- args[7]
seed <- as.integer(args[8])

# Wrap everything in tryCatch for fail-loud behavior.
tryCatch(
  {
    # Read h5ad → SingleCellExperiment.
    sce <- zellkonverter::readH5AD(in_h5ad)

    # Ensure counts assay exists (zellkonverter maps X → first assay).
    # CHOIR needs raw counts; rename X → counts if needed.
    if (!"counts" %in% assayNames(sce)) {
      if (length(assayNames(sce)) > 0) {
        # Rename the first assay to counts.
        assayNames(sce)[1] <- "counts"
      } else {
        stop("Input h5ad has no assays to use as counts.")
      }
    }

    # Resolve batch_labels (column name string or NULL).
    # CHOIR expects batch_labels to be a character column name, not the vector.
    batch_labels <- NULL
    batch_correction_method <- "none"
    if (batch_arg != "NONE" && batch_arg %in% colnames(colData(sce))) {
      batch_labels <- batch_arg  # Pass the column name, not the vector.
      batch_correction_method <- "Harmony"
    }

    # Run CHOIR (permutation-tested hierarchical clustering with pruning).
    # CHOIR does its own normalization + dim-reduction + batch correction.
    # verbose=FALSE to suppress progress messages.
    sce_choir <- CHOIR::CHOIR(
      object = sce,
      key = key,
      alpha = alpha,
      n_iterations = n_iterations,
      n_trees = n_trees,
      batch_correction_method = batch_correction_method,
      batch_labels = batch_labels,
      random_seed = seed,
      verbose = FALSE
    )

    # Extract CHOIR cluster labels from colData.
    # CHOIR writes results to colData with specific naming pattern.
    # The key parameter sets a prefix; CHOIR appends "_clusters_<alpha>".
    coldata_names <- colnames(colData(sce_choir))

    # Look for CHOIR result columns (pattern: key_clusters_* or CHOIR_clusters_*).
    choir_pattern <- paste0("^(", key, "|CHOIR)_clusters")
    choir_cols <- grep(choir_pattern, coldata_names, value = TRUE)

    if (length(choir_cols) == 0) {
      stop(paste("CHOIR result column not found in colData; expected pattern:", choir_pattern))
    }

    # Use the first match if multiple columns exist.
    choir_col <- choir_cols[1]

    # Extract per-cell cluster labels.
    cluster_labels <- colData(sce_choir)[[choir_col]]

    # Write barcode,subcluster CSV.
    barcodes <- colnames(sce_choir)
    results <- data.frame(
      barcode = barcodes,
      subcluster = as.character(cluster_labels),
      stringsAsFactors = FALSE
    )

    write.csv(results, out_csv, row.names = FALSE)
  },
  error = function(e) {
    cat(paste("ERROR:", e$message, "\n"), file = stderr())
    quit(status = 1)
  }
)
