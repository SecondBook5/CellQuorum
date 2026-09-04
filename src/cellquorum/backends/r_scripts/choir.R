# CHOIR permutation-tested clustering with significance pruning.
# Usage: Rscript choir.R <in.h5ad> <out.csv> <key> <alpha> <n_iterations> <n_trees> \
#            <batch_key_or_NONE> <seed> [reduction_obsm_or_NONE]
#
# The optional 9th argument names an obsm/reducedDim key holding a PRECOMPUTED
# cell embedding (e.g. "X_pca_harmony"). Supplying it is the supported way to get
# batch-corrected CHOIR clustering when CHOIR's own Harmony path is unavailable:
# the correction is already baked into the embedding, so CHOIR clusters on it
# directly and never calls harmony itself. See the harmony-compat note below.
suppressPackageStartupMessages({
  library(zellkonverter)
  library(CHOIR)
  library(SingleCellExperiment)
  library(scuttle)
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
reduction_arg <- if (length(args) >= 9) args[9] else "NONE"

# Wrap everything in tryCatch for fail-loud behavior.
tryCatch(
  {
    # Read h5ad → SingleCellExperiment. Use reader="R": the default python
    # (reticulate/anndata) reader crashes with an IORegistryError on
    # null-encoded uns entries (e.g. uns/log1p/base written by anndata >= 0.11),
    # which real user objects commonly carry. The native-R reader tolerates them.
    sce <- zellkonverter::readH5AD(in_h5ad, reader = "R")

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

    # CHOIR's dimensionality-reduction step reads a 'logcounts' assay; create
    # it from counts if absent (scuttle log-normalization). Without this CHOIR
    # errors with "'to' must be of length 1" at Step 2.
    if (!"logcounts" %in% assayNames(sce)) {
      sce <- scuttle::logNormCounts(sce)
    }

    # Resolve batch_labels (column name string or NULL).
    # CHOIR expects batch_labels to be a character column name, not the vector.
    batch_labels <- NULL
    batch_correction_method <- "none"
    if (batch_arg != "NONE" && batch_arg %in% colnames(colData(sce))) {
      batch_labels <- batch_arg  # Pass the column name, not the vector.
      batch_correction_method <- "Harmony"
    }

    # CHOIR (<= 0.3.0) calls harmony::HarmonyMatrix(), which harmony removed in
    # 1.0 (RunHarmony() replaced it). With a modern harmony installed, requesting
    # Harmony correction aborts the whole run. Degrade to uncorrected clustering
    # instead of failing: an uncorrected run is still interpretable, because
    # residual batch structure can only ADD splits, so a low surviving cluster
    # count remains a conservative result. Announced on stderr so the caller can
    # record that correction did not happen.
    if (batch_correction_method == "Harmony" &&
        !("HarmonyMatrix" %in% getNamespaceExports("harmony"))) {
      cat(paste0(
        "WARNING: harmony ", as.character(utils::packageVersion("harmony")),
        " does not export HarmonyMatrix(), which CHOIR ",
        as.character(utils::packageVersion("CHOIR")),
        " requires; falling back to batch_correction_method='none'.\n"
      ), file = stderr())
      batch_correction_method <- "none"
      batch_labels <- NULL
    }

    # Resolve an optional user-supplied embedding. Passing one means CHOIR skips
    # its own dimensionality reduction, so `subtree_reductions` must be FALSE:
    # regenerating a reduction per subtree is what would re-enter the broken
    # harmony path. The documented cost is possible UNDERclustering, which is the
    # conservative direction — a surviving split is still a supported split.
    reduction <- NULL
    var_features <- NULL
    subtree_reductions <- TRUE
    if (reduction_arg != "NONE") {
      if (!(reduction_arg %in% reducedDimNames(sce))) {
        stop(paste0(
          "reduction '", reduction_arg, "' not found in reducedDims; available: ",
          paste(reducedDimNames(sce), collapse = ", ")
        ))
      }
      reduction <- as.matrix(reducedDim(sce, reduction_arg))
      rownames(reduction) <- colnames(sce)
      subtree_reductions <- FALSE
      # CHOIR requires var_features alongside a user-supplied reduction: the
      # embedding alone does not tell it which genes the space was built from,
      # and it needs them for the per-split feature comparisons. Read them from
      # the boolean rowData column that scanpy's HVG step writes.
      if (!("highly_variable" %in% colnames(rowData(sce)))) {
        stop(paste(
          "a user-supplied reduction also requires var_features, expected as a",
          "boolean 'highly_variable' column in rowData; found:",
          paste(colnames(rowData(sce)), collapse = ", ")
        ))
      }
      hv <- as.logical(rowData(sce)$highly_variable)
      hv[is.na(hv)] <- FALSE
      var_features <- rownames(sce)[hv]
      if (length(var_features) < 2) {
        stop("rowData$highly_variable selected fewer than 2 features")
      }
      cat(paste0("Using ", length(var_features), " variable features.\n"),
          file = stderr())
      # The embedding already carries the correction; asking CHOIR to correct
      # again would double-correct and re-enter harmony.
      batch_correction_method <- "none"
      batch_labels <- NULL
      cat(paste0(
        "Using precomputed reduction '", reduction_arg, "' (",
        nrow(reduction), " cells x ", ncol(reduction),
        " dims); subtree_reductions=FALSE.\n"
      ), file = stderr())
    }

    # Run CHOIR (permutation-tested hierarchical clustering with pruning).
    # verbose=FALSE to suppress progress messages.
    sce_choir <- CHOIR::CHOIR(
      object = sce,
      key = key,
      alpha = alpha,
      n_iterations = n_iterations,
      n_trees = n_trees,
      batch_correction_method = batch_correction_method,
      batch_labels = batch_labels,
      reduction = reduction,
      var_features = var_features,
      subtree_reductions = subtree_reductions,
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
