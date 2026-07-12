# scDiagnostics annotation-confidence diagnostics.
# Usage: Rscript scdiagnostics.R <query.h5ad> <out.csv> <cell_type_col> \
#          <ref.h5ad|NONE> <soft_scores.csv|NONE> <pc_subset> <n_tree> <n_neighbor>
suppressPackageStartupMessages({
  library(zellkonverter)
  library(scDiagnostics)
  library(SingleCellExperiment)
})

# Parse command-line arguments.
args <- commandArgs(trailingOnly = TRUE)
query_h5ad <- args[1]
out_csv <- args[2]
cell_type_col <- args[3]
ref_arg <- args[4]
soft_scores_arg <- args[5]
pc_subset_str <- args[6]
n_tree <- as.integer(args[7])
n_neighbor <- as.integer(args[8])

# Parse PC subset (comma-separated 1-indexed integers).
pc_subset <- as.integer(strsplit(pc_subset_str, ",")[[1]])

# Wrap everything in tryCatch for fail-loud behavior.
tryCatch(
  {
    # Read query h5ad → SingleCellExperiment.
    query_sce <- zellkonverter::readH5AD(query_h5ad)

    # Ensure logcounts assay exists (zellkonverter maps X → first assay).
    if (!"logcounts" %in% assayNames(query_sce)) {
      # Rename the first assay to logcounts if present.
      if (length(assayNames(query_sce)) > 0) {
        assayNames(query_sce)[1] <- "logcounts"
      } else {
        stop("Query h5ad has no assays to use as logcounts.")
      }
    }

    # Ensure PCA reducedDim exists (zellkonverter maps obsm X_pca → PCA).
    if (!"PCA" %in% reducedDimNames(query_sce)) {
      # Try to rename X_pca if present.
      if ("X_pca" %in% reducedDimNames(query_sce)) {
        reducedDimNames(query_sce)[reducedDimNames(query_sce) == "X_pca"] <- "PCA"
      } else {
        stop("Query h5ad missing PCA reducedDim (X_pca in obsm).")
      }
    }

    # Ensure cell_type column exists in colData.
    if (!cell_type_col %in% colnames(colData(query_sce))) {
      stop(paste("Query h5ad missing cell_type column:", cell_type_col))
    }

    # Initialize result data frame (keyed by barcode).
    barcodes <- colnames(query_sce)
    results <- data.frame(barcode = barcodes, stringsAsFactors = FALSE)

    # Branch: reference-based diagnostics if reference provided.
    has_reference <- (ref_arg != "NONE" && file.exists(ref_arg))
    if (has_reference) {
      # Read reference h5ad → SingleCellExperiment.
      ref_sce <- zellkonverter::readH5AD(ref_arg)

      # Ensure reference has logcounts and PCA.
      if (!"logcounts" %in% assayNames(ref_sce)) {
        if (length(assayNames(ref_sce)) > 0) {
          assayNames(ref_sce)[1] <- "logcounts"
        } else {
          stop("Reference h5ad has no assays.")
        }
      }
      if (!"PCA" %in% reducedDimNames(ref_sce)) {
        if ("X_pca" %in% reducedDimNames(ref_sce)) {
          reducedDimNames(ref_sce)[reducedDimNames(ref_sce) == "X_pca"] <- "PCA"
        } else {
          stop("Reference h5ad missing PCA reducedDim.")
        }
      }
      if (!cell_type_col %in% colnames(colData(ref_sce))) {
        stop(paste("Reference h5ad missing cell_type column:", cell_type_col))
      }

      # Get unique cell types (use query cell types for filtering).
      query_types <- unique(colData(query_sce)[[cell_type_col]])
      cell_types <- as.character(query_types)

      # Run detectAnomaly (isolation forest).
      anomaly_result <- scDiagnostics::detectAnomaly(
        reference_data = ref_sce,
        query_data = query_sce,
        ref_cell_type_col = cell_type_col,
        query_cell_type_col = cell_type_col,
        cell_types = cell_types,
        pc_subset = pc_subset,
        n_tree = n_tree,
        anomaly_treshold = 0.5
      )
      # DEFENSIVE: detectAnomaly may return nested per-cell-type results.
      # Extract in query-cell order; skip if structure unexpected.
      if (!is.null(anomaly_result$anomaly_scores) &&
          length(anomaly_result$anomaly_scores) == length(barcodes)) {
        results$scdiag_anomaly <- anomaly_result$anomaly_scores
      } else {
        message("detectAnomaly scores not in expected per-query-cell format; ",
                "skipped")
      }

      # Run calculateNearestNeighborProbabilities (kNN confidence).
      knn_result <- scDiagnostics::calculateNearestNeighborProbabilities(
        query_data = query_sce,
        reference_data = ref_sce,
        query_cell_type_col = cell_type_col,
        ref_cell_type_col = cell_type_col,
        cell_types = cell_types,
        pc_subset = pc_subset,
        n_neighbor = n_neighbor
      )
      # DEFENSIVE: kNN probabilities may be nested per cell type.
      # Extract in query-cell order; skip if structure unexpected.
      if (!is.null(knn_result$nn_probabilities) &&
          length(knn_result$nn_probabilities) == length(barcodes)) {
        results$scdiag_knn_prob <- knn_result$nn_probabilities
      } else {
        message("kNN probabilities not in expected per-query-cell format; ",
                "skipped")
      }
    }

    # Query-only: calculateCategorizationEntropy if soft scores provided.
    if (soft_scores_arg != "NONE" && file.exists(soft_scores_arg)) {
      # Read soft scores (cells x cell_types matrix).
      soft_scores <- read.csv(soft_scores_arg, row.names = 1, check.names = FALSE)
      # Ensure same order as query barcodes.
      soft_scores <- soft_scores[barcodes, , drop = FALSE]
      # Convert to matrix.
      soft_matrix <- as.matrix(soft_scores)
      # Calculate per-cell entropy (Shannon entropy of soft probability distribution).
      # scDiagnostics::calculateCategorizationEntropy returns per-cell-type entropy,
      # but we want per-cell uncertainty → compute Shannon entropy manually.
      entropy_values <- apply(soft_matrix, 1, function(row) {
        p <- row[row > 0]  # Filter out zeros to avoid log(0).
        if (length(p) == 0) return(0)
        -sum(p * log2(p))
      })
      results$scdiag_entropy <- entropy_values
    }

    # Write per-cell CSV.
    write.csv(results, out_csv, row.names = FALSE)
  },
  error = function(e) {
    cat(paste("ERROR:", e$message, "\n"), file = stderr())
    quit(status = 1)
  }
)
