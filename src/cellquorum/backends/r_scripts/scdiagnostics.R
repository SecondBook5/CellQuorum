# scDiagnostics annotation-confidence diagnostics.
# Usage: Rscript scdiagnostics.R <query.h5ad> <out.csv> <cell_type_col> \
#          <ref.h5ad|NONE> <soft_scores.csv|NONE> <pc_subset> <n_tree> <n_neighbor>
suppressPackageStartupMessages({
  library(zellkonverter)
  library(scDiagnostics)
  library(SingleCellExperiment)
  # runPCA: the reference PCA has to be computed here so its rotation matrix exists. See the
  # reference branch below.
  library(scater)
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
    # reader="R" tolerates null-encoded uns entries the python reader crashes on.
    query_sce <- zellkonverter::readH5AD(query_h5ad, reader = "R")

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

    # Force the label column to CHARACTER.
    #
    # anndata writes a string obs column as an h5ad categorical (strings_to_categoricals is on
    # by default), zellkonverter reads that back as an R factor, and scDiagnostics' internal
    # `projectPCA` then reports `cell_type` as the factor's integer CODES — 1, 2, 3 — so its own
    # `cell_type == "<label>"` filter matches nothing. Every cell type came back with
    # `n_query = 0` and a NaN probability, while `detectAnomaly`, which subsets the SCE
    # directly, worked on the very same input. Writing strings from Python cannot prevent this
    # because anndata converts them on the way out, so it is undone here instead.
    colData(query_sce)[[cell_type_col]] <-
      as.character(colData(query_sce)[[cell_type_col]])

    # Initialize result data frame (keyed by barcode).
    barcodes <- colnames(query_sce)
    results <- data.frame(barcode = barcodes, stringsAsFactors = FALSE)

    # Branch: reference-based diagnostics if reference provided.
    has_reference <- (ref_arg != "NONE" && file.exists(ref_arg))
    if (has_reference) {
      # Read reference h5ad → SingleCellExperiment.
      ref_sce <- zellkonverter::readH5AD(ref_arg, reader = "R")

      # Ensure reference has logcounts and PCA.
      if (!"logcounts" %in% assayNames(ref_sce)) {
        if (length(assayNames(ref_sce)) > 0) {
          assayNames(ref_sce)[1] <- "logcounts"
        } else {
          stop("Reference h5ad has no assays.")
        }
      }
      if (!cell_type_col %in% colnames(colData(ref_sce))) {
        stop(paste("Reference h5ad missing cell_type column:", cell_type_col))
      }
      # Character, for the same reason as the query above.
      colData(ref_sce)[[cell_type_col]] <-
        as.character(colData(ref_sce)[[cell_type_col]])

      # Restrict BOTH objects to the genes they share, then compute the reference PCA HERE.
      #
      # scDiagnostics::projectPCA projects the query onto the reference's PC space using
      #   rotation_mat <- attributes(reducedDim(reference_data, "PCA"))[["rotation"]]
      #   PCA_genes    <- rownames(rotation_mat)
      # and stops with "Genes in reference PCA are not found in query data." when that is
      # missing. A PCA read back from an h5ad has no rotation attribute — zellkonverter maps
      # the coordinates and nothing else — so reusing a stored `X_pca` could never work, and
      # reusing coordinates computed on the reference's own gene set and normalization would
      # not be a shared space with the query even if it did. scater::runPCA attaches the
      # rotation, which is what makes the projection meaningful.
      shared_genes <- intersect(rownames(ref_sce), rownames(query_sce))
      if (length(shared_genes) < 10) {
        stop(paste0(
          "Reference and query share only ", length(shared_genes), " genes; ",
          "scDiagnostics projects the query onto the reference PC space and cannot do that ",
          "across disjoint gene sets. Reference names genes like ",
          paste(head(rownames(ref_sce), 2), collapse = "/"), " and query like ",
          paste(head(rownames(query_sce), 2), collapse = "/"), "."
        ))
      }
      ref_sce <- ref_sce[shared_genes, ]
      query_sce <- query_sce[shared_genes, ]

      n_pcs <- max(pc_subset)
      ref_sce <- scater::runPCA(ref_sce, ncomponents = n_pcs, exprs_values = "logcounts")
      message(paste0(
        "Reference PCA computed on ", length(shared_genes),
        " shared genes; ", n_pcs, " components, rotation retained for projection."
      ))

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
      # detectAnomaly returns a list KEYED BY CELL TYPE. Each element carries
      # `query_anomaly_scores` for that type's query cells only, and their barcodes are the
      # rownames of `query_mat_subset`. So the per-cell vector is assembled by scattering each
      # type's scores back to its own barcodes.
      #
      # The previous code read `anomaly_result$anomaly_scores` — a field this object does not
      # have — and fell through to a "not in expected format; skipped" message. That was not a
      # defensive branch catching an odd case: it was the only branch that ever ran, so the
      # reference diagnostics silently produced nothing every time, and the CSV came back
      # holding barcodes and no diagnostics at all.
      anomaly_by_barcode <- rep(NA_real_, length(barcodes))
      names(anomaly_by_barcode) <- barcodes
      anomaly_flag <- rep(NA, length(barcodes))
      names(anomaly_flag) <- barcodes
      n_anomaly_scored <- 0
      for (type_name in names(anomaly_result)) {
        element <- anomaly_result[[type_name]]
        scores <- element$query_anomaly_scores
        type_barcodes <- rownames(element$query_mat_subset)
        if (is.null(scores) || is.null(type_barcodes) ||
            length(scores) != length(type_barcodes)) {
          message(paste0("detectAnomaly: unusable result for '", type_name, "'; left NA"))
          next
        }
        known <- type_barcodes %in% barcodes
        anomaly_by_barcode[type_barcodes[known]] <- scores[known]
        if (!is.null(element$query_anomaly)) {
          anomaly_flag[type_barcodes[known]] <- element$query_anomaly[known]
        }
        n_anomaly_scored <- n_anomaly_scored + sum(known)
      }
      if (n_anomaly_scored > 0) {
        results$scdiag_anomaly <- as.numeric(anomaly_by_barcode[barcodes])
        results$scdiag_anomaly_flag <- as.logical(anomaly_flag[barcodes])
        message(paste0("detectAnomaly: scored ", n_anomaly_scored, " of ",
                       length(barcodes), " query cells."))
      } else {
        message("detectAnomaly returned no usable per-cell scores.")
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
      # calculateNearestNeighborProbabilities returns ONE number per cell type, not one per
      # cell: each element holds a scalar `query_prob` alongside `n_query`. So this is a
      # population-level statistic and the column name says so — `_group` — because a
      # group value broadcast into a per-cell column is read as per-cell evidence, and a
      # reviewer would take it for a per-cell confidence. `n_query` travels with it so the
      # denominator is visible: a probability from 4 cells is not a probability from 4,000.
      knn_by_barcode <- rep(NA_real_, length(barcodes))
      names(knn_by_barcode) <- barcodes
      knn_n <- rep(NA_integer_, length(barcodes))
      names(knn_n) <- barcodes
      query_labels <- as.character(colData(query_sce)[[cell_type_col]])
      n_knn_types <- 0
      for (type_name in names(knn_result)) {
        element <- knn_result[[type_name]]
        prob <- element$query_prob
        if (is.null(prob) || length(prob) != 1 || !is.finite(prob)) next
        in_type <- query_labels == type_name
        knn_by_barcode[in_type] <- as.numeric(prob)
        knn_n[in_type] <- if (is.null(element$n_query)) NA_integer_ else as.integer(element$n_query)
        n_knn_types <- n_knn_types + 1
      }
      if (n_knn_types > 0) {
        results$scdiag_knn_prob_group <- as.numeric(knn_by_barcode[barcodes])
        results$scdiag_knn_prob_group_n <- as.integer(knn_n[barcodes])
        message(paste0("kNN probabilities: ", n_knn_types,
                       " cell-type-level values broadcast to their member cells."))
      } else {
        message("kNN probabilities returned nothing usable.")
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
