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

    # Run testClusters (sc-SHC formal split significance test). scSHC also needs
    # the data.tree package to build/inspect the returned hierarchy.
    suppressPackageStartupMessages(library(data.tree))
    test_result <- scSHC::testClusters(
      data = counts_matrix,
      cluster_ids = cluster_ids,
      batch = batch_labels,
      alpha = alpha
    )

    # testClusters returns list(cluster_labels, node0):
    #   [[1]] cluster_labels: the significance-reconciled labels. Original
    #         clusters that are NOT significantly distinct are merged and
    #         relabeled "new<k>"; the number of distinct labels = number of
    #         clusters that survived the formal test.
    #   [[2]] node0: a data.tree Node whose node names embed the per-split
    #         adjusted p-value, e.g. "Node 1: 0.03" / "Cluster 2: 0.15".
    # There is NO $p_norm field (the previous assumption produced empty output).
    reconciled_labels <- as.character(test_result[[1]])
    tree_root <- test_result[[2]]

    # Per-split significance. scSHC labels every TESTED split with a trailing
    # ": <value>" (the multiple-testing-adjusted split QC value): internal
    # "Node <n>: <v>" splits that were significant enough to keep descending, and
    # "Cluster <n>: <v>" splits that were tested but stopped (not significant).
    # Terminal leaves are bare "Cluster <n>" with no colon and are NOT splits.
    # So a tested split == any node name containing ": <number>".
    node_names <- character(0)
    if (!is.null(tree_root) && inherits(tree_root, "Node")) {
      node_names <- tree_root$Get("name")
      node_names <- node_names[!is.na(node_names)]
    }

    # Keep only names with a ": <number>" suffix (tested splits, Node or Cluster).
    split_names <- node_names[grepl(":\\s*[0-9.]+\\s*$", node_names)]
    parse_pval <- function(nm) {
      # Take the trailing number after the final ": ".
      suppressWarnings(as.numeric(sub("^.*:\\s*", "", nm)))
    }
    p_values <- vapply(split_names, parse_pval, numeric(1), USE.NAMES = FALSE)

    if (length(p_values) > 0) {
      results <- data.frame(
        split_index = seq_along(p_values),
        node = split_names,
        p_value = p_values,
        # scSHC's node value is the (multiple-testing-adjusted) split QC value;
        # a split is significant/retained when it is <= alpha.
        significant = !is.na(p_values) & p_values <= alpha,
        stringsAsFactors = FALSE
      )
    } else {
      # No internal split nodes: the labeling reduced to a single cluster (no
      # testable split), which is a valid, informative result — not a failure.
      results <- data.frame(
        split_index = integer(0),
        node = character(0),
        p_value = numeric(0),
        significant = logical(0),
        stringsAsFactors = FALSE
      )
    }

    # Write per-split significance CSV.
    write.csv(results, out_csv, row.names = FALSE)

    # Write the reconciled cluster labels alongside (barcode,label) so the caller
    # can see which input clusters scSHC merged as not-significantly-distinct.
    labels_out <- sub("\\.csv$", "_labels.csv", out_csv)
    write.csv(
      data.frame(
        barcode = colnames(sce),
        scshc_label = reconciled_labels,
        stringsAsFactors = FALSE
      ),
      labels_out,
      row.names = FALSE
    )
  },
  error = function(e) {
    cat(paste("ERROR:", e$message, "\n"), file = stderr())
    quit(status = 1)
  }
)
