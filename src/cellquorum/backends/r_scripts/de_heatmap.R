# Advanced DE heatmap: genes x (cluster, condition), z-scored per gene, column
# split by cluster, ComplexHeatmap renders it directly (unlike edgeR/milo/
# propeller, which write CSVs the Python side plots -- this is the one figure
# ComplexHeatmap genuinely outperforms a hand-rolled matplotlib rendering for:
# real dendrograms, real annotation bars, no third-party equivalent in Python).
# Usage: Rscript de_heatmap.R <matrix.csv> <out.png>
suppressPackageStartupMessages({
  library(ComplexHeatmap)
  library(circlize)
})

args <- commandArgs(trailingOnly = TRUE)
matrix_csv <- args[1]
out_png <- args[2]

tryCatch(
  {
    mat <- read.csv(matrix_csv, row.names = 1, check.names = FALSE)
    mat <- as.matrix(mat)

    # Column names are "<cluster>_<condition>"; condition never contains "_"
    # (validated Python-side by controlled vocabulary), so splitting on the
    # FIRST underscore is unambiguous even for multi-word cluster labels.
    clusters <- sub("_[^_]+$", "", colnames(mat))
    conditions <- sub(".*_([^_]+)$", "\\1", colnames(mat))

    mat_z <- t(scale(t(mat)))
    mat_z[is.na(mat_z)] <- 0 # zero-variance rows (flat across all groups)

    col_fun <- colorRamp2(c(-2, 0, 2), c("#2b6e71", "#f4f5f1", "#a8402f"))

    # Order cluster blocks by similarity (centroid hclust on the z-scored
    # matrix) rather than alphabetically, so adjacent blocks are the ones
    # that actually look alike. Only the block ORDER is clustered --
    # clustering the 2 condition columns within a block too would let
    # Leiden left/right flip per cluster with only 2 leaves to place,
    # trading a real signal (which clusters resemble each other) for pure
    # layout noise (which side Lymphedema lands on).
    raw_cluster_levels <- unique(clusters)
    if (length(raw_cluster_levels) > 2) {
      centroids <- sapply(raw_cluster_levels, function(cl) rowMeans(mat_z[, clusters == cl, drop = FALSE]))
      slice_order <- raw_cluster_levels[hclust(dist(t(centroids)))$order]
    } else {
      slice_order <- sort(raw_cluster_levels)
    }
    clusters <- factor(clusters, levels = slice_order)
    cluster_levels <- levels(clusters)
    cluster_palette <- grDevices::hcl.colors(length(cluster_levels), "Set2")
    cluster_colors <- setNames(cluster_palette, cluster_levels)
    condition_levels <- sort(unique(conditions))
    condition_palette <- c("#c0392b", "#3a6ea5")[seq_along(condition_levels)]
    condition_colors <- setNames(condition_palette, condition_levels)

    # show_legend=FALSE + show_heatmap_legend=FALSE below: ComplexHeatmap's
    # default draw() treats the heatmap's own colour-scale legend and the
    # annotations' legends as two SEPARATE groups placed side by side
    # (visually two columns), even with everything on the "right" side --
    # merge_legend=TRUE doesn't fix the ordering either. Building all three
    # by hand and passing them as one ordered annotation_legend_list is the
    # only way to get a single vertical stack with z-score on top.
    top_anno <- HeatmapAnnotation(
      Cluster = clusters,
      Condition = conditions,
      col = list(Cluster = cluster_colors, Condition = condition_colors),
      annotation_name_side = "left",
      show_legend = FALSE
    )
    legend_list <- list(
      Legend(col_fun = col_fun, title = "mean expr\n(z-scored)", at = c(-2, -1, 0, 1, 2)),
      Legend(labels = cluster_levels, legend_gp = gpar(fill = cluster_colors), title = "Cluster"),
      Legend(labels = condition_levels, legend_gp = gpar(fill = condition_colors), title = "Condition")
    )

    n_genes <- nrow(mat_z)
    # ~32px/gene row gives the row-name text visible breathing room between
    # adjacent labels; the previous 22px/gene packed them close enough to
    # read as touching even though DIMNAME_PADDING (the heatmap-to-text gap)
    # was already generous -- that padding is horizontal, not the vertical
    # gap between one gene's label and the next.
    height_px <- max(1800, min(9000, 32 * n_genes + 400))
    show_names <- n_genes <= 200

    # Extra breathing room between the heatmap body and the row (gene) name
    # text -- ComplexHeatmap's default (1mm) sits nearly flush against the
    # matrix; 6mm is large enough to read as deliberate spacing rather than
    # rounding noise.
    ht_opt$DIMNAME_PADDING <- unit(6, "mm")

    # ComplexHeatmap's per-slice column_title (auto-generated from
    # column_split) draws centered text at a fixed font size with no width
    # constraint or wrapping -- a bare numeric id ("0", "1"...) never
    # overflowed its slice, but a real identity label ("Mechanosensory",
    # "Progenitor-like") reliably does, bleeding into its neighbor
    # regardless of device width or column_gap (both tried, neither
    # changes the text viewport itself). The "Cluster" annotation bar
    # below already conveys the same identity, unambiguously, colour-coded
    # with a legend -- so the redundant, overflow-prone per-slice title is
    # dropped rather than fought.
    #
    # Width scales on actual data columns (clusters x conditions), not
    # cluster count -- 6 clusters x 2 conditions is 12 columns, and the old
    # per-cluster-only formula left most of that freed-up horizontal room
    # (from moving the legend off the top) unused, making the figure read
    # as too tall/narrow for how few columns it actually has. The +1400
    # overhead covers the 45mm left padding, row dendrogram, gene-name
    # column, and right-side legend, none of which scale with column count.
    n_data_cols <- ncol(mat_z)
    width_px <- max(3200, 230 * n_data_cols + 1400)

    png(out_png, width = width_px, height = height_px, res = 300)
    ht <- Heatmap(
      mat_z,
      name = "z-score",
      col = col_fun,
      top_annotation = top_anno,
      cluster_columns = FALSE,
      column_split = clusters,
      cluster_column_slices = FALSE,
      column_title = NULL,
      column_gap = unit(2.5, "mm"),
      cluster_rows = TRUE,
      show_row_dend = TRUE,
      show_row_names = show_names,
      row_names_gp = gpar(fontsize = 7, fontface = "italic"),
      column_names_gp = gpar(fontsize = 8),
      column_names_rot = 45,
      use_raster = n_genes > 500,
      show_heatmap_legend = FALSE
    )
    # annotation_legend_list carries all three hand-built legends in the
    # order we want (z-score, Cluster, Condition), stacked as one column.
    # Extra left padding is required because the 45-degree column labels
    # are right-anchored at the top of each column and extend down-and-LEFT
    # from there -- with real identity labels ("Mechanosensory_Lymphedema")
    # the leftmost label's tail runs past the device's left edge and gets
    # clipped without it.
    draw(ht, annotation_legend_list = legend_list, padding = unit(c(2, 45, 2, 2), "mm"))
    dev.off()
    cat("de_heatmap.R: wrote", out_png, "(", n_genes, "genes )\n")
  },
  error = function(e) {
    cat("ERROR in de_heatmap.R:", conditionMessage(e), "\n", file = stderr())
    quit(status = 1)
  }
)
