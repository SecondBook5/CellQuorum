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
    cluster_levels <- sort(unique(clusters))
    cluster_palette <- grDevices::hcl.colors(length(cluster_levels), "Set2")
    cluster_colors <- setNames(cluster_palette, cluster_levels)
    condition_levels <- sort(unique(conditions))
    condition_palette <- c("#c0392b", "#3a6ea5")[seq_along(condition_levels)]
    condition_colors <- setNames(condition_palette, condition_levels)

    top_anno <- HeatmapAnnotation(
      Cluster = clusters,
      Condition = conditions,
      col = list(Cluster = cluster_colors, Condition = condition_colors),
      annotation_name_side = "left"
    )

    n_genes <- nrow(mat_z)
    height_px <- max(1800, min(8000, 22 * n_genes + 400))
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
    width_px <- max(2400, 200 * length(cluster_levels) + 900)

    png(out_png, width = width_px, height = height_px, res = 300)
    ht <- Heatmap(
      mat_z,
      name = "z-score",
      col = col_fun,
      top_annotation = top_anno,
      cluster_columns = FALSE,
      column_split = clusters,
      column_title = NULL,
      column_gap = unit(2.5, "mm"),
      cluster_rows = TRUE,
      show_row_dend = TRUE,
      show_row_names = show_names,
      row_names_gp = gpar(fontsize = 7, fontface = "italic"),
      column_names_gp = gpar(fontsize = 8),
      column_names_rot = 45,
      use_raster = n_genes > 500,
      heatmap_legend_param = list(title = "mean expr\n(z-scored)")
    )
    # Only the z-score colour scale moves to sit above the Cluster/Condition
    # annotation bars, freeing horizontal width for the heatmap body itself.
    # The Cluster/Condition annotation legends stay in their default
    # right-hand, vertically-stacked position -- untouched.
    draw(ht, heatmap_legend_side = "top")
    dev.off()
    cat("de_heatmap.R: wrote", out_png, "(", n_genes, "genes )\n")
  },
  error = function(e) {
    cat("ERROR in de_heatmap.R:", conditionMessage(e), "\n", file = stderr())
    quit(status = 1)
  }
)
