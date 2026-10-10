# NES-by-cell-type heatmap (GSEA): gene sets x cell types, drawn with ComplexHeatmap.
# Cells show the normalized enrichment score; a cell is coloured only where padj < alpha and is
# neutral grey otherwise ("grey = not significant"); stars mark padj < alpha (*) and < strong (**).
# A gene set a cell type could not test is left white and marked "n.t.", never drawn as zero.
# Rows and columns are clustered on NES (hierarchical, average linkage); count bars on the top and
# right show how many gene sets / cell types are significant up and down.
# Usage: Rscript gsea_nes_heatmap.R <nes.csv> <padj.csv> <out.png> <alpha> <strong> <title> <footnote>
# Writes <out.png> and a vector twin <out.pdf>.
suppressPackageStartupMessages({
  library(ComplexHeatmap)
  library(circlize)
})

args <- commandArgs(trailingOnly = TRUE)
nes_csv <- args[1]
padj_csv <- args[2]
out_png <- args[3]
alpha <- as.numeric(args[4])
strong <- as.numeric(args[5])
title <- if (length(args) >= 6) args[6] else "GSEA: NES by cell type"
footnote <- if (length(args) >= 7) args[7] else ""

tryCatch(
  {
    nes <- as.matrix(read.csv(nes_csv, row.names = 1, check.names = FALSE))
    padj <- as.matrix(read.csv(padj_csv, row.names = 1, check.names = FALSE))
    tested <- !is.na(nes)
    sig <- tested & !is.na(padj) & padj < alpha
    up <- sig & nes > 0
    down <- sig & nes < 0

    # NES colours are symmetric about 0 so red and blue are comparable; a missing cell is 0 only
    # for clustering, and is drawn as "n.t." below.
    lim <- ceiling(max(abs(nes[sig]), na.rm = TRUE) * 2) / 2
    if (!is.finite(lim) || lim < 2) lim <- 2
    col_fun <- colorRamp2(c(-lim, 0, lim), c("#2166AC", "#F7F7F7", "#B2182B"))
    ns_fill <- "#E6E6E6"
    up_col <- "#B2182B"
    down_col <- "#2166AC"
    nes0 <- nes
    nes0[!tested] <- 0

    stars <- matrix(ifelse(!tested | is.na(padj), "", ifelse(padj < strong, "**", ifelse(padj < alpha, "*", ""))),
                    nrow = nrow(nes))
    labels <- matrix(ifelse(tested, sprintf("%.1f%s", nes, stars), "n.t."), nrow = nrow(nes))

    cell_fun <- function(j, i, x, y, width, height, fill) {
      if (!tested[i, j]) {
        grid.rect(x, y, width, height, gp = gpar(fill = "white", col = "white", lwd = 1))
        grid.text("n.t.", x, y, gp = gpar(fontsize = 5.5, col = "#9A9A9A"))
        return(invisible())
      }
      grid.rect(x, y, width, height,
                gp = gpar(fill = if (sig[i, j]) col_fun(nes[i, j]) else ns_fill, col = "white", lwd = 1))
      dark <- sig[i, j] && abs(nes[i, j]) > 0.7 * lim
      grid.text(labels[i, j], x, y,
                gp = gpar(fontsize = 6.5, fontface = if (sig[i, j]) "bold" else "plain",
                          col = if (dark) "white" else "black"))
    }

    top_anno <- HeatmapAnnotation(
      "Significant\ngene sets" = anno_barplot(
        cbind(Up = colSums(up), Down = colSums(down)),
        gp = gpar(fill = c(up_col, down_col), col = NA), bar_width = 0.72,
        height = unit(1.7, "cm"), border = FALSE, axis_param = list(gp = gpar(fontsize = 6))
      ),
      annotation_name_gp = gpar(fontsize = 7), annotation_name_side = "left"
    )
    right_anno <- rowAnnotation(
      "Cell types" = anno_barplot(
        cbind(Up = rowSums(up), Down = rowSums(down)),
        gp = gpar(fill = c(up_col, down_col), col = NA), bar_width = 0.72,
        width = unit(1.5, "cm"), border = FALSE, axis_param = list(gp = gpar(fontsize = 6))
      ),
      annotation_name_gp = gpar(fontsize = 7), annotation_name_rot = 0
    )

    ht <- Heatmap(
      nes0, name = "NES", col = col_fun, rect_gp = gpar(type = "none"), cell_fun = cell_fun,
      width = ncol(nes) * unit(10, "mm"), height = nrow(nes) * unit(5.2, "mm"),
      cluster_rows = nrow(nes) > 1, cluster_columns = ncol(nes) > 1,
      clustering_method_rows = "average", clustering_method_columns = "average",
      row_dend_width = unit(1.2, "cm"), column_dend_height = unit(1.0, "cm"),
      row_names_side = "left", row_dend_side = "left",
      row_names_gp = gpar(fontsize = 7, fontface = "bold"),
      column_names_gp = gpar(fontsize = 8, fontface = "bold"), column_names_rot = 45,
      column_title = title, column_title_gp = gpar(fontsize = 12),
      top_annotation = top_anno, right_annotation = right_anno,
      show_heatmap_legend = FALSE
    )

    # One hand-built legend stack, so the colour scale, significance key and bar key sit in a
    # single column (ComplexHeatmap otherwise places heatmap and annotation legends side by side).
    # The key lists only what the figure can show: a "**" entry is dropped when no cell reaches it.
    has_strong <- any(!is.na(padj[tested]) & padj[tested] < strong)
    sig_labels <- c(sprintf("* padj < %g", alpha),
                    if (has_strong) sprintf("** padj < %g", strong),
                    "not significant (grey)", "not testable (n.t.)")
    sig_fill <- c("white", if (has_strong) "white", ns_fill, "white")
    legends <- list(
      Legend(col_fun = col_fun, title = "NES", at = c(-lim, -lim / 2, 0, lim / 2, lim),
             labels = sprintf("%g", c(-lim, -lim / 2, 0, lim / 2, lim))),
      Legend(labels = sig_labels, title = "Significance", type = "grid",
             legend_gp = gpar(fill = sig_fill, col = "#BBBBBB")),
      Legend(labels = c("Up", "Down"), title = "Significant\nenrichment", type = "grid",
             legend_gp = gpar(fill = c(up_col, down_col)))
    )

    n_col <- ncol(nes)
    n_row <- nrow(nes)
    width_in <- 3.4 + 0.39 * n_col + 2.6
    height_in <- 2.6 + 0.205 * n_row + 0.8

    render <- function() {
      draw(ht, annotation_legend_list = legends, annotation_legend_side = "right",
           padding = unit(c(9, 4, 3, 4), "mm"))
      if (nzchar(footnote)) {
        grid.text(footnote, x = unit(4, "mm"), y = unit(3, "mm"), just = c("left", "bottom"),
                  gp = gpar(fontsize = 6.5, col = "#555555"))
      }
    }
    png(out_png, width = width_in, height = height_in, units = "in", res = 300)
    render()
    dev.off()
    pdf(sub("\\.png$", ".pdf", out_png), width = width_in, height = height_in)
    render()
    dev.off()
    cat("gsea_nes_heatmap.R: wrote", out_png, "(", n_row, "gene sets x", n_col, "cell types )\n")
  },
  error = function(e) {
    message("gsea_nes_heatmap.R error: ", conditionMessage(e))
    quit(status = 1)
  }
)
