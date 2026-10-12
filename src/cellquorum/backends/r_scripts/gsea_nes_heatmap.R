# NES-by-cell-type heatmap (GSEA): gene sets x cell types, drawn with ComplexHeatmap.
# Every tested cell carries its NES colour. Significance (padj < alpha) is shown by full
# saturation, a dark outline, bold text and stars (* < alpha, ** < strong); a cell that is not
# significant keeps its colour, washed toward white, so a near-miss is not mistaken for no effect.
# A gene set a cell type could not test is white and marked "n.t.", never drawn as zero.
# Rows and columns are clustered on NES (average linkage) and the rows are cut into blocks; count
# bars show cells per cell type (log scale from 100) and how many gene sets are significant up and down, and how many
# cell types per gene set.
# Usage: Rscript gsea_nes_heatmap.R <nes.csv> <padj.csv> <out.png> <alpha> <strong> <title>
#          <footnote> [palette] [n_cells.csv]
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
palette_name <- if (length(args) >= 8 && nzchar(args[8])) args[8] else "teal_orange"
n_cells_csv <- if (length(args) >= 9) args[9] else ""
group_plural <- if (length(args) >= 10 && nzchar(args[10])) args[10] else "Cell types"
group_plural <- paste0(toupper(substring(group_plural, 1, 1)), substring(group_plural, 2))

PALETTES <- list(
  teal_orange = c(low = "#12768A", mid = "#FAF6EE", high = "#E4571C"),
  coolwarm = c(low = "#3B4CC0", mid = "#F2F2F2", high = "#B40426"),
  rdbu = c(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B")
)

# Blend a colour toward white (share = how much of the colour is kept).
fade <- function(col, share) {
  rgb_vals <- col2rgb(col)
  mixed <- rgb_vals * share + 255 * (1 - share)
  rgb(mixed[1, ], mixed[2, ], mixed[3, ], maxColorValue = 255)
}

tryCatch(
  {
    pal <- PALETTES[[palette_name]]
    if (is.null(pal)) stop("unknown palette: ", palette_name)
    nes <- as.matrix(read.csv(nes_csv, row.names = 1, check.names = FALSE))
    padj <- as.matrix(read.csv(padj_csv, row.names = 1, check.names = FALSE))
    tested <- !is.na(nes)
    sig <- tested & !is.na(padj) & padj < alpha
    up <- sig & nes > 0
    down <- sig & nes < 0

    # Symmetric about 0 so the two colours are comparable; a missing cell is 0 only for
    # clustering and is drawn as "n.t." below.
    lim <- ceiling(max(abs(nes[tested]), na.rm = TRUE) * 2) / 2
    if (!is.finite(lim) || lim < 2) lim <- 2
    col_fun <- colorRamp2(c(-lim, 0, lim), c(pal[["low"]], pal[["mid"]], pal[["high"]]))
    nes0 <- nes
    nes0[!tested] <- 0

    stars <- matrix(ifelse(!tested | is.na(padj), "",
                           ifelse(padj < strong, "**", ifelse(padj < alpha, "*", ""))),
                    nrow = nrow(nes))
    labels <- matrix(ifelse(tested, sprintf("%.1f%s", nes, stars), "n.t."), nrow = nrow(nes))

    cell_fun <- function(j, i, x, y, width, height, fill) {
      if (!tested[i, j]) {
        grid.rect(x, y, width, height, gp = gpar(fill = "white", col = "white", lwd = 1))
        grid.text("n.t.", x, y, gp = gpar(fontsize = 5.5, col = "#9A9A9A"))
        return(invisible())
      }
      base <- col_fun(nes[i, j])
      if (sig[i, j]) {
        grid.rect(x, y, width, height, gp = gpar(fill = base, col = "#2B2B2B", lwd = 0.9))
        dark <- abs(nes[i, j]) > 0.62 * lim
        grid.text(labels[i, j], x, y,
                  gp = gpar(fontsize = 6.5, fontface = "bold", col = if (dark) "white" else "black"))
      } else {
        grid.rect(x, y, width, height, gp = gpar(fill = fade(base, 0.55), col = "white", lwd = 1))
        grid.text(labels[i, j], x, y, gp = gpar(fontsize = 6.2, col = "#5F5F5F"))
      }
    }

    bar_cols <- c(up = pal[["high"]], down = pal[["low"]])
    sig_bars <- anno_barplot(
      cbind(Up = colSums(up), Down = colSums(down)),
      gp = gpar(fill = unname(bar_cols), col = NA), bar_width = 0.72,
      height = unit(1.6, "cm"), border = FALSE, axis_param = list(gp = gpar(fontsize = 6))
    )
    anno_list <- list("Significant\ngene sets" = sig_bars)
    if (nzchar(n_cells_csv) && file.exists(n_cells_csv)) {
      n_cells <- read.csv(n_cells_csv, row.names = 1, check.names = FALSE)[colnames(nes), "n_cells"]
      anno_list <- c(list("Cells" = anno_barplot(
        log10(pmax(n_cells, 100)), ylim = c(2, 5), baseline = 2,
        gp = gpar(fill = "#B8A07E", col = NA), bar_width = 0.72, height = unit(1.2, "cm"),
        border = FALSE,
        axis_param = list(at = log10(c(100, 1000, 10000, 100000)),
                          labels = c("100", "1k", "10k", "100k"), gp = gpar(fontsize = 6))
      )), anno_list)
    }
    top_anno <- do.call(HeatmapAnnotation, c(
      anno_list, list(annotation_name_gp = gpar(fontsize = 7), annotation_name_side = "left")
    ))
    right_bar <- anno_barplot(
      cbind(Up = rowSums(up), Down = rowSums(down)),
      gp = gpar(fill = unname(bar_cols), col = NA), bar_width = 0.72,
      width = unit(1.5, "cm"), border = FALSE, axis_param = list(gp = gpar(fontsize = 6)),
      which = "row"
    )
    right_anno <- do.call(rowAnnotation, c(
      setNames(list(right_bar), group_plural),
      list(annotation_name_gp = gpar(fontsize = 7), annotation_name_rot = 0)
    ))

    n_blocks <- if (nrow(nes) >= 12) 4 else NULL
    ht <- Heatmap(
      nes0, name = "NES", col = col_fun, rect_gp = gpar(type = "none"), cell_fun = cell_fun,
      width = ncol(nes) * unit(10, "mm"), height = nrow(nes) * unit(5.2, "mm"),
      cluster_rows = nrow(nes) > 1, cluster_columns = ncol(nes) > 1,
      clustering_method_rows = "average", clustering_method_columns = "average",
      row_split = n_blocks, row_gap = unit(2.2, "mm"), row_title = NULL,
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
    has_strong <- any(!is.na(padj[tested]) & padj[tested] < strong)
    sample_col <- col_fun(0.75 * lim)
    sig_labels <- c(sprintf("* padj < %g (bold, outlined)", alpha),
                    if (has_strong) sprintf("** padj < %g", strong),
                    "not significant (faded)", "not testable (n.t.)")
    sig_fill <- c(sample_col, if (has_strong) sample_col, fade(sample_col, 0.55), "white")
    sig_border <- c("#2B2B2B", if (has_strong) "#2B2B2B", "#BBBBBB", "#BBBBBB")
    legends <- list(
      Legend(col_fun = col_fun, title = "NES", at = c(-lim, -lim / 2, 0, lim / 2, lim),
             labels = sprintf("%g", c(-lim, -lim / 2, 0, lim / 2, lim))),
      Legend(labels = sig_labels, title = "Significance", type = "grid",
             legend_gp = gpar(fill = sig_fill, col = sig_border)),
      Legend(labels = c("Up", "Down"), title = "Significant\nenrichment", type = "grid",
             legend_gp = gpar(fill = unname(bar_cols)))
    )

    n_col <- ncol(nes)
    n_row <- nrow(nes)
    width_in <- 3.4 + 0.39 * n_col + 2.9
    height_in <- 2.7 + 0.205 * n_row + 0.8 + (if (is.null(n_blocks)) 0 else 0.35)

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
    cat("gsea_nes_heatmap.R: wrote", out_png, "(", n_row, "gene sets x", n_col, tolower(group_plural), ")\n")
  },
  error = function(e) {
    message("gsea_nes_heatmap.R error: ", conditionMessage(e))
    quit(status = 1)
  }
)
