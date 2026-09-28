#!/usr/bin/env Rscript
# hdWGCNA co-expression modules -- config-driven, biology-free port.
# Usage: Rscript hdwgcna.R <h5ad> <out_dir> <group_by> <condition_col> \
#        <n_hvg> <k> <min_cells> <soft_power|NA> <seed> <fig_dir> <edge_quantile>
# Graceful-skip: on any failure writes header-only modules.csv + hdwgcna_SKIPPED.txt, quit(status=0).

args <- commandArgs(trailingOnly = TRUE)
h5ad_path <- args[[1]]; out_dir <- args[[2]]
group_by <- args[[3]]; condition_col <- args[[4]]
n_hvg <- as.integer(args[[5]]); k <- as.integer(args[[6]])
min_cells <- as.integer(args[[7]])
soft_power <- suppressWarnings(as.integer(args[[8]]))  # NA -> auto
seed <- as.integer(args[[9]])
# figures/, not results/ -- matches the split every other CellQuorum stage
# uses; only this script's own raw-data CSVs (modules/eigengenes/etc.)
# belong in out_dir.
fig_dir <- if (length(args) >= 10) args[[10]] else out_dir
# Quantile of the topological overlap matrix kept as real coexpression
# edges (tom_edges.csv) -- 0.993 matches the published reference this
# feature was built to match (Adams et al. Sci Adv 2020, "GRN
# construction": "edge cutoff of the top 0.993 quantile of correlation
# coefficient").
edge_quantile <- if (length(args) >= 11) as.numeric(args[[11]]) else 0.993
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
out_csv <- file.path(out_dir, "modules.csv")

skip <- function(reason) {
  writeLines("gene,module", out_csv)
  writeLines(sprintf("hdWGCNA skipped: %s", reason), file.path(out_dir, "hdwgcna_SKIPPED.txt"))
  message(sprintf("[hdwgcna] SKIPPED: %s", reason)); quit(status = 0)
}

need <- c("Seurat", "hdWGCNA", "zellkonverter", "SingleCellExperiment", "WGCNA")
missing <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) skip(sprintf("missing R packages: %s", paste(missing, collapse = ", ")))

tryCatch({
  set.seed(seed)
  suppressPackageStartupMessages({
    library(Seurat); library(hdWGCNA); library(zellkonverter); library(SingleCellExperiment)
  })
  sce <- tryCatch(readH5AD(h5ad_path, reader = "R"), error = function(e) readH5AD(h5ad_path))
  for (an in SummarizedExperiment::assayNames(sce)) {
    SummarizedExperiment::assay(sce, an) <-
      methods::as(as(SummarizedExperiment::assay(sce, an), "CsparseMatrix"), "dgCMatrix")
  }
  seu <- as.Seurat(sce, counts = "X", data = NULL)
  seu <- NormalizeData(seu) |> FindVariableFeatures(nfeatures = n_hvg) |>
         ScaleData() |> RunPCA(npcs = 30, verbose = FALSE)
  seu <- SetupForWGCNA(seu, gene_select = "variable", wgcna_name = "cq")
  # Generic grouping fallback: if the requested column is absent, use one constant group.
  if (!(group_by %in% colnames(seu@meta.data))) seu@meta.data[[group_by]] <- "all"
  seu <- MetacellsByGroups(seu, group.by = group_by, ident.group = group_by,
                           reduction = "pca", k = k, max_shared = 10, min_cells = min_cells)
  seu <- NormalizeMetacells(seu)
  seu <- SetDatExpr(seu, group.by = NULL, group_name = NULL)
  seu <- TestSoftPowers(seu)
  if (!is.na(soft_power)) {
    seu <- ConstructNetwork(seu, soft_power = soft_power, tom_name = "cq", overwrite_tom = TRUE)
  } else {
    seu <- ConstructNetwork(seu, tom_name = "cq", overwrite_tom = TRUE)
  }
  seu <- tryCatch(ModuleConnectivity(ModuleEigengenes(seu)), error = function(e) {
    message(sprintf("[hdwgcna] eigengenes/connectivity failed (%s)", conditionMessage(e))); seu })
  seu <- tryCatch(ResetModuleNames(seu, new_name = "M"), error = function(e) seu)

  # Real coexpression edges from the topological overlap matrix -- modules.csv
  # alone carries only per-gene module labels + kME (eigengene correlation),
  # no gene-gene edges at all, so a community-network figure built from it
  # can only fake within-module cliques (module_curvature.py's proxy graph),
  # never the cross-module structure a real force-directed network needs.
  # Thresholded to the top edge_quantile (default 0.993, matching the
  # published reference this feature targets) instead of writing the full
  # TOM: a 3000-HVG TOM has ~4.5M upper-triangle entries, most near-zero.
  tryCatch({
    TOM <- GetTOM(seu)
    genes <- rownames(TOM)
    ut <- upper.tri(TOM)
    vals <- TOM[ut]
    thresh <- stats::quantile(vals, edge_quantile, na.rm = TRUE)
    keep <- ut & (TOM >= thresh)
    idx <- which(keep, arr.ind = TRUE)
    if (nrow(idx) > 0) {
      edge_df <- data.frame(
        gene1 = genes[idx[, 1]],
        gene2 = genes[idx[, 2]],
        weight = TOM[idx]
      )
      write.csv(edge_df, file.path(out_dir, "tom_edges.csv"), row.names = FALSE)
      message(sprintf("[hdwgcna] wrote %d TOM edges (top %.3f quantile, threshold=%.4f)",
                       nrow(edge_df), edge_quantile, thresh))
    } else {
      message("[hdwgcna] TOM edges: no edges survived the quantile threshold")
    }
  }, error = function(e) message(sprintf("[hdwgcna] TOM edges skipped: %s", conditionMessage(e))))

  # Gene dendrogram with the module color bar -- hdWGCNA's own canonical
  # figure for "did the clustering find real modules". ConstructNetwork()
  # above already computed and stored the dendrogram; this just renders it,
  # same graceful-skip-on-failure pattern as every other tryCatch here.
  tryCatch({
    pdf(file.path(fig_dir, "gene_dendrogram.pdf"), width = 10, height = 6)
    PlotDendrogram(seu, main = "hdWGCNA gene dendrogram")
    dev.off()
  }, error = function(e) {
    message(sprintf("[hdwgcna] dendrogram skipped: %s", conditionMessage(e)))
  })

  # Per-module network plots -- hdWGCNA's own canonical "networks" figure
  # (one PDF per module: top hub genes + their strongest connections), a
  # ModuleConnectivity kME map rather than the CellQuorum-side Ollivier-Ricci
  # curvature network module_curvature.py already renders separately.
  # ModuleNetworkPlot writes its own files into outdir; nothing to capture
  # here beyond letting it run.
  tryCatch({
    net_dir <- file.path(fig_dir, "module_networks")
    dir.create(net_dir, showWarnings = FALSE, recursive = TRUE)
    ModuleNetworkPlot(seu, outdir = net_dir)
  }, error = function(e) {
    message(sprintf("[hdwgcna] module network plots skipped: %s", conditionMessage(e)))
  })

  mods <- GetModules(seu)
  keep <- intersect(c("gene_name", "module", "color", grep("^kME_", colnames(mods), value = TRUE)),
                    colnames(mods))
  out_tab <- mods[, keep, drop = FALSE]
  names(out_tab)[names(out_tab) == "gene_name"] <- "gene"
  write.csv(out_tab, out_csv, row.names = FALSE)
  message(sprintf("[hdwgcna] wrote %d gene-module assignments", nrow(out_tab)))

  # Module eigengenes (metacell x module).
  me <- tryCatch(GetMEs(seu), error = function(e) NULL)
  if (!is.null(me)) write.csv(me, file.path(out_dir, "eigengenes.csv"))

  # Module UMAP coordinate table (flagship figure input).
  tryCatch({
    seu <- RunModuleUMAP(seu, n_hubs = 10, n_neighbors = 15, min_dist = 0.1)
    umap_df <- GetModuleUMAP(seu)
    write.csv(umap_df, file.path(out_dir, "module_umap.csv"), row.names = FALSE)
  }, error = function(e) message(sprintf("[hdwgcna] module UMAP skipped: %s", conditionMessage(e))))

  # Module-condition correlation with GENERIC ordinal encoding (no hardcoded biology).
  tryCatch({
    md <- seu@meta.data
    if (!is.null(me) && condition_col %in% colnames(md)) {
      # Aggregate condition per metacell by modal value, encode by sorted level order.
      # (Metacell rownames align with GetMEs rows.)
      lv <- sort(unique(as.character(md[[condition_col]])))
      enc <- setNames(seq_along(lv) - 1, lv)
      # Map each metacell to its modal condition via hdWGCNA metacell obj if available; else skip.
      mc <- tryCatch(GetMetacellObject(seu), error = function(e) NULL)
      if (!is.null(mc) && condition_col %in% colnames(mc@meta.data)) {
        cond_num <- enc[as.character(mc@meta.data[[condition_col]])]
        # Index by metacell ID so cond_num[common] aligns with the eigengene rows
        # (enc's names are condition LEVELS, not metacell barcodes).
        names(cond_num) <- rownames(mc@meta.data)
        common <- intersect(rownames(me), rownames(mc@meta.data))
        if (length(common) >= 5) {
          cors <- sapply(colnames(me), function(m)
            suppressWarnings(cor(me[common, m], cond_num[common], use = "complete.obs")))
          write.csv(data.frame(module = names(cors), correlation = as.numeric(cors)),
                    file.path(out_dir, "module_condition_corr.csv"), row.names = FALSE)
        }
      }
    }
  }, error = function(e) message(sprintf("[hdwgcna] condition corr skipped: %s", conditionMessage(e))))

}, error = function(e) skip(sprintf("runtime error: %s", conditionMessage(e))))
