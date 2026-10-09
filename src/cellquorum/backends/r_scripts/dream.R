# Specimen-level pseudobulk mixed-model DE: edgeR prefilter/normalisation ->
# variancePartition::voomWithDreamWeights -> dream (explicit contrast matrix L) -> eBayes.
# Usage: Rscript dream.R <counts.csv> <meta.csv> <L.csv> <formula> <out_dir> <ddf> \
#                        <min_count> <min_total_count> <n_workers>
# n_workers: 1 = deterministic SerialParam; >1 = BiocParallel workers for the gene fits
# (MulticoreParam on Unix, SnowParam on Windows). Estimates do not depend on it.
# counts.csv: first column 'sample', remaining columns genes (specimens x genes).
# meta.csv:   first column sample id; numeric fixed-effect columns named cq_fx*, random-
#             effect grouping columns named cq_re*. CellQuorum's Python encoder owns the
#             factor coding; R never re-codes a factor.
# L.csv:      first column 'coef' ('(Intercept)', cq_fx*), remaining columns = contrasts.
# formula:    machine formula over cq_fx*/cq_re* names, e.g. ~ cq_fx1 + cq_fx2 + (1 | cq_re1).
# Writes into out_dir: results_long.csv, fit_info.json and (when any gene fails)
# failed_genes.csv. Exit status 0 with fit_info$status != "ok" for the handled outcomes
# "no_genes_after_filter" / "all_genes_failed"; any other failure is a non-zero exit.
suppressPackageStartupMessages({
  library(edgeR); library(variancePartition); library(BiocParallel); library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
counts_csv <- args[1]; meta_csv <- args[2]; l_csv <- args[3]
formula_str <- args[4]; out_dir <- args[5]; ddf <- args[6]
min_count <- as.integer(args[7]); min_total_count <- as.integer(args[8])
n_workers <- max(1L, as.integer(args[9]))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

counts <- read.csv(counts_csv, check.names = FALSE, stringsAsFactors = FALSE)
rownames(counts) <- counts[["sample"]]; counts[["sample"]] <- NULL
counts <- t(as.matrix(counts)); storage.mode(counts) <- "integer"      # genes x specimens
meta <- read.csv(meta_csv, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
meta <- meta[colnames(counts), , drop = FALSE]
for (col in grep("^cq_re", colnames(meta), value = TRUE)) meta[[col]] <- factor(meta[[col]])
L <- as.matrix(read.csv(l_csv, row.names = 1, check.names = FALSE))
form <- as.formula(formula_str)

if (n_workers <= 1L) {
  bpp <- SerialParam()
} else if (.Platform$OS.type == "windows") {
  bpp <- SnowParam(workers = n_workers, progressbar = FALSE)
} else {
  bpp <- MulticoreParam(workers = n_workers, progressbar = FALSE)
}

info <- list(
  status = "ok",
  variancePartition = as.character(packageVersion("variancePartition")),
  edgeR = as.character(packageVersion("edgeR")),
  limma = as.character(packageVersion("limma")),
  R = R.version.string,
  formula = formula_str, ddf_requested = ddf,
  normalization = "TMM (edgeR::calcNormFactors)",
  filterByExpr = list(min.count = min_count, min.total.count = min_total_count),
  n_specimens = ncol(counts), n_genes_before_filter = nrow(counts),
  bpparam = list(class = class(bpp)[1], workers = bpnworkers(bpp))
)

# Fixed-effect design for filterByExpr: the COMPLETE fixed part of the fitted model --
# explicit intercept plus every cq_fx column (the random intercept is deliberately not in
# this matrix). It is written out and its columns/rank recorded so this is auditable.
fx <- grep("^cq_fx", colnames(meta), value = TRUE)
design <- model.matrix(as.formula(paste("~", paste(c("1", fx), collapse = " + "))), meta)
write.csv(design, file.path(out_dir, "filter_design.csv"))
info$filter_design_columns <- colnames(design)
info$filter_design_rank <- qr(design)$rank
dge <- DGEList(counts = counts)
keep <- filterByExpr(dge, design = design, min.count = min_count, min.total.count = min_total_count)
info$n_genes_after_filter <- sum(keep)
finish <- function(info) {
  writeLines(toJSON(info, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null"),
             file.path(out_dir, "fit_info.json"))
}
if (sum(keep) == 0L) { info$status <- "no_genes_after_filter"; finish(info); quit(status = 0) }
dge <- calcNormFactors(dge[keep, , keep.lib.sizes = FALSE])

caught <- character()
collect <- function(expr) withCallingHandlers(expr, warning = function(w) {
  caught <<- c(caught, conditionMessage(w)); invokeRestart("muffleWarning")
})
fit_start <- Sys.time()
v <- collect(voomWithDreamWeights(dge, form, meta, BPPARAM = bpp, plot = FALSE,
                                  normalize.method = "none"))
fit <- collect(dream(v, form, meta, L, ddf = ddf, BPPARAM = bpp))
info$fit_seconds <- as.numeric(difftime(Sys.time(), fit_start, units = "secs"))
errs <- attr(fit, "errors")        # named character vector: gene -> error text
if (!is.null(errs) && length(errs)) {
  write.csv(data.frame(gene = names(errs), error = trimws(unname(errs))),
            file.path(out_dir, "failed_genes.csv"), row.names = FALSE)
}
info$n_genes_failed <- if (is.null(errs)) 0L else length(errs)
if (nrow(fit$coefficients) == 0L) { info$status <- "all_genes_failed"; finish(info); quit(status = 0) }

fit <- eBayes(fit)
info$dream_fit_method <- fit$method
info$df_prior <- fit$df.prior

# One row per gene x contrast. standard_error is the model-based SE sqrt(L' Cov L)
# (= stdev.unscaled * sigma); moderated_standard_error is the SE the moderated t uses
# (= stdev.unscaled * sqrt(s2.post) = logFC / t). df_residual is the per-gene, per-
# contrast denominator df from the chosen ddf method; df_total adds the EB prior df.
rows <- lapply(colnames(L), function(cn) {
  tt <- topTable(fit, coef = cn, number = Inf, sort.by = "none", adjust.method = "BH")
  genes <- rownames(tt)
  data.frame(
    gene = genes, contrast = cn, logFC = tt$logFC, AveExpr = tt$AveExpr,
    standard_error = fit$stdev.unscaled[genes, cn] * fit$sigma[genes],
    moderated_standard_error = fit$stdev.unscaled[genes, cn] * sqrt(fit$s2.post[genes]),
    t = tt$t, df_residual = fit$df.residual[genes, cn], df_total = fit$df.total[genes, cn],
    PValue = tt$P.Value, FDR = tt$adj.P.Val, row.names = NULL
  )
})
write.csv(do.call(rbind, rows), file.path(out_dir, "results_long.csv"), row.names = FALSE)

# Benign, expected warning for single-coefficient contrasts is recorded, not hidden.
info$r_warnings <- unique(caught)
finish(info)
