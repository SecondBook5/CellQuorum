# Pseudobulk edgeR quasi-likelihood DE fit.
# Usage: Rscript edger.R <counts.csv> <meta.csv> <out.csv> \
#        <condition_col> <case> <control> <design_rhs> <min_count> \
#        <min_total_count> [test_coef]
# counts.csv: first column 'sample', remaining columns genes (pseudo-samples x genes).
# meta.csv:   first column sample id, columns include condition_col (+ donor/covars).
# test_coef (optional): "" (default) tests the case-vs-control condition
#   coefficient; ":interaction" jointly tests every interaction coefficient (a
#   difference-of-differences F-test); otherwise a comma-separated list of exact
#   design-column names to test jointly.
suppressPackageStartupMessages({ library(edgeR) })

args <- commandArgs(trailingOnly = TRUE)
counts_csv <- args[1]; meta_csv <- args[2]; out_csv <- args[3]
condition_col <- args[4]; case <- args[5]; control <- args[6]
design_rhs <- args[7]
min_count <- as.integer(args[8]); min_total_count <- as.integer(args[9])
test_coef <- if (length(args) >= 10) args[10] else ""
if (is.na(test_coef)) test_coef <- ""

# Read pseudobulk counts (samples x genes) and transpose to genes x samples.
counts <- read.csv(counts_csv, check.names = FALSE, stringsAsFactors = FALSE)
rownames(counts) <- counts[["sample"]]; counts[["sample"]] <- NULL
counts <- t(as.matrix(counts))               # genes x samples
storage.mode(counts) <- "integer"

# Read sample metadata and align to the count columns.
meta <- read.csv(meta_csv, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
meta <- meta[colnames(counts), , drop = FALSE]

# Restrict to the two condition levels and set control as the reference.
keep_samples <- meta[[condition_col]] %in% c(case, control)
counts <- counts[, keep_samples, drop = FALSE]
meta <- meta[keep_samples, , drop = FALSE]
meta[[condition_col]] <- factor(meta[[condition_col]], levels = c(control, case))

# Build the design matrix from the requested right-hand side.
for (col in setdiff(all.vars(as.formula(paste("~", design_rhs))), condition_col)) {
  if (!is.numeric(meta[[col]])) {
    meta[[col]] <- factor(meta[[col]])
  }
}
design <- model.matrix(as.formula(paste("~", design_rhs)), data = meta)

# Standard edgeR quasi-likelihood pipeline.
dge <- DGEList(counts = counts)
keep <- filterByExpr(dge, design = design,
                     min.count = min_count, min.total.count = min_total_count)
dge <- dge[keep, , keep.lib.sizes = FALSE]
dge <- calcNormFactors(dge)
dge <- estimateDisp(dge, design)
fit <- glmQLFit(dge, design)

# Select the coefficient(s) to test. Default: the case-vs-control condition
# coefficient (control is the reference level). ":interaction": every interaction
# column (jointly, an F-test). Otherwise: the named columns, comma-separated.
if (nzchar(test_coef) && identical(test_coef, ":interaction")) {
  coef_idx <- grep(":", colnames(design), fixed = TRUE)
  if (length(coef_idx) == 0L) {
    stop("edgeR DE: no interaction columns in design: ",
         paste(colnames(design), collapse = ", "))
  }
} else if (nzchar(test_coef)) {
  wanted <- trimws(strsplit(test_coef, ",", fixed = TRUE)[[1]])
  coef_idx <- match(wanted, colnames(design))
  if (anyNA(coef_idx)) {
    stop("edgeR DE: coefficient(s) not found: ",
         paste(wanted[is.na(coef_idx)], collapse = ", "),
         " | design columns: ", paste(colnames(design), collapse = ", "))
  }
} else {
  coef_name <- paste0(condition_col, case)
  coef_idx <- match(coef_name, colnames(design))
  if (is.na(coef_idx)) {
    stop("edgeR DE: condition coefficient '", coef_name,
         "' not found in design columns: ",
         paste(colnames(design), collapse = ", "))
  }
}
qlf <- glmQLFTest(fit, coef = coef_idx)

# Write the full DE table sorted by significance. A single tested coefficient
# yields a signed logFC; a joint (multi-coefficient) interaction test has no
# single fold-change, so logFC is NA there and the F/PValue/FDR carry the test.
tt <- topTags(qlf, n = Inf, sort.by = "PValue")$table
lfc <- if ("logFC" %in% colnames(tt)) tt$logFC else NA_real_
out <- data.frame(
  gene = rownames(tt), logFC = lfc, logCPM = tt$logCPM,
  F = tt$F, PValue = tt$PValue, FDR = tt$FDR, row.names = NULL
)
write.csv(out, out_csv, row.names = FALSE)
