# Pseudobulk edgeR quasi-likelihood DE fit.
# Usage: Rscript edger.R <counts.csv> <meta.csv> <out.csv> \
#        <condition_col> <case> <control> <design_rhs> <min_count> <min_total_count>
# counts.csv: first column 'sample', remaining columns genes (pseudo-samples x genes).
# meta.csv:   first column sample id, columns include condition_col (+ donor/covars).
suppressPackageStartupMessages({ library(edgeR) })

args <- commandArgs(trailingOnly = TRUE)
counts_csv <- args[1]; meta_csv <- args[2]; out_csv <- args[3]
condition_col <- args[4]; case <- args[5]; control <- args[6]
design_rhs <- args[7]
min_count <- as.integer(args[8]); min_total_count <- as.integer(args[9])

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
  meta[[col]] <- factor(meta[[col]])
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

# The condition coefficient is the last column (case vs control).
coef_name <- paste0(condition_col, case)
coef_idx <- match(coef_name, colnames(design))
if (is.na(coef_idx)) coef_idx <- ncol(design)
qlf <- glmQLFTest(fit, coef = coef_idx)

# Write the full DE table sorted by significance.
tt <- topTags(qlf, n = Inf, sort.by = "PValue")$table
out <- data.frame(
  gene = rownames(tt), logFC = tt$logFC, logCPM = tt$logCPM,
  F = tt$F, PValue = tt$PValue, FDR = tt$FDR, row.names = NULL
)
write.csv(out, out_csv, row.names = FALSE)
