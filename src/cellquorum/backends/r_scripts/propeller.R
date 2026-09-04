# Speckle propeller proportion test for cell-type differential abundance.
#
# Usage: Rscript propeller.R <counts.csv> <meta.csv> <out.csv> \
#        <condition_col> <case> <control> <transform> [donor_col] [paired]
#
# counts.csv: first column 'sample', remaining columns cell types (samples x celltypes, integer counts).
# meta.csv:   first column sample id, columns include condition_col (and donor_col when paired).
# transform:  "asin" (default) or "logit" for transformed proportions.
# donor_col:  column in meta.csv holding the donor/subject id. Only read when paired.
# paired:     "TRUE" fits the donor block; anything else (or absent) fits arms only.
#
# ---------------------------------------------------------------------------
# Why the design is a means model
# ---------------------------------------------------------------------------
# speckle's propeller.ttest builds its PropRatio column as prod(coef ^ contrast)
# over the fitted per-arm coefficients. That is a RATIO only when those
# coefficients are the two arm means -- a means model (~ 0 + grp) contrasted
# c(-1, 1). Under R's default treatment contrasts (~ grp, contrast c(0, 1)) the
# coefficients are (control mean, difference) and the same arithmetic returns
# control_mean^0 * difference^1 = the raw DIFFERENCE, reported under a column
# named PropRatio. So this script always fits the means model: the reported ratio
# is a ratio, and the two arm means come out of the same fit instead of being
# discarded.
#
# ---------------------------------------------------------------------------
# Why the donor block matters
# ---------------------------------------------------------------------------
# When the cohort is matched, every sample's proportion carries its donor's
# baseline. Fitting arms only leaves that between-donor variance in the residual,
# so the moderated t divides the arm difference by the wrong spread and a real
# within-donor shift reads as noise. On a nine-donor matched cohort in this
# project that difference was the whole result: arms-only put nothing below FDR
# 0.39, while the same proportions tested within donor cleared FDR 0.031.
#
# The caller decides whether the design is paired, because the caller is the one
# that knows which donors span both arms and can check the design is estimable
# before spending an R subprocess on it. This script re-checks the rank and stops
# with a message rather than fitting a rank-deficient design and returning
# coefficients that are silently NA.
suppressPackageStartupMessages({ library(speckle); library(limma) })

args <- commandArgs(trailingOnly = TRUE)
counts_csv <- args[1]; meta_csv <- args[2]; out_csv <- args[3]
condition_col <- args[4]; case <- args[5]; control <- args[6]; transform <- args[7]
donor_col <- if (length(args) >= 8) args[8] else ""
paired <- length(args) >= 9 && toupper(args[9]) == "TRUE"

# Read aggregated counts (samples x celltypes).
counts <- read.csv(counts_csv, check.names = FALSE, stringsAsFactors = FALSE)
rownames(counts) <- counts[["sample"]]; counts[["sample"]] <- NULL
counts <- as.matrix(counts)
storage.mode(counts) <- "integer"

# Read sample metadata and align to count rows.
meta <- read.csv(meta_csv, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
meta <- meta[rownames(counts), , drop = FALSE]

# Restrict to the two condition levels.
keep <- meta[[condition_col]] %in% c(case, control)
counts <- counts[keep, , drop = FALSE]
meta <- meta[keep, , drop = FALSE]

if (paired && !(donor_col %in% colnames(meta))) {
  stop(sprintf("paired fit requested but donor column '%s' is not in the metadata", donor_col))
}

# getTransformedProps expects long-form per-cell vectors. Expand the count matrix
# by index rather than by appending inside a double loop: the loop reallocated the
# growing vector once per (sample, cell type) cell, which scales with the total
# number of cells for no reason.
idx <- which(counts > 0, arr.ind = TRUE)
clusters_vec <- rep(colnames(counts)[idx[, "col"]], times = counts[idx])
sample_vec <- rep(rownames(counts)[idx[, "row"]], times = counts[idx])

# Get transformed proportions (returns celltypes x samples matrices).
tp <- getTransformedProps(clusters = clusters_vec, sample = sample_vec, transform = transform)

# Build the design in the column order tp returns its samples in.
sample_ids <- colnames(tp$Proportions)
grp <- factor(meta[sample_ids, condition_col], levels = c(control, case))

if (paired) {
  donor <- factor(meta[sample_ids, donor_col])
  design <- model.matrix(~ 0 + grp + donor)
} else {
  design <- model.matrix(~ 0 + grp)
}

# The contrast is case minus control over the two arm-mean columns, located by
# name so that adding blocking terms cannot shift it onto a nuisance coefficient.
grp_cols <- paste0("grp", c(control, case))
missing_cols <- setdiff(grp_cols, colnames(design))
if (length(missing_cols) > 0) {
  stop(sprintf("design is missing arm column(s): %s", paste(missing_cols, collapse = ", ")))
}
contrast_vec <- rep(0, ncol(design))
contrast_vec[match(grp_cols[1], colnames(design))] <- -1
contrast_vec[match(grp_cols[2], colnames(design))] <- 1

if (qr(design)$rank < ncol(design)) {
  stop(sprintf(
    "design is rank-deficient (rank %d < %d columns); the requested %s fit is not estimable",
    qr(design)$rank, ncol(design), if (paired) "paired" else "unpaired"
  ))
}

res <- propeller.ttest(
  prop.list = tp,
  design = design,
  contrasts = contrast_vec,
  robust = TRUE,
  trend = FALSE,
  sort = TRUE
)

# Both of propeller.ttest's branches fit the arm means on the two contrasted
# columns, so the PropMean columns are named for the arm levels either way. The
# name is looked up through make.names as well, because propeller.ttest builds its
# output with data.frame(), which rewrites a condition label that is not a
# syntactic R name (a hyphen or a space in "Lymphedema-LE") while model.matrix
# leaves the design column alone.
arm_mean <- function(level) {
  wanted <- paste0("PropMean.grp", level)
  for (column in c(wanted, make.names(wanted))) {
    if (column %in% colnames(res)) return(res[[column]])
  }
  rep(NA_real_, nrow(res))
}
control_mean <- arm_mean(control)
case_mean <- arm_mean(case)

# Emit the contract columns (cell_type, PropRatio, Tstatistic, PValue, FDR) plus
# the arm means the fit already produced, so the table states the magnitudes it
# ranks, and the design actually fitted, so a null is readable as a null rather
# than as an arms-only fit on a matched cohort.
# Note: speckle returns P.Value (with dot); rename to PValue.
out <- data.frame(
  cell_type = rownames(res),
  control_mean_prop = control_mean,
  case_mean_prop = case_mean,
  effect_pp = (case_mean - control_mean) * 100,
  PropRatio = res$PropRatio,
  Tstatistic = res$Tstatistic,
  PValue = res$P.Value,
  FDR = res$FDR,
  paired = paired,
  n_donors_blocked = if (paired) length(levels(donor)) else 0L,
  row.names = NULL
)
write.csv(out, out_csv, row.names = FALSE)
