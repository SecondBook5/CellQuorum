# Milo neighborhood-level differential-abundance test.
# Usage: Rscript milo.R <rep.csv> <meta.csv> <out.csv> \
#        <condition_col> <case> <control> <donor_col> <k> <prop> [celltype_col] [paired]
# rep.csv:  first column 'cell' id, remaining columns reduced-dim embedding (cells x dims).
# meta.csv: first column cell id, columns include condition_col, donor_col, optional cell_type.
# out.csv:  nhood, logFC, PValue, SpatialFDR, nhood_size, majority_celltype, celltype_fraction.
# paired:   "true" or "false" (default "false") — use ~ donor + condition (paired/blocked design).
suppressPackageStartupMessages({ library(miloR); library(SingleCellExperiment); library(edgeR) })

args <- commandArgs(trailingOnly = TRUE)
rep_csv <- args[1]; meta_csv <- args[2]; out_csv <- args[3]
condition_col <- args[4]; case <- args[5]; control <- args[6]; donor_col <- args[7]
k <- as.integer(args[8]); prop <- as.numeric(args[9])
celltype_col <- if (length(args) >= 10 && nzchar(args[10])) args[10] else NA
paired <- length(args) >= 11 && tolower(args[11]) == "true"

# Read embedding (cells x dims) and metadata (cells x columns).
emb <- read.csv(rep_csv, row.names = 1, check.names = FALSE)          # cells x dims
meta <- read.csv(meta_csv, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
meta <- meta[rownames(emb), , drop = FALSE]

# Validate input columns exist.
if (!condition_col %in% colnames(meta)) {
  stop("condition_col '", condition_col, "' not found in meta.csv columns: ",
       paste(colnames(meta), collapse = ", "))
}
if (!donor_col %in% colnames(meta)) {
  stop("donor_col '", donor_col, "' not found in meta.csv columns: ",
       paste(colnames(meta), collapse = ", "))
}
if (!is.na(celltype_col) && !celltype_col %in% colnames(meta)) {
  stop("celltype_col '", celltype_col, "' not found in meta.csv columns: ",
       paste(colnames(meta), collapse = ", "))
}

# Restrict to case/control levels.
keep <- meta[[condition_col]] %in% c(case, control)
emb <- emb[keep, , drop = FALSE]
meta <- meta[keep, , drop = FALSE]

# Validate both case and control are present after filtering.
present_levels <- unique(meta[[condition_col]])
if (!case %in% present_levels) {
  stop("case level '", case, "' not found in meta[[", condition_col, "]] after filtering. ",
       "Present levels: ", paste(present_levels, collapse = ", "))
}
if (!control %in% present_levels) {
  stop("control level '", control, "' not found in meta[[", condition_col, "]] after filtering. ",
       "Present levels: ", paste(present_levels, collapse = ", "))
}

# Validate non-empty data after filtering.
if (nrow(meta) == 0) {
  stop("No cells remaining after filtering for case='", case, "' and control='", control, "'")
}

# Build SingleCellExperiment with placeholder assay and reduced-dim embedding.
# Attach cell_type to colData if provided so annotateNhoods can read it.
n_cells <- nrow(emb)
assays_list <- list(logcounts = matrix(0, nrow = 1, ncol = n_cells))
colnames(assays_list$logcounts) <- rownames(emb)

if (!is.na(celltype_col)) {
  # Attach cell_type to colData so annotateNhoods can find it
  coldata <- meta[, celltype_col, drop = FALSE]
  sce <- SingleCellExperiment(assays = assays_list, colData = coldata)
} else {
  sce <- SingleCellExperiment(assays = assays_list)
}
reducedDim(sce, "PCA") <- as.matrix(emb)

# Build Milo object and construct neighborhood graph.
milo <- Milo(sce)
d <- ncol(emb)

# Set seed for deterministic neighborhood sampling (makeNhoods is stochastic).
set.seed(0)

milo <- buildGraph(milo, k = k, d = d, reduced.dim = "PCA")
milo <- makeNhoods(milo, prop = prop, k = k, d = d, refined = TRUE, reduced_dims = "PCA")

# Determine sample column for countCells (donor vs donor-condition pair).
# For paired designs, samples = unique donor-condition combinations.
# For unpaired designs, samples = unique donors.
if (paired) {
  meta$sample_id <- paste(meta[[donor_col]], meta[[condition_col]], sep = "__")
  sample_col <- "sample_id"
} else {
  sample_col <- donor_col
}

# Count cells per neighborhood, aggregated by sample.
milo <- countCells(milo, samples = sample_col, meta.data = meta)

# Calculate neighborhood distances for spatial FDR weighting (matches validated pipeline).
# testNhoods default fdr.weighting="k-distance" requires these distances for correct SpatialFDR.
milo <- calcNhoodDistance(milo, d = d, reduced.dim = "PCA")

# Build per-sample design.df: one row per sample.
# Must reorder to match colnames(nhoodCounts(milo)).
if (paired) {
  # Paired design: ~ donor + condition (donor-blocked).
  # Extract unique sample_id, donor, condition combinations from meta.
  sm <- unique(meta[, c("sample_id", donor_col, condition_col), drop = FALSE])
  rownames(sm) <- sm$sample_id
  sm[[donor_col]] <- factor(sm[[donor_col]])
  sm[[condition_col]] <- factor(sm[[condition_col]], levels = c(control, case))
  design_df <- sm[colnames(nhoodCounts(milo)), c(donor_col, condition_col), drop = FALSE]
  design_formula <- as.formula(paste("~", donor_col, "+", condition_col))
} else {
  # Unpaired design: ~ condition (no donor blocking).
  # Rownames = donor IDs (assuming each donor appears in only one condition).
  sm <- unique(meta[, c(donor_col, condition_col), drop = FALSE])
  rownames(sm) <- sm[[donor_col]]
  sm[[condition_col]] <- factor(sm[[condition_col]], levels = c(control, case))
  design_df <- sm[colnames(nhoodCounts(milo)), , drop = FALSE]
  design_formula <- as.formula(paste("~", condition_col))
}

# Run testNhoods with the design formula and per-sample data.
# Wrap in tryCatch to ensure rank-deficient or other testNhoods failures exit non-zero.
res <- tryCatch(
  {
    testNhoods(
      milo,
      design = design_formula,
      design.df = design_df,
      reduced.dim = "PCA"
    )
  },
  error = function(e) {
    message("testNhoods failed: ", conditionMessage(e))
    quit(status = 1)
  }
)

# Compute nhood_size from the nhoods matrix (testNhoods output does NOT include it).
nhood_sizes <- colSums(nhoods(milo))
res$nhood_size <- nhood_sizes[res$Nhood]

# Annotate neighborhoods with majority cell type if celltype_col is provided.
if (!is.na(celltype_col)) {
  # annotateNhoods returns the data.frame with added columns: <celltype_col> and <celltype_col>_fraction
  res <- annotateNhoods(milo, res, coldata_col = celltype_col)
  # Rename annotation columns to the contract names
  res$majority_celltype <- res[[celltype_col]]
  res$celltype_fraction <- res[[paste0(celltype_col, "_fraction")]]
} else {
  res$majority_celltype <- NA
  res$celltype_fraction <- NA
}

# Emit the output CSV with the contract columns.
out <- data.frame(
  nhood = res$Nhood,
  logFC = res$logFC,
  PValue = res$PValue,
  SpatialFDR = res$SpatialFDR,
  nhood_size = res$nhood_size,
  majority_celltype = res$majority_celltype,
  celltype_fraction = res$celltype_fraction,
  row.names = NULL
)
write.csv(out, out_csv, row.names = FALSE)
