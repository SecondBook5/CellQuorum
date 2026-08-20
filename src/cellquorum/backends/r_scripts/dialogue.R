# DIALOGUE multicellular-program (MCP) detection (Jerby-Arnon & Regev).
# Usage: Rscript dialogue.R <scratch_dir> <out_dir> <k> <n_program_genes> \
#        <seed> <pheno_col_or_NA> <abn_c>
#
# <scratch_dir>: contains celltypes.json = {stripped_name: {"label": original, "dir": subdir}}
#   and one subdir per cell type, each holding:
#     expr.mtx   MatrixMarket genes x cells (rows=genes, cols=cells)
#     genes.txt  one gene id per line  -> rownames(tpm)
#     cells.txt  one cell id per line  -> colnames(tpm)
#     X.csv      cells x features, first column 'cell'
#     meta.csv   first column 'cell'; columns include 'sample','cellQ',
#                optional phenotype column + confounders
# <out_dir>: receives mcp_gene_programs.csv, mcp_scores.csv, mcp_associations.csv, run_meta.json.
#   cell_type columns carry the ORIGINAL label (stripped->original via celltypes.json), because
#   make.cell.type() strips underscores from names (gsub("_","",name)).
#
# NOTE ON CONVERGENCE: DIALOGUE only recovers programs when the per-sample structure
# in X is real and shared across cell types (ANOVA filter needs >=5 sample-varying
# features; sparse-CCA needs cross-type correlation). Pure noise -> "features passed
# the ANOVA filter" error (nonzero exit = method skip). See tests/test_mcp_dialogue_script.py.
suppressPackageStartupMessages({ library(DIALOGUE); library(Matrix); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop("Usage: dialogue.R <scratch_dir> <out_dir> <k> <n_program_genes> ",
       "<seed> <pheno_col_or_NA> <abn_c>")
}
scratch_dir <- args[1]
out_dir <- args[2]
k <- as.integer(args[3])
n_program_genes <- as.integer(args[4])
seed <- as.integer(args[5])
pheno <- if (is.na(args[6]) || args[6] %in% c("NA", "NULL", "")) NULL else args[6]
abn_c <- as.integer(args[7])

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

# ---- read cell-type manifest ------------------------------------------------
ct_json <- file.path(scratch_dir, "celltypes.json")
if (!file.exists(ct_json)) stop("celltypes.json not found in scratch_dir: ", scratch_dir)
ct_map <- jsonlite::fromJSON(ct_json, simplifyVector = FALSE)  # stripped -> list(label, dir)
if (length(ct_map) < 2) stop("DIALOGUE requires >=2 cell types; got ", length(ct_map))

# ---- build one make.cell.type object per cell type --------------------------
rA <- list()
cell_counts <- list()
for (sct in names(ct_map)) {
  ctdir <- file.path(scratch_dir, ct_map[[sct]]$dir)
  if (!dir.exists(ctdir)) stop("cell-type dir missing for '", sct, "': ", ctdir)

  mtx <- Matrix::readMM(file.path(ctdir, "expr.mtx"))
  genes <- readLines(file.path(ctdir, "genes.txt")); genes <- genes[nzchar(genes)]
  cells <- readLines(file.path(ctdir, "cells.txt")); cells <- cells[nzchar(cells)]
  if (nrow(mtx) != length(genes) || ncol(mtx) != length(cells)) {
    stop("expr.mtx dims (", nrow(mtx), "x", ncol(mtx), ") != genes/cells (",
         length(genes), "x", length(cells), ") for '", sct, "'")
  }
  tpm <- as.matrix(mtx)
  rownames(tpm) <- genes; colnames(tpm) <- cells

  # X: cells x features, first col 'cell' -> rownames; reorder rows to colnames(tpm).
  Xdf <- read.csv(file.path(ctdir, "X.csv"), check.names = FALSE, stringsAsFactors = FALSE)
  if (!"cell" %in% colnames(Xdf)) stop("X.csv missing 'cell' column for '", sct, "'")
  rownames(Xdf) <- Xdf$cell; Xdf$cell <- NULL
  if (!all(cells %in% rownames(Xdf))) stop("X.csv missing cells for '", sct, "'")
  Xmat <- as.matrix(Xdf[cells, , drop = FALSE])

  # meta: first col 'cell' -> rownames; reorder rows to colnames(tpm).
  meta <- read.csv(file.path(ctdir, "meta.csv"), check.names = FALSE, stringsAsFactors = FALSE)
  if (!"cell" %in% colnames(meta)) stop("meta.csv missing 'cell' column for '", sct, "'")
  rownames(meta) <- meta$cell; meta$cell <- NULL
  if (!all(cells %in% rownames(meta))) stop("meta.csv missing cells for '", sct, "'")
  meta <- meta[cells, , drop = FALSE]
  for (req in c("sample", "cellQ")) {
    if (!req %in% colnames(meta)) {
      stop("meta.csv missing required column '", req, "' for '", sct, "'; have: ",
           paste(colnames(meta), collapse = ", "))
    }
  }
  if (!is.null(pheno) && !pheno %in% colnames(meta)) {
    stop("pheno column '", pheno, "' not found in meta.csv for '", sct, "'; have: ",
         paste(colnames(meta), collapse = ", "))
  }

  # cellQ is added to metadata automatically by make.cell.type; pass only the
  # phenotype column (confounders are the hardcoded conf/covar='cellQ').
  md <- if (!is.null(pheno)) meta[, pheno, drop = FALSE] else NULL

  # Gotcha #1: rownames(X) must equal colnames(tpm) in identical order.
  if (!identical(rownames(Xmat), colnames(tpm))) {
    stop("Cell ordering mismatch between X and tpm for '", sct, "'")
  }

  # Gotcha #2/#3: name is stripped (gsub("_","",name)); cellQ REQUIRED.
  r <- make.cell.type(
    name = sct, tpm = tpm, samples = as.character(meta$sample),
    X = Xmat, metadata = md, cellQ = as.numeric(meta$cellQ)
  )
  if (is.character(r)) stop("make.cell.type failed for '", sct, "': ", r)
  rA[[r@name]] <- r
  cell_counts[[r@name]] <- ncol(tpm)
}

# ---- run DIALOGUE -----------------------------------------------------------
dlg_dir <- file.path(out_dir, "dlg")
dir.create(dlg_dir, recursive = TRUE, showWarnings = FALSE)
param <- DLG.get.param(
  k = k, results.dir = paste0(dlg_dir, "/"), seed1 = seed, pheno = pheno,
  conf = "cellQ", covar = "cellQ",              # gotcha #3: avoid default tme.qc covar
  n.genes = n_program_genes, abn.c = abn_c, plot.flag = FALSE
)
res <- tryCatch(
  DIALOGUE.run(rA = rA, main = "cellquorum", param = param, plot.flag = FALSE),
  error = function(e) { message("DIALOGUE.run failed: ", conditionMessage(e)); quit(status = 1) }
)

# ---- helpers for stripped->original mapping ---------------------------------
label_of <- function(sct) if (!is.null(ct_map[[sct]])) ct_map[[sct]]$label else sct

prog_cols   <- c("program", "cell_type", "gene", "loading", "direction")
score_cols  <- c("cell_id", "sample", "cell_type", "program", "score")
assoc_cols  <- c("program", "statistic", "pvalue", "padj", "direction")
empty_df <- function(cols) { d <- data.frame(matrix(nrow = 0, ncol = length(cols))); names(d) <- cols; d }

# Graceful null result (no programs recovered): headed-but-empty CSVs, exit 0.
if (is.null(res$MCPs)) {
  write.csv(empty_df(prog_cols),  file.path(out_dir, "mcp_gene_programs.csv"), row.names = FALSE)
  write.csv(empty_df(score_cols), file.path(out_dir, "mcp_scores.csv"),        row.names = FALSE)
  write.csv(empty_df(assoc_cols), file.path(out_dir, "mcp_associations.csv"),  row.names = FALSE)
  meta_out <- list(k = k, cell_counts = cell_counts,
                   dialogue_version = as.character(packageVersion("DIALOGUE")), seed = seed)
  writeLines(jsonlite::toJSON(meta_out, auto_unbox = TRUE, pretty = TRUE),
             file.path(out_dir, "run_meta.json"))
  quit(status = 0)
}

# ---- mcp_gene_programs.csv: res$MCPs = list(MCP1..MCPk); each NULL or a --------
# named list keyed by "<stripped>.up"/"<stripped>.down" -> gene char vector.
# loading = NNLS coef from res$gene.pval[[stripped]] (cols: program, up, genes, coef).
prog_rows <- list()
for (pg in names(res$MCPs)) {
  mcp <- res$MCPs[[pg]]
  if (is.null(mcp)) next
  for (key in names(mcp)) {
    dir_ <- if (grepl("\\.down$", key)) "down" else "up"
    sct <- sub("\\.(up|down)$", "", key)
    genes <- mcp[[key]]
    if (length(genes) == 0) next
    loadings <- rep(NA_real_, length(genes))
    gp <- res$gene.pval[[sct]]
    if (!is.null(gp)) {
      gp2 <- gp[gp$program == pg & gp$up == (dir_ == "up"), , drop = FALSE]
      loadings <- gp2$coef[match(genes, gp2$genes)]
    }
    prog_rows[[length(prog_rows) + 1]] <- data.frame(
      program = pg, cell_type = label_of(sct), gene = genes,
      loading = loadings, direction = dir_, stringsAsFactors = FALSE
    )
  }
}
prog_df <- if (length(prog_rows)) do.call(rbind, prog_rows) else empty_df(prog_cols)
write.csv(prog_df, file.path(out_dir, "mcp_gene_programs.csv"), row.names = FALSE)

# ---- mcp_scores.csv: res$scores[[stripped]] = data.frame(MCP1..MCPk, samples, --
# cells, cell.type, metadata...). Emit long form for surviving programs only.
surviving <- names(res$MCPs)[!vapply(res$MCPs, is.null, logical(1))]
score_rows <- list()
for (sct in names(res$scores)) {
  sdf <- res$scores[[sct]]
  for (pg in intersect(surviving, colnames(sdf))) {
    score_rows[[length(score_rows) + 1]] <- data.frame(
      cell_id = sdf$cells, sample = sdf$samples, cell_type = label_of(sct),
      program = pg, score = sdf[[pg]], stringsAsFactors = FALSE
    )
  }
}
score_df <- if (length(score_rows)) do.call(rbind, score_rows) else empty_df(score_cols)
write.csv(score_df, file.path(out_dir, "mcp_scores.csv"), row.names = FALSE)

# ---- mcp_associations.csv: res$phenoZ = matrix [cell types + "All"] x MCP -----
# columns; values are HLM z-scores. Use the pooled "All" row (one row/program).
assoc_df <- empty_df(assoc_cols)
if (!is.null(pheno) && !is.null(res$phenoZ)) {
  Z <- res$phenoZ
  if (is.null(dim(Z))) Z <- matrix(Z, nrow = 1, dimnames = list("All", names(Z)))
  row_key <- if ("All" %in% rownames(Z)) "All" else rownames(Z)[nrow(Z)]
  zvals <- Z[row_key, ]
  progs <- colnames(Z)
  keep <- !is.na(zvals)
  progs <- progs[keep]; zvals <- as.numeric(zvals[keep])
  if (length(progs)) {
    pv <- 2 * pnorm(-abs(zvals))
    assoc_df <- data.frame(
      program = progs, statistic = zvals, pvalue = pv,
      padj = p.adjust(pv, method = "BH"),
      direction = ifelse(zvals > 0, "up", "down"), stringsAsFactors = FALSE
    )
  }
}
write.csv(assoc_df, file.path(out_dir, "mcp_associations.csv"), row.names = FALSE)

# ---- run_meta.json ----------------------------------------------------------
meta_out <- list(
  k = k, cell_counts = cell_counts,
  dialogue_version = as.character(packageVersion("DIALOGUE")), seed = seed
)
writeLines(jsonlite::toJSON(meta_out, auto_unbox = TRUE, pretty = TRUE),
           file.path(out_dir, "run_meta.json"))
