# NicheNet ligand-activity for one or more senders -> one receiver via nichenetr.
# All settings via argv; no study-specific biology baked in.
#
# ``sender`` is a comma-separated list. With more than one entry the ligand pool is the union
# of what those senders express, which is nichenetr's own sender-agnostic setup: the activity
# of a ligand is a property of the receiver's response, not of who sent it, so ranking ligands
# once against a pooled pool and attributing them afterwards is both cheaper and cleaner than
# one ranking per sender (which makes the AUPR values non-comparable across senders, since
# each run has a different candidate pool).
#
# Attribution is therefore a separate output: ``out_sender_expression`` carries, for every top
# ligand and every sender, the fraction of that sender's cells expressing it and the mean
# expression. "Which cell type sends this ligand" is an expression question, and this is the
# table that answers it.
suppressPackageStartupMessages({ library(Matrix) })

args <- commandArgs(trailingOnly = TRUE)
counts_mtx <- args[[1]]; genes_csv <- args[[2]]; barcodes_csv <- args[[3]]
obs_csv <- args[[4]]; geneset_csv <- args[[5]]; background_csv <- args[[6]]
out_activities <- args[[7]]; out_links <- args[[8]]
celltype_col <- args[[9]]; sender <- args[[10]]; receiver <- args[[11]]
ligand_target_rds <- args[[12]]; lr_network_rds <- args[[13]]
weighted_networks_rds <- args[[14]]
expr_prop <- as.numeric(args[[15]]); top_ligands <- as.integer(args[[16]])
top_targets <- as.integer(args[[17]]); seed <- as.integer(args[[18]])
out_target_weights <- if (length(args) >= 19) args[[19]] else NA_character_
out_sender_expression <- if (length(args) >= 20) args[[20]] else NA_character_

set.seed(seed)
stopifnot(file.exists(ligand_target_rds), file.exists(lr_network_rds),
          file.exists(weighted_networks_rds))

suppressPackageStartupMessages({ library(nichenetr); library(dplyr) })

counts <- as(Matrix::readMM(counts_mtx), "CsparseMatrix")  # genes x cells
genes <- read.csv(genes_csv, stringsAsFactors = FALSE)$gene
barcodes <- read.csv(barcodes_csv, stringsAsFactors = FALSE)$barcode
rownames(counts) <- genes; colnames(counts) <- barcodes
obs <- read.csv(obs_csv, stringsAsFactors = FALSE); rownames(obs) <- obs$barcode
obs <- obs[barcodes, , drop = FALSE]

ligand_target_matrix <- readRDS(ligand_target_rds)
lr_network <- readRDS(lr_network_rds)
weighted_networks <- readRDS(weighted_networks_rds)

geneset <- read.csv(geneset_csv, stringsAsFactors = FALSE)$gene
background <- read.csv(background_csv, stringsAsFactors = FALSE)$gene
geneset <- geneset[geneset %in% rownames(ligand_target_matrix)]
background <- background[background %in% rownames(ligand_target_matrix)]
stopifnot(length(geneset) > 0, length(background) > 0)

senders <- trimws(strsplit(sender, ",", fixed = TRUE)[[1]])
senders <- senders[nzchar(senders)]
stopifnot(length(senders) > 0)

# Per-cell-type expression: the fraction of cells with a non-zero count, and the mean.
cells_of <- function(ct) rownames(obs)[obs[[celltype_col]] == ct]
frac_expr <- function(ct) {
  cells <- cells_of(ct)
  if (length(cells) == 0) return(character(0))
  sub <- counts[, cells, drop = FALSE]
  rownames(sub)[Matrix::rowMeans(sub > 0) >= expr_prop]
}
expressed_sender <- unique(unlist(lapply(senders, frac_expr)))
expressed_receiver <- frac_expr(receiver)

ligands <- unique(lr_network$from)
receptors <- unique(lr_network$to)
expressed_ligands <- intersect(ligands, expressed_sender)
expressed_receptors <- intersect(receptors, expressed_receiver)
potential_ligands <- lr_network %>%
  dplyr::filter(from %in% expressed_ligands, to %in% expressed_receptors) %>%
  dplyr::pull(from) %>% unique()
stopifnot(length(potential_ligands) > 0)

activities <- predict_ligand_activities(
  geneset = geneset, background_expressed_genes = background,
  ligand_target_matrix = ligand_target_matrix, potential_ligands = potential_ligands
)
activities <- activities[order(-activities$aupr_corrected), ]
activities$rank <- seq_len(nrow(activities))
top <- head(activities$test_ligand, top_ligands)

# Ligand -> target weights, kept rather than discarded: these are what a NicheNet figure is
# made of, and they are the only output that says *through which target genes* a ligand is
# predicted to act on this gene set.
weights <- lapply(top, function(lg) {
  wt <- get_weighted_ligand_target_links(
    ligand = lg, geneset = geneset,
    ligand_target_matrix = ligand_target_matrix, n = top_targets
  )
  if (is.null(wt) || nrow(wt) == 0) return(NULL)
  data.frame(ligand = lg, target = wt$target, weight = wt$weight,
             stringsAsFactors = FALSE)
})
weights <- do.call(rbind, weights)

links <- lapply(top, function(lg) {
  if (is.null(weights) || !any(weights$ligand == lg)) return(NULL)
  # attach a cognate receptor from lr_network for the canonical schema
  recs <- lr_network$to[lr_network$from == lg & lr_network$to %in% expressed_receptors]
  data.frame(ligand = lg, receptor = if (length(recs)) recs[[1]] else NA,
             aupr_corrected = activities$aupr_corrected[activities$test_ligand == lg][1],
             stringsAsFactors = FALSE)
})
links <- do.call(rbind, links)

write.csv(activities, out_activities, row.names = FALSE)
write.csv(links, out_links, row.names = FALSE)
if (!is.na(out_target_weights) && !is.null(weights)) {
  write.csv(weights, out_target_weights, row.names = FALSE)
}

if (!is.na(out_sender_expression)) {
  rows <- lapply(senders, function(ct) {
    cells <- cells_of(ct)
    if (length(cells) == 0) return(NULL)
    present <- intersect(top, rownames(counts))
    if (length(present) == 0) return(NULL)
    sub <- counts[present, cells, drop = FALSE]
    data.frame(
      sender = ct, ligand = present,
      n_cells = length(cells),
      fraction_expressing = as.numeric(Matrix::rowMeans(sub > 0)),
      mean_expression = as.numeric(Matrix::rowMeans(sub)),
      stringsAsFactors = FALSE
    )
  })
  rows <- do.call(rbind, rows)
  if (!is.null(rows)) {
    rows$expressed <- rows$fraction_expressing >= expr_prop
    write.csv(rows, out_sender_expression, row.names = FALSE)
  }
}
