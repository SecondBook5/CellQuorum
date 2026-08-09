# NicheNet ligand-activity for a single sender->receiver pair via nichenetr.
# All settings via argv; no study-specific biology baked in.
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

# Expression fraction per gene within a cell-type group.
frac_expr <- function(ct) {
  cells <- rownames(obs)[obs[[celltype_col]] == ct]
  if (length(cells) == 0) return(character(0))
  sub <- counts[, cells, drop = FALSE]
  rownames(sub)[Matrix::rowMeans(sub > 0) >= expr_prop]
}
expressed_sender <- frac_expr(sender)
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
top <- head(activities$test_ligand, top_ligands)

links <- lapply(top, function(lg) {
  wt <- get_weighted_ligand_target_links(
    ligand = lg, geneset = geneset,
    ligand_target_matrix = ligand_target_matrix, n = top_targets
  )
  if (is.null(wt) || nrow(wt) == 0) return(NULL)
  # attach a cognate receptor from lr_network for the canonical schema
  recs <- lr_network$to[lr_network$from == lg & lr_network$to %in% expressed_receptors]
  data.frame(ligand = lg, receptor = if (length(recs)) recs[[1]] else NA,
             aupr_corrected = activities$aupr_corrected[activities$test_ligand == lg][1],
             stringsAsFactors = FALSE)
})
links <- do.call(rbind, links)

write.csv(activities, out_activities, row.names = FALSE)
write.csv(links, out_links, row.names = FALSE)
