#!/usr/bin/env Rscript
# Export the public MultiNicheNet paired SCC subset as per-library count matrices.
suppressPackageStartupMessages(library(SingleCellExperiment))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript prepare_scc.R sce_subset_scc.rds output_directory")
stopifnot(unname(tools::md5sum(args[1])) == "3d9884f9d55bb5c4eeb8a8261f47d215")
sce <- readRDS(args[1])
obs <- as.data.frame(colData(sce))
dir.create(args[2], recursive = TRUE, showWarnings = FALSE)
libraries <- split(seq_len(ncol(sce)), obs$sample_id, drop = TRUE)
rows <- lapply(names(libraries), function(id) {
    cells <- libraries[[id]]
    d <- obs[cells, , drop = FALSE]
    sample_dir <- file.path(args[2], id)
    dir.create(sample_dir, showWarnings = FALSE)
    DropletUtils::write10xCounts(file.path(sample_dir, "counts.h5"),
        counts(sce)[, cells, drop = FALSE], version = "3", type = "HDF5",
        gene.id = rownames(sce), gene.symbol = rownames(sce), barcodes = colnames(sce)[cells],
        overwrite = TRUE)
    write.csv(data.frame(barcode = colnames(sce)[cells], d, check.names = FALSE),
        file.path(sample_dir, "cells.csv"), row.names = FALSE)
    data.frame(sample_id = id, path = file.path(id, "counts.h5"),
        metadata_path = file.path(id, "cells.csv"))
})
write.csv(do.call(rbind, rows), file.path(args[2], "samples.csv"), row.names = FALSE)
