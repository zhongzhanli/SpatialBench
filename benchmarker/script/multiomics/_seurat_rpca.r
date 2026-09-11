library(rhdf5)
library(Seurat)
library(Signac)
library(ggplot2)
library(cowplot)

options(future.globals.maxSize = 8 * 1024^3)

is_null_string <- function(x) {
  toupper(as.character(x)) == "NULL"
}

h5_to_matrix <- function(path, sparse = TRUE) {
  suppressPackageStartupMessages({
    library(rhdf5)
    library(Matrix)
  })
  # h5_exists <- function(file, name) {
  #   info <- h5ls(file)
  #   any(paste0(info$group, "/", info$name) == name)
  # }
  # if (!h5_exists(path, "/matrix")) {
  #   stop("H5 file does not contain group '/matrix'.")
  # }
  # is_sparse <- sparse &&
  #   h5_exists(path, "/matrix/data") &&
  #   h5_exists(path, "/matrix/indices") &&
  #   h5_exists(path, "/matrix/indptr") &&
  #   h5_exists(path, "/matrix/shape")

  # if (is_sparse) {
    x      <- h5read(path, "matrix/data")
    indices <- h5read(path, "matrix/indices")
    indptr  <- h5read(path, "matrix/indptr")
    shape   <- h5read(path, "matrix/shape")

    # if (h5_exists(path, "/matrix/features/name")) {
    #   feature <- h5read(path, "matrix/features/name")
    # } else if (h5_exists(path, "/matrix/features")) {
    feature <- h5read(path, "matrix/features")
    # } else {
    #   stop("Sparse H5: cannot find features at 'matrix/features' or 'matrix/features/name'.")
    # }

    # if (h5_exists(path, "/matrix/barcodes")) {
    barcode <- h5read(path, "matrix/barcodes")
    # } else {
    #   stop("Sparse H5: cannot find barcodes at 'matrix/barcodes'.")
    # }

    feature <- as.character(feature)
    barcode <- as.character(barcode)
    n_cell <- as.integer(shape[1])
    n_gene <- as.integer(shape[2])
    nnz_per_row <- diff(indptr)
    if (length(nnz_per_row) != n_cell) {
      stop("Sparse H5: length(diff(indptr)) != n_cell; CSR structure mismatch.")
    }

    i <- rep(seq_len(n_cell), times = nnz_per_row)
    j <- as.integer(indices) + 1L  # 0-based -> 1-based
    mat_cg <- sparseMatrix(
      i = i,
      j = j,
      x = x,
      dims = c(n_cell, n_gene),
      giveCsparse = TRUE
    )
    data <- t(mat_cg)
    colnames(data) <- paste0(seq_len(ncol(data)), barcode)
    rownames(data) <- feature

    return(data)
  # }
  # h5_data <- h5read(path, "matrix")

  # if (is.null(h5_data$data) || is.null(h5_data$features) || is.null(h5_data$barcodes)) {
  #   stop("Dense H5: expected 'matrix' to contain $data, $features, $barcodes.")
  # }

  # feature <- as.character(h5_data$features)
  # barcode <- as.character(h5_data$barcodes)

  # data <- t(h5_data$data)

  # colnames(data) <- paste0(seq_len(ncol(data)), barcode)
  # rownames(data) <- paste0(seq_len(nrow(data)), feature)

  # return(data)
}

find_fixed_clus <- function(seu, reduction = "pca", dims = 1:30,
                            n_neighbors = 50, fixed_clus_count = 10,
                            resolution_seq = seq(0.1, 2, by = 0.02)) {

  best_res <- NULL
  best_diff <- Inf

  for (res in rev(resolution_seq)) {
    seu_tmp <- FindClusters(seu, resolution = res, verbose = FALSE)
    k <- length(unique(seu_tmp$seurat_clusters))
    d <- abs(k - fixed_clus_count)
    if (d < best_diff) {
      best_diff <- d
      best_res <- res
    }
    if (k == fixed_clus_count) {
      best_res <- res
      break
    }
  }
  return(best_res)
}

run_seurat_v3_unified <- function(
  n_cluster,
  rna_path,
  atac_path,
  adt_path,
  save_path,
  save_key,
  hvg_num,
  batch_key
) {
  if (is_null_string(adt_path)) {
    stop("This Seurat v3 runner requires gene activity input in adt_path.")
  }

  if (!dir.exists(save_path)) {
    dir.create(save_path, recursive = TRUE)
  }

  cat("[1/6] Loading data ...\n")
  rna <- h5_to_matrix(rna_path)
  atac <- h5_to_matrix(atac_path)
  gene.activities <- h5_to_matrix(adt_path)
  
  # stop(length(rownames(gene.activities)))

  gene_overlap <- intersect(rownames(rna), rownames(gene.activities))
  rna = rna[gene_overlap,]
  gene.activities = gene.activities[gene_overlap,]

#   common_cells <- Reduce(intersect, list(colnames(rna), colnames(atac), colnames(gene.activities)))
#   if (length(common_cells) == 0) {
#     stop("RNA, ATAC and gene activity have no overlapping cell barcodes!")
#   }

#   rna <- rna[, common_cells, drop = FALSE]
#   atac <- atac[, common_cells, drop = FALSE]
#   gene.activities <- gene.activities[, common_cells, drop = FALSE]

  cat("RNA dim:", dim(rna)[1], dim(rna)[2], "\n")
  cat("ATAC dim:", dim(atac)[1], dim(atac)[2], "\n")
  cat("Gene activity dim:", dim(gene.activities)[1], dim(gene.activities)[2], "\n")

  cat("[2/6] Building Seurat objects ...\n")
  rna_seurat <- CreateSeuratObject(counts = rna)
  atac_assay <- CreateChromatinAssay(counts = atac, sep = c(":", "-"))
  atac_seurat <- CreateSeuratObject(counts = atac_assay, assay = "ATAC")

  cat("[3/6] Preprocessing RNA / ATAC ...\n")
  rna_seurat <- NormalizeData(rna_seurat, verbose = FALSE)
  rna_seurat <- FindVariableFeatures(
    rna_seurat,
    selection.method = "vst",
    nfeatures = hvg_num,
    verbose = FALSE
  )
  rna_seurat <- ScaleData(rna_seurat, verbose = FALSE)
  rna_seurat <- RunPCA(rna_seurat, npcs = 50, verbose = FALSE)

  atac_seurat <- RunTFIDF(atac_seurat)
  atac_seurat <- FindTopFeatures(atac_seurat, min.cutoff = "q0")
  atac_seurat <- RunSVD(atac_seurat)

  atac_seurat[["ACTIVITY"]] <- CreateAssayObject(counts = gene.activities)
  DefaultAssay(atac_seurat) <- "ACTIVITY"
  atac_seurat <- NormalizeData(atac_seurat, verbose = FALSE)
  atac_seurat <- ScaleData(atac_seurat, features = rownames(atac_seurat), verbose = FALSE)

  cat("[4/6] Finding anchors and co-embedding ...\n")
  transfer.anchors <- FindTransferAnchors(
    reference = rna_seurat,
    query = atac_seurat,
    features = VariableFeatures(rna_seurat),
    reference.assay = "RNA",
    query.assay = "ACTIVITY",
    reduction = "rpca",
    k.anchor = 20
  )

  genes.use <- VariableFeatures(rna_seurat)
  refdata <- GetAssayData(rna_seurat, assay = "RNA", slot = "data")[genes.use, , drop = FALSE]

  imputation <- TransferData(
    anchorset = transfer.anchors,
    refdata = refdata,
    weight.reduction = atac_seurat[["lsi"]],
    dims = 2:30
  )

  atac_seurat[["RNA"]] <- imputation

  coembed <- merge(x = rna_seurat, y = atac_seurat)
  DefaultAssay(coembed) <- "RNA"
  coembed <- ScaleData(coembed, features = genes.use, do.scale = FALSE, verbose = FALSE)
  coembed <- RunPCA(coembed, features = genes.use, npcs = 50, verbose = FALSE)

  cat("[5/6] Saving latent ...\n")
  latent <- coembed@reductions[["pca"]]@cell.embeddings

  write.csv(
    latent,
    file = file.path(save_path, paste0(save_key, "_latent.csv")),
    quote = FALSE,
    row.names =T, col.names = T, sep=','
  )

  cat("[6/6] Clustering and UMAP ...\n")
  coembed <- FindNeighbors(
    coembed,
    reduction = "pca",
    dims = 1:30,
    verbose = FALSE
  )
  res_use <- find_fixed_clus(
    seu = coembed,
    reduction = "pca",
    dims = 1:30,
    n_neighbors = 50,
    fixed_clus_count = n_cluster
  )
  coembed <- FindClusters(coembed, resolution = res_use, verbose = FALSE)
  coembed <- RunUMAP(coembed, reduction = "pca", dims = 1:30, verbose = FALSE)

  umap <- as.data.frame(coembed@reductions$umap@cell.embeddings)
  colnames(umap) <- c("UMAP1", "UMAP2")
  umap[, "cluster"] <- coembed@meta.data$seurat_clusters
  write.table(umap, 
              file = file.path(save_path, paste0(save_key, ".csv")), 
              row.names =T, col.names = T, sep=',', quote=F)
}

args <- commandArgs(trailingOnly = TRUE)

n_cluster <- as.integer(args[1])
rna_path   <- args[2]
atac_path  <- args[3]
adt_path   <- args[4]
save_path  <- args[5]
save_key   <- args[6]
hvg_num    <- as.integer(args[7])
batch_key  <- args[8]

run_seurat_v3_unified(
  n_cluster = n_cluster,
  rna_path = rna_path,
  atac_path = atac_path,
  adt_path = adt_path,
  save_path = save_path,
  save_key = save_key,
  hvg_num = hvg_num,
  batch_key = batch_key
)