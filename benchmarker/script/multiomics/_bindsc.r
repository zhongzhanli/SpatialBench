library(bindSC)
library(Seurat)
library(rhdf5)
library(Matrix)
options(Seurat.object.assay.version = "v3")
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

find_fixed_clus <- function(
  seu,
  reduction,
  dims,
  n_neighbors = 50,
  fixed_clus_count = 10,
  resolution_seq = seq(0.1, 2, by = 0.05)
) {

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

run_bindsc_unified <- function(
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
    stop("BindSC requires gene activity input in adt_path.")
  }

  if (!dir.exists(save_path)) {
    dir.create(save_path, recursive = TRUE)
  }

  cat("[1/6] Loading data ...\n")
  rna_mat <- h5_to_matrix(rna_path)
  atac_mat <- h5_to_matrix(atac_path)
  ga_mat <- h5_to_matrix(adt_path)

  # common_cells <- Reduce(intersect, list(colnames(rna_mat), colnames(atac_mat), colnames(ga_mat)))
  # if (length(common_cells) == 0) {
  #   stop("RNA, ATAC and gene activity have no overlapping cells!")
  # }

  # rna_mat <- rna_mat[, common_cells, drop = FALSE]
  # atac_mat <- atac_mat[, common_cells, drop = FALSE]
  # ga_mat <- ga_mat[, common_cells, drop = FALSE]

  cat("RNA dim:", dim(rna_mat)[1], dim(rna_mat)[2], "\n")
  cat("ATAC dim:", dim(atac_mat)[1], dim(atac_mat)[2], "\n")
  cat("Gene activity dim:", dim(ga_mat)[1], dim(ga_mat)[2], "\n")

  cat("[2/6] Building Seurat objects ...\n")
  rna <- CreateSeuratObject(counts = rna_mat)
  genescore <- CreateSeuratObject(counts = ga_mat)

  cat("[3/6] Preprocessing RNA / gene activity ...\n")
  rna <- NormalizeData(rna, verbose = FALSE)
  rna <- FindVariableFeatures(rna, verbose = FALSE, nfeatures = hvg_num)
  rna <- ScaleData(rna, verbose = FALSE)
  rna <- RunPCA(rna, npcs = 30, verbose = FALSE)
  rna <- FindNeighbors(rna, reduction = "pca", dims = 1:30, verbose = FALSE)
  rna <- FindClusters(rna, resolution = 0.8, verbose = FALSE)

  genescore <- NormalizeData(genescore, verbose = FALSE)
  genescore <- FindVariableFeatures(genescore, verbose = FALSE, nfeatures = hvg_num)
  genescore <- ScaleData(genescore, verbose = FALSE)
  genescore <- RunPCA(genescore, npcs = 30, verbose = FALSE)
  genescore <- FindNeighbors(genescore, reduction = "pca", dims = 1:30, verbose = FALSE)
  genescore <- FindClusters(genescore, resolution = 0.8, verbose = FALSE)

  cat("[4/6] Constructing BindSC inputs ...\n")
  X <- GetAssayData(rna, assay = "RNA", slot = "counts")
  Z0 <- GetAssayData(genescore, assay = "RNA", slot = "counts")
  Y <- atac_mat

  gene_overlap <- intersect(rownames(X), rownames(Z0))
  cell_overlap <- Reduce(intersect, list(colnames(Y), colnames(Z0)))

  if (length(gene_overlap) == 0) {
    stop("RNA and gene activity have no overlapping genes!")
  }
  if (length(cell_overlap) == 0) {
    stop("RNA, ATAC and gene activity have no overlapping cells after preprocessing!")
  }

  X <- as.matrix(X[gene_overlap,])
  Z0 <- as.matrix(Z0[gene_overlap, cell_overlap])
  Y <- as.matrix(Y[, cell_overlap])

  out_shared <- dimReduce(dt1 = X, dt2 = Z0, K = 30)
  x <- out_shared$dt1
  z0 <- out_shared$dt2
  y <- dimReduce(dt1 = Y, K = 30)

  cat("[5/6] Running BindSC and saving latent ...\n")
  res <- BiCCA(
    X = t(x),
    Y = t(y),
    Z0 = t(z0),
    X.clst = as.vector(rna$seurat_clusters),
    Y.clst = as.vector(genescore$seurat_clusters),
    alpha = 0.1,
    lambda = 0.7,
    K = 15,
    temp.path = file.path(save_path, paste0(save_key, "_bindsc_tmp")),
    num.iteration = 50,
    tolerance = 0.01,
    save = TRUE,
    parameter.optimize = FALSE,
    block.size = 0
  )

#   rna_embed <- as.data.frame(res$u)
#   atac_embed <- as.data.frame(res$r)
  embed = rbind(res$u, res$r)

  write.csv(
    embed,
    file = file.path(save_path, paste0(save_key, "_latent.csv")),
    quote = FALSE
  )

  cat("[6/6] Clustering and UMAP ...\n")
  dummy_counts <- Matrix(0, nrow = 100, ncol = dim(embed)[1], sparse = TRUE)

  rownames(dummy_counts) <- paste0("Gene_", 1:100)
  colnames(dummy_counts) <- paste0("Cell_", 1:dim(embed)[1])

  seu_out <- CreateSeuratObject(counts = dummy_counts)

  colnames(embed) <- paste0("BINDSC_", 1:dim(embed)[2])
  rownames(embed) <- paste0("Cell_", 1:dim(embed)[1])

  seu_out[["BINDSC"]] <- CreateDimReducObject(
    embeddings = as.matrix(embed),
    key = "BINDSC_",
    assay = DefaultAssay(seu_out)
  )

  dims_use <- 1:ncol(embed)
  # stop(sum(duplicated(colnames(seu_out))))
  seu_out <- FindNeighbors(
    seu_out,
    reduction = "BINDSC",
    dims = dims_use,
    verbose = FALSE
  )
  res_use <- find_fixed_clus(
    seu = seu_out,
    reduction = "BINDSC",
    dims = dims_use,
    fixed_clus_count = n_cluster
  )

  seu_out <- FindClusters(seu_out, resolution = res_use, verbose = FALSE)
  seu_out <- RunUMAP(
    seu_out,
    reduction = "BINDSC",
    dims = dims_use,
    verbose = FALSE
  )

  umap <- as.data.frame(seu_out@reductions$umap@cell.embeddings)
  colnames(umap) <- c("UMAP1", "UMAP2")
  umap[, "cluster"] <- seu_out@meta.data$seurat_clusters
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

run_bindsc_unified(
  n_cluster = n_cluster,
  rna_path = rna_path,
  atac_path = atac_path,
  adt_path = adt_path,
  save_path = save_path,
  save_key = save_key,
  hvg_num = hvg_num,
  batch_key = batch_key
)