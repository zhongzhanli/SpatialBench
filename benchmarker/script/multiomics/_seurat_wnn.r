suppressPackageStartupMessages({
  library(Seurat)
  library(patchwork)
  library(dplyr)
  library(rhdf5)
  library(mclust)
  library(HDF5Array)
})
suppressPackageStartupMessages({
  library(Signac)
  library(Seurat)
  library(dplyr)
  library(rhdf5)
  library(HDF5Array)
})

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
    rownames(data) <- paste0(seq_len(nrow(data)), feature)

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

find_fixed_clus <- function(obj, n_cluster, vec = seq(0.1, 2.0, 0.05)) {
  closest_res <- NA
  closest_diff <- Inf
  
  for (i in vec) {
    obj <- FindClusters(obj, graph.name = "wsnn", algorithm = 3, resolution = i, verbose = FALSE)
    n_found <- length(unique(obj$seurat_clusters))
    
    if (n_found == n_cluster) {
      return(i)
    }
    
    diff <- abs(n_found - n_cluster)
    if (diff < closest_diff) {
      closest_diff <- diff
      closest_res <- i
    }
  }
  
  return(closest_res)
}

run_Seurat_RNA_ADT <- function(rna, adt, hvg_num){
  bm <- CreateSeuratObject(counts = rna)
  adt_assay <- CreateAssayObject(counts = adt)
  bm[["ADT"]] <- adt_assay
  
  DefaultAssay(bm) <- 'RNA'
  bm <- NormalizeData(bm) %>% FindVariableFeatures(nfeatures=hvg_num) %>% ScaleData() %>% RunPCA()

  DefaultAssay(bm) <- 'ADT'
  # we will use all ADT features for dimensional reduction
  # we set a dimensional reduction name to avoid overwriting the 
  VariableFeatures(bm) <- rownames(bm[["ADT"]])
  bm <- NormalizeData(bm, normalization.method = 'CLR', margin = 2) %>% 
    ScaleData() %>% RunPCA(reduction.name = 'apca')
  
  # Identify multimodal neighbors. These will be stored in the neighbors slot, 
  # and can be accessed using bm[['weighted.nn']]
  # The WNN graph can be accessed at bm[["wknn"]], 
  # and the SNN graph used for clustering at bm[["wsnn"]]
  # Cell-specific modality weights can be accessed at bm$RNA.weight
  bm <- FindMultiModalNeighbors(
    bm, reduction.list = list("pca", "apca"), 
    dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
  )

  bm <- RunUMAP(bm, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_", return.model = TRUE)
  res <- find_fixed_clus(bm, n_cluster=n_cluster)
  bm <- FindClusters(bm, graph.name = "wsnn", algorithm = 3, resolution = res, verbose = FALSE)
  
  return(bm) # the dimension of embedding is 2
}

run_Seurat_RNA_ATAC <- function(rna, atac, hvg_num){
    bm <- CreateSeuratObject(counts = rna)
    atac_assay <- CreateAssayObject(counts = atac)
    bm[["ATAC"]] <- atac_assay
    
    DefaultAssay(bm) <- "RNA"
    bm <- SCTransform(bm, verbose = FALSE, variable.features.n=hvg_num) %>% RunPCA() %>% RunUMAP(dims = 1:50, reduction.name = 'umap.rna', reduction.key = 'rnaUMAP_')
    
    # ATAC analysis
    # We exclude the first dimension as this is typically correlated with sequencing depth
    DefaultAssay(bm) <- "ATAC"
    bm <- RunTFIDF(bm)
    bm <- FindTopFeatures(bm, min.cutoff = 'q0')
    bm <- RunSVD(bm)
    bm <- RunUMAP(bm, reduction = 'lsi', dims = 2:50, reduction.name = "umap.atac", reduction.key = "atacUMAP_")
    
    bm <- FindMultiModalNeighbors(bm, reduction.list = list("pca", "lsi"), dims.list = list(1:50, 2:50))
    bm <- RunUMAP(bm, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_", return.model = TRUE)
    res <- find_fixed_clus(bm, n_cluster=n_cluster)
    bm <- FindClusters(bm, graph.name = "wsnn", algorithm = 3, resolution = res, verbose = FALSE)
    return(bm)
}

# load parameters from
args <- commandArgs()

n_cluster = as.integer(args[6])
rna_path = args[7]
atac_path = args[8]
adt_path = args[9]
save_path = args[10]
save_key = args[11]
hvg_num = as.integer(args[12])
batch_key = args[13]


begin_time <- Sys.time()
if (rna_path!="NULL"){rna <- h5_to_matrix(rna_path)} else{rna <- "NULL"}
if (adt_path!="NULL"){adt <- h5_to_matrix(adt_path)} else{adt <- "NULL"}
if (atac_path!="NULL"){atac <- h5_to_matrix(atac_path)} else{atac <- "NULL"}
if ((rna_path!="NULL") & (adt_path!="NULL")){result <- run_Seurat_RNA_ADT(rna,adt,hvg_num)}
if ((rna_path!="NULL") & (atac_path!="NULL")){result <- run_Seurat_RNA_ATAC(rna,atac,hvg_num)}

end_time <- Sys.time()
all_time <- difftime(end_time, begin_time, units="secs")

if (!dir.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
  print("path create")
}
# write.csv(all_time, paste0(save_path,"seurat_time.csv"))
wnn_graph = result[['weighted.nn']]
library(Matrix)

NeighborToSparseMatrices <- function(neighbor_obj) {
  idx <- Indices(neighbor_obj)      # n_cell × k
  dist <- Distances(neighbor_obj)   # n_cell × k
  cells <- Cells(neighbor_obj)
  
  n <- length(cells)
  k <- ncol(idx)

  i_idx <- rep(seq_len(n), each = k)
  
  j_idx <- as.vector(t(idx))
  
  x_dist <- as.vector(t(dist))

  x_conn <- rep(1, length(j_idx))
  
  dist_mat <- sparseMatrix(
    i = i_idx,
    j = j_idx,
    x = x_dist,
    dims = c(n, n),
    dimnames = list(cells, cells)
  )
  
  conn_mat <- sparseMatrix(
    i = i_idx,
    j = j_idx,
    x = x_conn,
    dims = c(n, n),
    dimnames = list(cells, cells)
  )
  
  return(list(
    distance_matrix = dist_mat,
    connection_matrix = conn_mat
  ))
}
graph = NeighborToSparseMatrices(wnn_graph)
dist_mat = graph$distance_matrix
conn_mat = graph$connection_matrix

library(rhdf5)

save_sparse_h5 <- function(mat, file, group_name = "matrix") {
  if (!inherits(mat, "dgCMatrix")) {
    mat <- as(mat, "dgCMatrix")
  }
  
  if (file.exists(file)) {
    file.remove(file)
  }
  
  h5createFile(file)
  h5createGroup(file, group_name)
  
  h5write(mat@i, file, paste0(group_name, "/i"))
  h5write(mat@p, file, paste0(group_name, "/p"))
  h5write(mat@x, file, paste0(group_name, "/x"))
  h5write(mat@Dim, file, paste0(group_name, "/dim"))
  
  rn <- rownames(mat)
  cn <- colnames(mat)
  
  if (!is.null(rn)) {
    h5write(rn, file, paste0(group_name, "/rownames"))
  }
  if (!is.null(cn)) {
    h5write(cn, file, paste0(group_name, "/colnames"))
  }
  
  h5write("dgCMatrix", file, paste0(group_name, "/class"))
}


save_sparse_h5(dist_mat, paste0(save_path, "seurat_wnn_distance.h5"), group_name = "matrix")
save_sparse_h5(conn_mat, paste0(save_path, "seurat_wnn_connection.h5"), group_name = "matrix")

umap <- as.data.frame(result@reductions$wnn.umap@cell.embeddings)
colnames(umap) <- c("UMAP1", "UMAP2")
umap[, "cluster"] <- result@meta.data$seurat_clusters
write.table(umap, 
            file = paste0(save_path, "/seurat_wnn.csv"), 
            row.names =T, col.names = T, sep=',', quote=F)