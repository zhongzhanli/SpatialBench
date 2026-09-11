library(reticulate)
conda_path <- c("/mnt/datadisk/lizhongzhan/miniconda3/envs/mofa2_env/")
use_condaenv(conda_path)

suppressPackageStartupMessages({
  library(MOFA2)
  library(dplyr)
  library(ggplot2)
  library(data.table)
  library(Matrix)
  library(stringr)
  library(Seurat)
  library(Signac)
})

options(stringsAsFactors = FALSE)

# =======================
# 1. 参数读取
# =======================
args <- commandArgs(trailingOnly = TRUE)

h5_to_matrix <- function(path, sparse = TRUE) {
  suppressPackageStartupMessages({
    library(rhdf5)
    library(Matrix)
  })

  x       <- h5read(path, "matrix/data")
  indices <- h5read(path, "matrix/indices")
  indptr  <- h5read(path, "matrix/indptr")
  shape   <- h5read(path, "matrix/shape")

  feature <- h5read(path, "matrix/features")
  barcode <- h5read(path, "matrix/barcodes")

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

  spatial <- t(h5read(path, "matrix/spatial"))
  spatial <- as.matrix(spatial)
  if (nrow(spatial) != n_cell) {
    stop(sprintf(
        "spatial rows (%d) do not match number of cells (%d).",
        nrow(spatial), n_cell
        ))
  }

  rownames(spatial) <- colnames(data)

  if (ncol(spatial) == 2) {
    colnames(spatial) <- c("x", "y")
  }

  return(list(
    mat = data,
    spatial = spatial
  ))
}

if (length(args) < 8) {
  stop(
    paste0(
      "Expected 8 arguments:\n",
      "1) n_cluster\n",
      "2) rna_file_path\n",
      "3) atac_file_path\n",
      "4) adt_file_path\n",
      "5) save_path\n",
      "6) save_key\n",
      "7) hvg_num\n",
      "8) batch_key\n"
    )
  )
}

n_cluster     <- as.integer(args[1])
rna_file_path <- args[2]
atac_file_path<- args[3]
adt_file_path <- args[4]
save_path     <- args[5]
save_key      <- args[6]
hvg_num       <- as.integer(args[7])
batch_key     <- args[8]

# =======================
# 2. 工具函数
# =======================
is_null_string <- function(x) {
  toupper(x) == "NULL"
}

dir.create(save_path, recursive = TRUE, showWarnings = FALSE)

message("[Info] n_cluster = ", n_cluster)
message("[Info] rna_file_path = ", rna_file_path)
message("[Info] atac_file_path = ", atac_file_path)
message("[Info] adt_file_path = ", adt_file_path)
message("[Info] save_path = ", save_path)
message("[Info] save_key = ", save_key)
message("[Info] hvg_num = ", hvg_num)
message("[Info] batch_key = ", batch_key, " (currently not used)")

# =======================
# 3. 模式判断
# =======================
if (!is_null_string(atac_file_path) && is_null_string(adt_file_path)) {
  mode <- "atac"
}
if (is_null_string(atac_file_path) && !is_null_string(adt_file_path)) {
  mode <- "protein"
} else {
  stop("Input error: atac_file_path and adt_file_path must have exactly one 'NULL'.")
}

message("[Info] inferred mode = ", mode)

# =======================
# 4. 读取数据
# 注意：这里按原始示例保留为 RDS 输入
# =======================
message("[1/6] Loading data ...")

rna_counts <- h5_to_matrix(rna_file_path)$mat
if(mode=="protein"){
    adt_counts <- h5_to_matrix(adt_file_path)$mat
}else{
    adt_counts <- h5_to_matrix(atac_file_path)$mat
}

# =======================
# 5. 提取空间坐标
# =======================
message("[2/6] Extracting spatial coordinates ...")

spatial_loc <- h5_to_matrix(rna_file_path)$spatial[, c("x", "y"), drop = FALSE]

# =======================
# 6. 构建多模态 Seurat 对象
# 按你原始示例保留 Spatial_RNA / Spatial_ADT assay 命名
# =======================
message("[3/6] Building multi-modal Seurat object ...")


obj.multi <- CreateSeuratObject(
  counts = rna_counts
)
obj.multi[["ADT"]] <- CreateAssayObject(
  counts = adt_counts
)

# # 确保 cell 顺序一致
# common_cells <- intersect(colnames(obj.multi), colnames(adt_counts))
# obj.multi <- subset(obj.multi, cells = common_cells)

# =======================
# 7. RNA / ADT 预处理
# =======================
message("[4/6] Preprocessing RNA and ADT ...")

# RNA
DefaultAssay(obj.multi) <- "RNA"
obj.multi <- SCTransform(obj.multi, verbose = FALSE, variable.features.n=hvg_num)
obj.multi <- RunPCA(obj.multi, npcs = 50, verbose = FALSE)
obj.multi <- RunUMAP(
  obj.multi,
  dims = 1:50,
  reduction = "pca",
  reduction.name = "umap.rna",
  reduction.key = "rnaUMAP_",
  verbose = FALSE
)
obj.multi <- FindVariableFeatures(
  obj.multi,
  selection.method = "vst",
  nfeatures = hvg_num,
  verbose = FALSE
)

# ADT
DefaultAssay(obj.multi) <- "ADT"
if(mode=="protein"){
obj.multi <- NormalizeData(
  obj.multi,
  normalization.method = "CLR",
  margin = 2,
  verbose = FALSE
)
obj.multi <- ScaleData(obj.multi, verbose = FALSE)
obj.multi <- FindVariableFeatures(obj.multi, verbose = FALSE)
}else{
    obj.multi <- NormalizeData(obj.multi)
    obj.multi <- ScaleData(obj.multi)
    obj.multi <- RunTFIDF(obj.multi, assay = "ADT")
    obj.multi <- FindTopFeatures(obj.multi, min.cutoff = 'q80', assay = "ADT")
}


# =======================
# 8. 构建并训练 MOFA/MEFISTO
# =======================
message("[5/6] Training MEFISTO ...")

mofa <- create_mofa(obj.multi, assays = c("SCT", "ADT"))

spatial_loc2 <- data.frame(spatial_loc)
spatial_loc2 <- spatial_loc2[colnames(obj.multi), , drop = FALSE]
rownames(spatial_loc2) <- colnames(obj.multi)
colnames(spatial_loc2) <- c("coord1", "coord2")

mofa <- set_covariates(mofa, t(spatial_loc2))

data_opts <- get_default_data_options(mofa)

model_opts <- get_default_model_options(mofa)
model_opts$num_factors <- max(2, n_cluster)

train_opts <- get_default_training_options(mofa)
train_opts$maxiter <- 1000

mefisto_opts <- get_default_mefisto_options(mofa)

mofa <- prepare_mofa(
  object = mofa,
  model_options = model_opts,
  mefisto_options = mefisto_opts,
  training_options = train_opts,
  data_options = data_opts
)

model_file <- file.path(save_path, paste0(save_key, "_model.hdf5"))

mofa <- run_mofa(
  object = mofa,
  outfile = model_file,
  use_basilisk = FALSE
)

mofa <- load_model(model_file, remove_inactive_factors = FALSE)

factors <- 1:get_dimensions(mofa)[["K"]]

mofa <- run_umap(
  mofa,
  factors = factors,
  n_neighbors = 15,
  min_dist = 0.30
)

# =======================
# 9. 提取 embedding 并聚类
# =======================

mofa_cluster <- cluster_samples(mofa, k=n_cluster, factors = "all")
mofa_cluster <- as.data.frame(mofa_cluster$cluster)
colnames(mofa_cluster) <- "cluster"
umap <- cbind(mofa@dim_red$UMAP, mofa_cluster)
umap <- umap[, -1]

write.table(umap, file = paste0(save_path, "mefisto.csv"), 
                row.names =T, col.names = T, sep=',', quote=F)

write.table(mofa@expectations$Z$group1,
        file = paste0(save_path, "mefisto_latent.csv"), , 
        row.names =T, col.names = T, sep=',', quote=F)