library(rhdf5)
library(MOFA2)
library(reshape2)
library(HDF5Array)
library(Matrix)
library(MOFA2)
library(Seurat)
library(Signac)
library(reticulate)
conda_path <- c("/mnt/datadisk/lizhongzhan/miniconda3/envs/mofa2_env/")
use_condaenv(conda_path)
options(Seurat.object.assay.version = "v3")

h5_to_matrix <- function(path, sparse = TRUE) {
  suppressPackageStartupMessages({
    library(rhdf5)
    library(Matrix)
  })
    x      <- h5read(path, "matrix/data")
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

    return(data)
}

read_h5_batch <- function(path, batch_key) {
  h5f <- H5Fopen(path, flags="H5F_ACC_RDONLY")

  if (H5Lexists(h5f, paste0("matrix/", batch_key))) {
      batches <- h5read(h5f, paste0("matrix/", batch_key))
  }else{
      batches <- NULL
  }

  H5Fclose(h5f)
  return(batches)
}

make_unique_colnames <- function(colnames) {
  new_colnames <- colnames
  dupes <- colnames[duplicated(colnames)]
  for (dupe in unique(dupes)) {
    idx <- which(colnames == dupe)
    new_colnames[idx] <- paste0(dupe, "_", seq_along(idx))
  }
  return(new_colnames)
}

run_MOFA2 <- function(RNA_file_path, ATAC_file_path, ADT_file_path, batch_key, plot=FALSE){
    if (!is.null(RNA_file_path) & !is.null(ADT_file_path) & is.null(ATAC_file_path)){
        rna <- h5_to_matrix(RNA_file_path)
        adt <- h5_to_matrix(ADT_file_path)

        mofa_data <- CreateSeuratObject(counts = rna)
        DefaultAssay(mofa_data) <- "RNA"
        mofa_data <- SCTransform(mofa_data, verbose = FALSE, variable.features.n=hvg_num) %>% RunPCA() %>% RunUMAP(dims = 1:50, reduction.name = 'umap.rna', reduction.key = 'rnaUMAP_')
        mofa_data <- FindVariableFeatures(mofa_data, selection.method = "vst", nfeatures = 3000) 
        mofa_data[["ADT"]] <- CreateAssayObject(counts = adt)

        DefaultAssay(mofa_data) <- 'ADT'
        # we will use all ADT features for dimensional reduction
        VariableFeatures(mofa_data) <- rownames(mofa_data[["ADT"]])
        batches <- read_h5_batch(RNA_file_path, batch_key)
        mofa_data <- NormalizeData(mofa_data, normalization.method = 'CLR', margin = 2) %>% ScaleData() %>% RunPCA(reduction.name = 'apca')

        # Merge Data
        if(is.null(batches)){
          MOFAobject <- create_mofa(mofa_data, assays = c("SCT","ADT"))
        }else{
          MOFAobject <- create_mofa(mofa_data, assays = c("SCT","ADT"), groups=batches)
        }
    }
    if (!is.null(RNA_file_path) & is.null(ADT_file_path) & !is.null(ATAC_file_path)){
        rna <- h5_to_matrix(RNA_file_path)
        atac <- h5_to_matrix(ATAC_file_path)
        batches <- read_h5_batch(RNA_file_path, batch_key)
        # stop(is.null(batches))

        multi_data <- CreateSeuratObject(counts = rna, assay = "RNA")
        atac_assay <- CreateAssayObject(counts = atac)
        multi_data[["ATAC"]] <- atac_assay

        DefaultAssay(multi_data) <- "RNA"
        multi_data <- NormalizeData(multi_data, normalization.method = "LogNormalize", assay = "RNA")
        multi_data <- ScaleData(multi_data, do.center = TRUE, do.scale = FALSE, assay = "RNA")
        ## ATAC
        DefaultAssay(multi_data) <- "ATAC"
        multi_data <- NormalizeData(multi_data)
        multi_data <- ScaleData(multi_data)
        multi_data <- RunTFIDF(multi_data, assay = "ATAC")

        # Feature selection
        ## RNA
        DefaultAssay(multi_data) <- "RNA"
        multi_data <- FindVariableFeatures(multi_data, selection.method = "vst", nfeatures = hvg_num, assay = "RNA") 
        ## ATAC
        DefaultAssay(multi_data) <- "ATAC"
        # FindTopFeatures(multi_data, min.cutoff = 2000, assay = "ATAC")
        multi_data <- FindTopFeatures(multi_data, min.cutoff = 'q80', assay = "ATAC")

        # Merge Data
        
        if(is.null(batches)){
          MOFAobject <- create_mofa(multi_data, assays = c("RNA","ATAC"))
        }else{
          MOFAobject <- create_mofa(multi_data, assays = c("RNA","ATAC"), groups=batches)
        }
    
    }


    # if (is.null(processed_data$rna) & !is.null(processed_data$adt) & !is.null(processed_data$atac)){
    #   data <- c(list(view_1 = processed_data$adt), list(view_2 = processed_data$atac))

    # if (!is.null(processed_data$rna) & !is.null(processed_data$adt) & !is.null(processed_data$atac)){
    #   data <- c(list(view_1 = processed_data$rna), list(view_2 = processed_data$adt), list(view_3 = processed_data$atac))}
  
    # new_colname <- make_unique_colnames(colnames(data[[1]]))
    # for(view in names(data)) {
    #   colnames(data[[view]]) <- new_colname
    # }
    # groups <- processed_data$batch
    # MOFAobject <- create_mofa_from_matrix(data, groups)
    # MOFAobject <- create_mofa(data,)# groups=groups)
    data_opts <- get_default_data_options(MOFAobject)
    model_opts <- get_default_model_options(MOFAobject)
    train_opts <- get_default_training_options(MOFAobject)
    train_opts$seed <- 42
    train_opts$gpu_mode <- TRUE
    MOFAobject <- prepare_mofa(
      object = MOFAobject,
      data_options = data_opts,
      model_options = model_opts,
      training_options = train_opts
    )
    MOFAobject.trained <- run_mofa(MOFAobject, use_basilisk=FALSE) #TRUE)
    return (MOFAobject.trained)
}

# input data
# load parameters from
args <- commandArgs()

n_cluster = as.integer(args[6])
rna_path = args[7]
atac_path = args[8]
adt_path = args[9]
save_path = args[10]
save_key = args[11] # NOT USED
hvg_num = as.integer(args[12])
batch_key = args[13] # NOT USED

# run methods
begin_time <- Sys.time()
if (atac_path[1]=="NULL"){
    MOFAobject.trained <- run_MOFA2(rna_path, NULL, adt_path, batch_key, plot = FALSE)
}else if (adt_path[1]=="NULL"){
    MOFAobject.trained <- run_MOFA2(rna_path, atac_path, NULL, batch_key, plot = FALSE)
}
# else if (rna_path[1]=="NULL"){
#     MOFAobject.trained <- run_MOFA2(RNA_file_path, ATAC_file_path, NULL, plot = FALSE)
# }
# }else {
# processed_data <- read_h5_data(file_paths$rna_path, file_paths$adt_path, file_paths$atac_path)
# }
# MOFAobject.trained <- run_MOFA2(processed_data, plot = FALSE)
MOFAobject.trained <- run_umap(MOFAobject.trained,
                      n_neighbors = 20, 
                      min_dist = 0.30

)
mofa_cluster <- cluster_samples(MOFAobject.trained, k=n_cluster, 
                                    factors = "all")

mofa_cluster <- as.data.frame(mofa_cluster$cluster)
colnames(mofa_cluster) <- "cluster"
umap <- cbind(MOFAobject.trained@dim_red$UMAP, mofa_cluster)
umap <- umap[, -1]
write.table(umap, file = paste0(save_path, "mofa2.csv"), 
                row.names =T, col.names = T, sep=',', quote=F)
batches <- read_h5_batch(rna_path, batch_key)
if(is.null(batches)){
  write.table(MOFAobject.trained@expectations$Z$group1,
        file = paste0(save_path, "mofa2_latent.csv"), , 
        row.names =T, col.names = T, sep=',', quote=F)
}else{
  embedding <- c()
  for (i in unique(batches)){
    embedding <- rbind(embedding, MOFAobject.trained@expectations$Z[[i]])
  }
  write.table(embedding,
        file = paste0(save_path, "mofa2_latent.csv"), , 
        row.names =T, col.names = T, sep=',', quote=F)
}


        