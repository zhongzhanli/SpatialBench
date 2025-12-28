args <- commandArgs()

library(Seurat) 
library(patchwork)
library(dplyr)
library(harmony)
library(mclust)

n_cluster = as.integer(args[6])
file_path = args[7]
save_path = args[8]
save_key = args[9]
batch_key = args[10]
hvg_num = as.integer(args[11])
cluster_method = args[12]

find_fixed_clus <- function(obj, n_cluster, vec = seq(0.1, 2.0, 0.05)) {
  closest_res <- NA
  closest_diff <- Inf
  
  for (i in vec) {
    obj <- FindClusters(obj, resolution = i, verbose = FALSE)
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

call_harmony <- function(file_path, save_path, n_cluster, batch_key, hvg_num, save_key, cluster_method){
  obj <- readRDS(file_path)
  obj[["Spatial"]] <- as(object = obj[["Spatial"]], Class = "Assay5")
  obj <- NormalizeData(obj, verbose=FALSE) %>% FindVariableFeatures(nfeatures = hvg_num, verbose=FALSE) %>% ScaleData(verbose=FALSE) %>% RunPCA(verbose=FALSE)
  
  obj <- RunHarmony(obj, group.by.vars = batch_key, verbose=FALSE)
  obj <- RunUMAP(obj, reduction = "harmony", dims = 1:30, verbose=FALSE)

  if(cluster_method=="leiden"){
    obj <- FindNeighbors(obj, dims = 1:30, reduction = "harmony")
    res <- find_fixed_clus(obj, n_cluster = n_cluster)
    obj <- FindClusters(obj, resolution = res, verbose=FALSE)
  }
  else{
    clust <- Mclust(obj@reductions$harmony@cell.embeddings[,1:30], n_cluster, modelNames = "EEE")
    obj$seurat_clusters <- clust$classification
  }
  
  write.csv(file=paste0(save_path,"/embed_",save_key,".csv"),obj@reductions$harmony@cell.embeddings)
  write.csv(file=paste0(save_path,"/umap_",save_key,".csv"),obj@reductions$umap@cell.embeddings)
  write.csv(file=paste0(save_path,"/batch_",save_key,".csv"), obj@meta.data[batch_key])
  write.csv(file=paste0(save_path,"/cluster_",save_key,".csv"),obj@meta.data["seurat_clusters"])
}

call_harmony(file_path=file_path, save_path=save_path, n_cluster=n_cluster, batch_key=batch_key, hvg_num=hvg_num, save_key=save_key, cluster_method=cluster_method)