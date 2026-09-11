args <- commandArgs()

library(Seurat) 
library(patchwork)
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

call_seurat <- function(file_path, save_path, n_cluster, batch_key, hvg_num, save_key, cluster_method){
  obj <- readRDS(file_path)
  obj[["Spatial"]] <- as(object = obj[["Spatial"]], Class = "Assay5")

  datalist <- SplitObject(obj, split.by = batch_key)
  
  datalist <- lapply(X = datalist, FUN = function(x) {
    x <- NormalizeData(x, verbose=FALSE)
    #x <- FindVariableFeatures(x, nfeatures = hvg_num)
  })
  features <- SelectIntegrationFeatures(object.list = datalist, nfeatures=hvg_num, verbose=FALSE)
  
  immune.anchors <- FindIntegrationAnchors(object.list = datalist, anchor.features = features, verbose=FALSE)
  immune.combined <- IntegrateData(anchorset = immune.anchors, verbose=FALSE)
  
  DefaultAssay(immune.combined) <- "integrated"
  
  immune.combined <- ScaleData(immune.combined, verbose = FALSE)
  immune.combined <- RunPCA(immune.combined, verbose = FALSE)
  immune.combined <- RunUMAP(immune.combined, reduction = "pca", dims = 1:30, verbose=FALSE)

  if(cluster_method=="leiden"){
    immune.combined <- FindNeighbors(immune.combined, dims = 1:30, reduction = "pca", verbose=FALSE)
    res <- find_fixed_clus(immune.combined, n_cluster = n_cluster)
    immune.combined <- FindClusters(immune.combined, resolution = res, verbose=FALSE)
  }
  else{
    clust <- Mclust(immune.combined@reductions$pca@cell.embeddings[,1:30], n_cluster, modelNames = "EEE")
    immune.combined$seurat_clusters <- clust$classification
  }

  write.csv(file=paste0(save_path,"/embed_",save_key,".csv"),immune.combined@reductions$pca@cell.embeddings)
  write.csv(file=paste0(save_path,"/umap_",save_key,".csv"),immune.combined@reductions$umap@cell.embeddings)
  write.csv(file=paste0(save_path,"/batch_",save_key,".csv"), immune.combined@meta.data[batch_key])
  write.csv(file=paste0(save_path,"/cluster_",save_key,".csv"),immune.combined@meta.data["seurat_clusters"])
}

call_seurat(file_path=file_path, save_path=save_path, n_cluster=n_cluster, batch_key=batch_key, hvg_num=hvg_num, save_key=save_key, cluster_method=cluster_method)