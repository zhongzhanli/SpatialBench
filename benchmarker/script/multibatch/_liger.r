args <- commandArgs()

library(Seurat)
library(patchwork)
library(rliger)
library(dplyr)
library(mclust)

n_cluster = as.integer(args[6])
file_path = args[7]
save_path = args[8]
save_key = args[9]
batch_key = args[10]
hvg_num = as.integer(args[11])
cluster_method = args[12]
nFactors = as.integer(args[13])

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

call_liger <- function(file_path, save_path, n_cluster, batch_key, hvg_num, save_key, nFactors, cluster_method){
  obj <- readRDS(file_path)
  datalist <- SplitObject(obj, split.by = batch_key)
  for(i in c(1:length(datalist))){
    datalist[[i]] = datalist[[i]]@assays$Spatial$counts
  }
  lobj <- createLiger(datalist)
  lobj <- lobj %>%
    normalize(verbose=FALSE) %>%
    selectGenes(num.genes=hvg_num) %>%
    scaleNotCenter(verbose=FALSE)

  lobj <- optimizeALS(lobj, k=nFactors, verbose=FALSE)
  lobj <- quantile_norm(lobj)

  iNMF_embed  <-  lobj@H.norm
  colnames(iNMF_embed) <- paste0("iNMF_", 1:nFactors)
  obj[["iNMF"]] <- CreateDimReducObject(embeddings = iNMF_embed, key = "iNMF_", assay=DefaultAssay(obj))
  
  if(cluster_method=="leiden"){
    obj <- FindNeighbors(obj, dims = 1:nFactors, reduction = "iNMF", verbose=FALSE)
    res <- find_fixed_clus(obj, n_cluster = n_cluster)
    obj <- FindClusters(obj, resolution = res, verbose=FALSE)
  }
  else{
    clust <- Mclust(obj@reductions$iNMF@cell.embeddings[,1:nFactors], n_cluster, modelNames = "EEE")
    obj$seurat_clusters <- clust$classification
  }

  obj <- RunUMAP(obj, reduction = "iNMF", dims = 1:nFactors, verbose=FALSE)
 
  write.csv(file=paste0(save_path,"/embed_",save_key,".csv"),obj@reductions$iNMF@cell.embeddings)
  write.csv(file=paste0(save_path,"/umap_",save_key,".csv"),obj@reductions$umap@cell.embeddings)
  write.csv(file=paste0(save_path,"/batch_",save_key,".csv"), obj@meta.data[batch_key])
  write.csv(file=paste0(save_path,"/cluster_",save_key,".csv"), obj@meta.data["seurat_clusters"])
}

call_liger(file_path=file_path, save_path=save_path, n_cluster=n_cluster, 
batch_key=batch_key, hvg_num=hvg_num, save_key=save_key, nFactors=nFactors, cluster_method=cluster_method)