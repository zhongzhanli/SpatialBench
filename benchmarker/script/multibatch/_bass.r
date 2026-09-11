args <- commandArgs()

library(Seurat) 
library(BASS)
library(Matrix)
library(patchwork)
library(mclust)

n_cluster = as.integer(args[6])
file_path = args[7]
save_path = args[8]
save_key = args[9]
batch_key = args[10]
hvg_num = as.integer(args[11])
cluster_method = args[12]

find_fixed_clus <- function(obj, n_cluster, vec = seq(0.1, 2.0, 0.1)) {
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
call_bass <- function(file_path, save_path, n_cluster, batch_key, hvg_num, save_key, cluster_method){
  obj <- readRDS(file_path)
  obj[["Spatial"]] <- as(object = obj[["Spatial"]], Class = "Assay5")
  datalist <- SplitObject(obj, split.by = batch_key)
  C <- 20
  R <- n_cluster
  cntm <- list()
  xym <- list()
  for(data in c(1:length(datalist))){
    cntm[[data]] = datalist[[data]]@assays$Spatial$counts
    xym[[data]] = datalist[[data]]$spatial@cell.embeddings
  }
  set.seed(0)
  BASS <- createBASSObject(cntm, xym, C = C, R = R,
    beta_method = "SW", init_method = "mclust", 
    nsample = 10000)
  BASS <- BASS.preprocess(BASS, doLogNormalize = TRUE,
        geneSelect = "hvgs", nSE = hvg_num, doPCA = TRUE, 
        scaleFeature = FALSE, nPC = 20)
  BASS <- BASS.run(BASS)
  BASS <- BASS.postprocess(BASS)

  embed = as.data.frame(t(BASS@X_run))
  rownames(embed) = colnames(obj)
  colnames(embed) <- paste0("BASS_", 1:length(colnames(embed)))

  obj[["BASS"]] <- CreateDimReducObject(embeddings = as.matrix(embed), key = "BASS_", assay=DefaultAssay(obj))
  obj <- RunUMAP(obj, reduction = "BASS", dims = 1:20, verbose=FALSE)

  cluster = as.data.frame(unlist(BASS@results$z))
  rownames(cluster) = rownames(embed)
  colnames(cluster) <- c("cluster")

  write.csv(file=paste0(save_path,"/embed_",save_key,".csv"),embed)
  write.csv(file=paste0(save_path,"/umap_",save_key,".csv"),obj@reductions$umap@cell.embeddings)
  write.csv(file=paste0(save_path,"/batch_",save_key,".csv"), obj@meta.data[batch_key])
  write.csv(file=paste0(save_path,"/cluster_",save_key,".csv"), cluster)
}

call_bass(file_path=file_path, save_path=save_path, n_cluster=n_cluster, batch_key=batch_key, hvg_num=hvg_num, save_key=save_key, cluster_method=cluster_method)