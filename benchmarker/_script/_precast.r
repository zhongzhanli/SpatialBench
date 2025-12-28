args <- commandArgs()

library(Seurat) 
library("PRECAST")
library(patchwork)
library(mclust)

n_cluster = as.integer(args[6])
file_path = args[7]
save_path = args[8]
save_key = args[9]
batch_key = args[10]
hvg_num = as.integer(args[11])
cluster_method = args[12]

call_precast <- function(file_path, save_path, n_cluster, batch_key, hvg_num, save_key, cluster_method){{
  obj <- readRDS(file_path)
  # obj <- RenameAssays(obj, assay.name = "Spatial", new.assay.name = "RNA", verbose=FALSE,)
  # obj$row <- obj$spatial1
  # obj$col <- obj$spatial2
  row <- obj$spatial1
  col <- obj$spatial2
  batch <- obj@meta.data[batch_key]

  obj <- CreateSeuratObject(counts = obj@assays$Spatial$counts)
  obj$row = row
  obj$col = col
  obj@meta.data[batch_key] = batch

  obj$nCount_RNA <- colSums(obj@assays$RNA$counts)
  obj$nFeature_RNA <- colSums(obj@assays$RNA$counts>0)

  # obj[["RNA"]] <- as(object = obj[["RNA"]], Class = "Assay5")

  datalist <- SplitObject(obj, split.by = batch_key)
  
  PRECASTObj <- CreatePRECASTObject(seuList = datalist, gene.number = hvg_num, selectGenesMethod = "Hvgs",
                                    premin.spots = -1, premin.features = -1, postmin.spots = -1, postmin.features = -1,
                                    rawData.preserve = TRUE, verbose=FALSE)

  PRECASTObj <- AddAdjList(PRECASTObj, platform = "Visium")
  PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = FALSE, int.model = NULL)
  PRECASTObj <- PRECAST(PRECASTObj, K = n_cluster)
  PRECASTObj <- SelectModel(PRECASTObj)
  seuInt <- IntegrateSpaData(PRECASTObj, species = "Unknown")
  seuInt <- AddUMAP(seuInt, n_comp = 2)

  write.csv(file=paste0(save_path,"/embed_",save_key,".csv"),seuInt@reductions$PRECAST@cell.embeddings)
  write.csv(file=paste0(save_path,"/umap_",save_key,".csv"),seuInt@reductions$UMAP@cell.embeddings)
  write.csv(file=paste0(save_path,"/batch_",save_key,".csv"), obj@meta.data[batch_key])
  write.csv(file=paste0(save_path,"/cluster_",save_key,".csv"),seuInt@meta.data["cluster"])
}}

call_precast(file_path=file_path, save_path=save_path, n_cluster=n_cluster, batch_key=batch_key, hvg_num=hvg_num, save_key=save_key, cluster_method=cluster_method)