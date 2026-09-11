args <- commandArgs()

library(Seurat)
library(patchwork)

file_path = args[6]
save_path = args[7]
batch_key = args[8]

obj <- readRDS(file_path)

# metadata <- obj@meta.data

# obj <- CreateSeuratObject(counts = obj@assays$Spatial$counts, , min.cells = 0, min.features = 0)
# obj@meta.data = metadata

datalist <- SplitObject(obj, split.by = batch_key)

cluster_cols <- grep("_cluster$", colnames(obj@meta.data), value = TRUE)

for(col in cluster_cols){
    for(data in datalist){
        Idents(data) <- col
        batch_name = data@meta.data[batch_key][1,]
        markers  <- FindAllMarkers(data, test.use="roc", min.pct=0, min.diff.pct=0, logfc.threshold=0,
        slot="counts", return.thresh=-100, min.cells.feature=0)
        # print(paste0(save_path,"/",col,"_",batch_name,".csv"))
        write.csv(markers, file=paste0(save_path,"/",col,"_",batch_name,".csv"))
    }
}
