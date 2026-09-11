import scanpy as sc
import pandas as pd
import scanorama
import warnings
import sys
import anndata as ad
warnings.filterwarnings("ignore")

from utils import search_resolution, split_adata, mclust

args = sys.argv

file_path = args[1]
save_path = args[2]
n_cluster = int(args[3])
batch_key =  args[4]
hvg_num = int(args[5])
save_key = args[6]
cluster_method = args[7]

if __name__ == "__main__":

    adata = sc.read_h5ad(file_path)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_num, layer="counts", subset=True, batch_key=batch_key)
    sc.pp.log1p(adata)

    adata_list = split_adata(adata, batch_key=batch_key)
    corrected = scanorama.correct_scanpy(adatas=adata_list, return_dimred=True)
    adata_cor = ad.concat(corrected)
    sc.pp.neighbors(adata_cor, use_rep="X_scanorama")
    sc.tl.umap(adata_cor)
    if(cluster_method=="mclust"):
        try:
            adata_cor.obs["cluster"] = mclust(adata_cor.obsm["X_scanorama"], n_cluster=n_cluster)
        except:
            res = search_resolution(adata_cor, fixed_clus_count=n_cluster)
            sc.tl.leiden(adata_cor, resolution=res, key_added="cluster")
    else:
        res = search_resolution(adata_cor, fixed_clus_count=n_cluster)
        sc.tl.leiden(adata_cor, resolution=res, key_added="cluster")

    embed = pd.DataFrame(adata_cor.obsm["X_scanorama"])
    embed.index = list(adata_cor.obs_names)
    embed.to_csv(save_path+f"/embed_{save_key}.csv")
    umap = pd.DataFrame(adata_cor.obsm["X_umap"])
    umap.index = list(adata_cor.obs_names)
    umap.to_csv(save_path+"/umap_"+save_key+".csv")
    batch = pd.DataFrame(adata_cor.obs[batch_key])
    batch.index = list(adata_cor.obs_names)
    batch.to_csv(save_path+"/batch_"+save_key+".csv")
    cluster = pd.DataFrame(adata_cor.obs["cluster"])
    cluster.index = list(adata_cor.obs_names)
    cluster.to_csv(save_path+"/cluster_"+save_key+".csv")