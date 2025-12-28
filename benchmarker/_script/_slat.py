import sys
sys.path.append('../../Benchmark/external/SLAT/')
import scSLAT
from scSLAT.model import load_anndatas, Cal_Spatial_Net, run_SLAT, run_SLAT_multi
import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
import warnings
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
knn_cutoff = int(args[8])

if __name__ == "__main__":

    adata = sc.read_h5ad(file_path)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_num, batch_key=batch_key)
    adata.layers["counts"] = adata.X.copy()
    adata_list = split_adata(adata, batch_key)
    if(len(adata_list)>2):
        raise ValueError("SLAT does not support more than 2 dataset")
    else:
        for i in adata_list:
            Cal_Spatial_Net(i, k_cutoff=knn_cutoff, model='KNN')

        edges, features = load_anndatas([adata_list[0], adata_list[1]], feature='DPCA', check_order=False)
        embd0, embd1, time = run_SLAT(features, edges)

    embedding1 = pd.DataFrame(embd0.cpu().detach().numpy())
    embedding2 = pd.DataFrame(embd1.cpu().detach().numpy())
    embedding = pd.concat([embedding1, embedding2])
    adata_cor = ad.concat([adata_list[0], adata_list[1]])
    adata_cor.obsm["X_embed"] = np.array(embedding)
    sc.pp.neighbors(adata_cor, use_rep="X_embed")
    sc.tl.umap(adata_cor)
    if(cluster_method=="mclust"): # mclust for SLAT will cost too many time
        res = search_resolution(adata_cor, fixed_clus_count=n_cluster)
        sc.tl.leiden(adata_cor, resolution=res, key_added="cluster")
        # adata_cor.obs["cluster"] = mclust(adata_cor.obsm["X_embed"], n_cluster=n_cluster)
    elif(cluster_method=="leiden"):
        res = search_resolution(adata_cor, fixed_clus_count=n_cluster)
        sc.tl.leiden(adata_cor, resolution=res, key_added="cluster")

    embed = pd.DataFrame(adata_cor.obsm["X_embed"])
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
