import scanpy as sc
import numpy as np
import pandas as pd
import sys
import scvi

from utils import search_resolution, mclust

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
    adata.raw = adata 
    sc.pp.normalize_total(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_num, layer="counts",
                                subset=True, batch_key=batch_key)
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key, layer="counts")
    vae = scvi.model.SCVI(adata)
    vae.train()
    adata.obsm["X_scVI"] = vae.get_latent_representation()
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata)
    if(cluster_method=="mclust"):
        adata.obs["cluster"] = mclust(adata.obsm["X_scVI"], n_cluster=n_cluster)
    elif(cluster_method=="leiden"):
        res = search_resolution(adata, fixed_clus_count=n_cluster)
        sc.tl.leiden(adata, resolution=res, key_added="cluster")
    embed = pd.DataFrame(adata.obsm["X_scVI"])
    embed.index = list(adata.obs_names)
    embed.to_csv(save_path+f"/embed_{save_key}.csv")
    umap = pd.DataFrame(adata.obsm["X_umap"])
    umap.index = list(adata.obs_names)
    umap.to_csv(save_path+"/umap_"+save_key+".csv")
    batch = pd.DataFrame(adata.obs[batch_key])
    batch.index = list(adata.obs_names)
    batch.to_csv(save_path+"/batch_"+save_key+".csv")
    cluster = pd.DataFrame(adata.obs["cluster"])
    cluster.index = list(adata.obs_names)
    cluster.to_csv(save_path+"/cluster_"+save_key+".csv")