import sys
sys.path.append('../../Benchmark/external/STAligner/')
import STAligner
import scanpy as sc
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.linalg
import numpy as np
import torch
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

    used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    adata = sc.read_h5ad(file_path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    adata.X = sp.csr_matrix(adata.X)

    Batch_list = []
    adj_list = []

    adata_list = split_adata(adata, batch_key=batch_key)
    for i in adata_list:
        STAligner.Cal_Spatial_Net(i, k_cutoff=knn_cutoff, model="KNN")
        sc.pp.highly_variable_genes(i, flavor="seurat_v3", n_top_genes=hvg_num) 
        sc.pp.normalize_total(i, target_sum=1e4)
        sc.pp.log1p(i)
        i = i[:, i.var['highly_variable']]

        adj_list.append(i.uns['adj'])
        Batch_list.append(i)
    adata_concat = ad.concat(Batch_list)
    adata_concat.obs["batch_name"] = adata_concat.obs[batch_key].astype('category')
    adj_concat = np.asarray(adj_list[0].todense())
    for batch_id in range(1,len(Batch_list)):
        adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
    adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

    adata_concat = STAligner.train_STAligner(adata_concat, verbose=False, knn_neigh = 50, device=used_device)
    sc.pp.neighbors(adata_concat, use_rep='STAligner', random_state=666)
    sc.tl.umap(adata_concat, random_state=666)
    if(cluster_method=="mclust"):
        adata_concat.obs["cluster"] = mclust(adata_concat.obsm["STAligner"], n_cluster=n_cluster)
    else:
        res = search_resolution(adata_concat, fixed_clus_count=n_cluster)
        sc.tl.leiden(adata_concat, resolution=res, key_added="cluster")

    embed = pd.DataFrame(adata_concat.obsm["STAligner"])
    embed.index = list(adata_concat.obs_names)
    embed.to_csv(save_path+f"/embed_{save_key}.csv")
    umap = pd.DataFrame(adata_concat.obsm["X_umap"])
    umap.index = list(adata_concat.obs_names)
    umap.to_csv(save_path+"/umap_"+save_key+".csv")
    batch = pd.DataFrame(adata_concat.obs[batch_key])
    batch.index = list(adata_concat.obs_names)
    batch.to_csv(save_path+"/batch_"+save_key+".csv")
    cluster = pd.DataFrame(adata_concat.obs["cluster"])
    cluster.index = list(adata_concat.obs_names)
    cluster.to_csv(save_path+"/cluster_"+save_key+".csv")
