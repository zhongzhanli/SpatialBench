import sys
sys.path.append("../../Benchmark/external/Graspot/")
from Graspot import Cal_Spatial_Net, train_Graspot_Sub
import scanpy as sc
import pandas as pd
import warnings
import sys
import scipy.sparse as sp
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
    adata.layers['count'] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=hvg_num, subset=True, batch_key=batch_key)
    if(not sp.issparse(adata.X)):
        adata.X = sp.csr_matrix(adata.X)
    adata_list = split_adata(adata)
    adata.obs["batch_name"] = list(adata.obs[batch_key])
    batch_size = len(adata_list)
    iter_comb = []
    for i in range(1, batch_size):
        iter_comb.append([0, i])
    
    for t_ad in adata_list:
        Cal_Spatial_Net(t_ad)

    adata, _ = train_Graspot_Sub(adata=adata, Batch_list=adata_list, iter_comb=iter_comb, device="cpu")
    sc.pp.neighbors(adata, use_rep="Graspot")
    sc.tl.umap(adata)
    if(cluster_method=="mclust"):
        try:
            adata.obs["cluster"] = mclust(adata.obsm["Graspot"], n_cluster=n_cluster)
        except:
            res = search_resolution(adata, fixed_clus_count=n_cluster)
            sc.tl.leiden(adata, resolution=res, key_added="cluster")
    elif(cluster_method=="leiden"):
        res = search_resolution(adata, fixed_clus_count=n_cluster)
        sc.tl.leiden(adata, resolution=res, key_added="cluster")

    embed = pd.DataFrame(adata.obsm["Graspot"])
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

