import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
sys.path.append("../../Benchmark/external/SEDR/")
import SEDR
import scanpy as sc
import pandas as pd
from sklearn.decomposition import PCA 
import harmonypy as hm
from scipy.cluster.vq import kmeans2
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
    adata_list = split_adata(adata, batch_key)

    for i in range(len(adata_list)):
        t_ad = adata_list[i]
        t_ad.var_names_make_unique()
        graph_dict_tmp = SEDR.graph_construction(t_ad, n=knn_cutoff)

        if i==0:
            adata_concat = t_ad
            graph_dict = graph_dict_tmp
        else:
            var_names = adata_concat.var_names.intersection(t_ad.var_names)
            adata_concat=adata_concat[:, var_names]
            t_ad = t_ad[:, var_names]
            adata_concat = adata_concat.concatenate(t_ad, batch_key="temp")
            graph_dict = SEDR.combine_graph_dict(graph_dict, graph_dict_tmp)

    adata_concat.layers['count'] = adata_concat.X
    sc.pp.normalize_total(adata_concat, target_sum=1e6)
    sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", layer='count', n_top_genes=hvg_num, batch_key=batch_key)
    adata_concat = adata_concat[:, adata_concat.var['highly_variable'] == True]
    sc.pp.scale(adata_concat)

    n_components = min(int(adata.shape[1]/2), 200)
    adata_X = PCA(n_components=n_components, random_state=42).fit_transform(adata_concat.X)
    adata_concat.obsm['X_pca'] = adata_X
    sedr_net = SEDR.Sedr(adata_concat.obsm['X_pca'], graph_dict, mode='clustering', device='cuda:0')
    using_dec = False
    if using_dec:
        sedr_net.train_with_dec()
    else:
        sedr_net.train_without_dec()

    sedr_feat, _, _, _ = sedr_net.process()
    adata_concat.obsm['SEDR'] = sedr_feat
    meta_data = adata_concat.obs[[batch_key]]
    data_mat = adata_concat.obsm['SEDR']
    vars_use = [batch_key]

    def cluster_fn(data, K):
        centroid, label = kmeans2(data, K, minit='++')
        return centroid
    ho = hm.run_harmony(data_mat, meta_data, vars_use, cluster_fn=cluster_fn)

    res = pd.DataFrame(ho.Z_corr).T
    res_df = pd.DataFrame(data=res.values, columns=['X{}'.format(i+1) for i in range(res.shape[1])], index=adata_concat.obs.index)
    adata_concat.obsm[f'SEDR.Harmony'] = res_df.to_numpy()
    sc.pp.neighbors(adata_concat, use_rep='SEDR.Harmony', metric='cosine')
    sc.tl.umap(adata_concat)
    if(cluster_method=="mclust"):
        adata_concat.obs["cluster"] = mclust(adata_concat.obsm['SEDR.Harmony'], n_cluster=n_cluster)
    elif(cluster_method=="leiden"):
        res = search_resolution(adata_concat, fixed_clus_count=n_cluster)
        sc.tl.leiden(adata_concat, resolution=res, key_added="cluster")

    embed = pd.DataFrame(adata_concat.obsm["SEDR.Harmony"])
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
