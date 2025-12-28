import sys
sys.path.append("../../Benchmark/external/DeepST/")
import deepstkit as dt
import os
import scanpy as sc
import pandas as pd
import shutil
from utils import search_resolution, split_adata, mclust
args = sys.argv

file_path = args[1]
save_path = args[2]
n_cluster = int(args[3])
batch_key =  args[4]
hvg_num = int(args[5])
save_key = args[6]
cluster_method = args[7]
# knn_cutoff = int(args[8])

if __name__ == "__main__":

    dt.utils_func.seed_torch(seed=0)
    adata = sc.read_h5ad(file_path)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_num, subset=True, batch_key=batch_key)

    if(not os.path.exists(f"{save_path}/bm_temp_deepst")):
        os.mkdir(f"{save_path}/bm_temp_deepst")
    temp_dir = f"{save_path}/bm_temp_deepst"

    integration_model = dt.main.run(
        save_path=temp_dir,
        task="Integration", 
        pre_epochs=500,           
        epochs=500,              
        use_gpu=True              
    )
    adata_list = split_adata(adata, batch_key)
    processed_data = []
    spatial_graphs = []

    for ad in adata_list:

        ad = integration_model._get_augment(
            ad,
            spatial_type="BallTree",
            use_morphological=False,
        )
        
        graph = integration_model._get_graph(
            ad.obsm["spatial"],
            distType="KDTree"
        )
        
        processed_data.append(ad)
        spatial_graphs.append(graph)

    combined_adata, combined_graph = integration_model._get_multiple_adata(
        adata_list=processed_data,
        data_name_list=[list(i.obs[batch_key])[0] for i in adata_list],
        graph_list=spatial_graphs
    )
    n_components = min(int(adata.shape[1]/2), 200)

    integrated_data = integration_model._data_process(
        combined_adata,
        pca_n_comps=n_components
    )

    embeddings = integration_model._fit(
        data=integrated_data,
        graph_dict=combined_graph,
        domains=combined_adata.obs["batch"].values,
        n_domains=len(adata_list))

    combined_adata.obsm["DeepST_embed"] = embeddings
    sc.pp.neighbors(combined_adata, use_rep='DeepST_embed')
    sc.tl.umap(combined_adata)

    if(cluster_method=="mclust"):
        combined_adata.obs["cluster"] = mclust(combined_adata.obsm["DeepST_embed"], n_cluster=n_cluster)
    elif(cluster_method=="leiden"):
        res = search_resolution(combined_adata, fixed_clus_count=n_cluster)
        sc.tl.leiden(combined_adata, resolution=res, key_added="cluster")

    embed = pd.DataFrame(combined_adata.obsm["DeepST_embed"])
    embed.index = list(combined_adata.obs_names)
    embed.to_csv(save_path+f"/embed_{save_key}.csv")
    umap = pd.DataFrame(combined_adata.obsm["X_umap"])
    umap.index = list(combined_adata.obs_names)
    umap.to_csv(save_path+"/umap_"+save_key+".csv")
    batch = pd.DataFrame(combined_adata.obs[batch_key])
    batch.index = list(combined_adata.obs_names)
    batch.to_csv(save_path+"/batch_"+save_key+".csv")
    cluster = pd.DataFrame(combined_adata.obs["cluster"])
    cluster.index = list(combined_adata.obs_names)
    cluster.to_csv(save_path+"/cluster_"+save_key+".csv")

    # if(os.path.exists(temp_dir)):
    #     shutil.rmtree(temp_dir)
