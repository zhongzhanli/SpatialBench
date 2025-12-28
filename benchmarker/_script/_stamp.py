import sys
sys.path.append("../../Benchmark/external/scTM/")
import sctm
import scanpy as sc
import pandas as pd
import squidpy as sq
import warnings
warnings.filterwarnings("ignore")

from utils import split_adata, transform_coord

args = sys.argv

file_path = args[1]
save_path = args[2]
n_cluster = int(args[3])
batch_key =  args[4]
hvg_num = int(args[5])
save_key = args[6]
cluster_method = args[7]

if __name__ == "__main__":

    sctm.seed.seed_everything(0)

    adata = sc.read_h5ad(file_path)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_num+3000, subset=True, batch_key=batch_key)

    adata_list = split_adata(adata, batch_key)
    # raw_coords = [ad.obsm["spatial"] for ad in adata_list]
    # new_coords = transform_coord(raw_coords)
    # for i in range(len(adata_list)):
    #     adata_list[i].obsm["spatial"] = new_coords[i].copy()
    # adata = sc.concat(adata_list)

    if(len(adata_list)==3 and list(adata.obs[batch_key])[0]=="E9.5"): # Mouse_embryo
        sq.gr.spatial_neighbors(adata,  n_neighs=round(1 / 1000 * adata.n_obs))
        adata.obs["time"] = list(adata.obs[batch_key].map({"E9.5":0, "E10.5":1, "E11.5":2}))
        model = sctm.stamp.STAMP(
            adata,
            n_topics=n_cluster,
            time_covariate_keys="time",
        )
        model.train(batch_size=4096, sampler="W")
    elif(len(adata_list)==3 and list(adata.obs[batch_key])[0] in ["10xVisium", "Slide-seqV2", "Stereo-seq"]): # Mouse ob
        sq.gr.spatial_neighbors(adata, library_key=batch_key)
        model = sctm.stamp.STAMP(
            adata,
            n_topics=n_cluster,
            categorical_covariate_keys=[batch_key],
            gene_likelihood="nb",
            # mode="sgc",
            dropout = 0.1,
        )
        model.train(batch_size=4096, sampler="W")
    else:
        sq.gr.spatial_neighbors(adata, library_key=batch_key)

        model = sctm.stamp.STAMP(
            adata,
            n_topics = n_cluster,
            categorical_covariate_keys=[batch_key],
            mode="sgc",
            gene_likelihood="nb")
        model.train(learning_rate = 0.01, min_epochs = 200, batch_size=4096,) 

    topic_prop = model.get_cell_by_topic()
    beta = model.get_feature_by_topic()
    for i in topic_prop.columns:
        adata.obs[i] =  topic_prop[i]
    adata.obsm["X_stamp"] = topic_prop.values
    if(adata.shape[0] > 4096):
        sc.pp.neighbors(adata, metric = "hellinger", use_rep = "X_stamp")
    else:
        sc.pp.neighbors(adata, use_rep = "X_stamp")
    sc.tl.umap(adata)
    adata.obs["cluster"] = list(topic_prop.idxmax(axis=1))

    embed = pd.DataFrame(adata.obsm["X_stamp"])
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
