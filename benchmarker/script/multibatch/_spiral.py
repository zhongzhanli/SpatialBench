import sys
sys.path.append('/mnt/datadisk/lizhongzhan/SpaMultiOmics/Benchmark/external/SPIRAL/')
from spiral.main import SPIRAL_integration
from spiral.layers import *
from spiral.utils import *
from spiral.CoordAlignment import CoordAlignment
import os
import scipy.sparse as sp
import numpy as np
import argparse
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import shutil
import warnings
warnings.filterwarnings('ignore')

from utils import search_resolution, split_adata, Cal_Spatial_Net, mclust

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
    # raise ValueError("Test SPIRAL")
    feat_file=[]
    edge_file=[]
    meta_file=[]
    coord_file=[]
    flags=''
    Batch_list = []

    adata = sc.read_h5ad(file_path)
    adata.layers["counts"] = adata.X.copy()
    adata_list = split_adata(adata, batch_key)

    for i in adata_list:
        sc.pp.normalize_total(i)
        sc.pp.log1p(i)
        sc.pp.highly_variable_genes(i, flavor="seurat_v3", n_top_genes=hvg_num, layer="counts")
        i = i[:, i.var['highly_variable']]
        Batch_list.append(i)

    adata_concat = ad.concat(Batch_list, label=batch_key)


    if(not os.path.exists(f"{save_path}/bm_temp_spiral")):
        os.mkdir(f"{save_path}/bm_temp_spiral")
    temp_dir = f"{save_path}/bm_temp_spiral"

    vf = np.array(adata_concat.var.index)
    for i in range(len(adata_list)):
        t_ad = adata_concat[adata_concat.obs[batch_key]==str(i)]
        cells = list(t_ad.obs_names)
        mat = t_ad.X.todense() if sp.issparse(t_ad.X) else t_ad.X
        # raise ValueError(f"{t_ad.X.todense()}")
        mat = pd.DataFrame(mat, index=cells, columns=vf)
        mat.to_csv(temp_dir+"/gtt_input_"+str(i)+"_mat.csv")
        feat_file.append(temp_dir+"/gtt_input_"+str(i)+"_mat.csv")

        Cal_Spatial_Net(t_ad, k_cutoff=knn_cutoff , model="KNN")
        G_df = t_ad.uns["Spatial_Net"].copy()
        np.savetxt(temp_dir+"/gtt_input_"+str(i)+"_edge_KNN_"+str(knn_cutoff)+".csv",G_df.values[:,:2],fmt='%s')
        edge_file.append(temp_dir+"/gtt_input_"+str(i)+"_edge_KNN_"+str(knn_cutoff)+".csv")

        meta = t_ad.obs
        meta.to_csv(temp_dir+"/gtt_input_"+str(i)+"_meta.csv")
        meta_file.append(temp_dir+"/gtt_input_"+str(i)+"_meta.csv")

        coord=pd.DataFrame(t_ad.obsm['spatial'],index=cells,columns=['x','y'])
        coord.to_csv(temp_dir+"/gtt_input_"+str(i)+"_coord.csv")
        coord_file.append(temp_dir+"/gtt_input_"+str(i)+"_coord.csv")

    N=pd.read_csv(feat_file[0],header=0,index_col=0).shape[1]
    if (len(adata_list)==2):
        M=1
    else:
        M=len(adata_list)

    SEP=','
    net_cate='_KNN_'
    knn=knn_cutoff
    N_WALKS=knn
    WALK_LEN=1
    N_WALK_LEN=knn
    NUM_NEG=knn
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='The seed of initialization.')
    parser.add_argument('--AEdims', type=list, default=[N,[512],32], help='Dim of encoder.')
    parser.add_argument('--AEdimsR', type=list, default=[32,[512],N], help='Dim of decoder.')
    parser.add_argument('--GSdims', type=list, default=[512,32], help='Dim of GraphSAGE.')
    parser.add_argument('--zdim', type=int, default=32, help='Dim of embedding.')
    parser.add_argument('--znoise_dim', type=int, default=4, help='Dim of noise embedding.')
    parser.add_argument('--CLdims', type=list, default=[4,[],M], help='Dim of classifier.')
    parser.add_argument('--DIdims', type=list, default=[28,[32,16],M], help='Dim of discriminator.')
    parser.add_argument('--beta', type=float, default=1.0, help='weight of GraphSAGE.')
    parser.add_argument('--agg_class', type=str, default=MeanAggregator, help='Function of aggregator.')
    parser.add_argument('--num_samples', type=str, default=knn_cutoff, help='number of neighbors to sample.')
    parser.add_argument('--N_WALKS', type=int, default=N_WALKS, help='number of walks of random work for postive pairs.')
    parser.add_argument('--WALK_LEN', type=int, default=WALK_LEN, help='walk length of random work for postive pairs.')
    parser.add_argument('--N_WALK_LEN', type=int, default=N_WALK_LEN, help='number of walks of random work for negative pairs.')
    parser.add_argument('--NUM_NEG', type=int, default=NUM_NEG, help='number of negative pairs.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=256, help='Size of batches to train.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--alpha1', type=float, default=N, help='Weight of decoder loss.')
    parser.add_argument('--alpha2', type=float, default=1, help='Weight of GraphSAGE loss.')
    parser.add_argument('--alpha3', type=float, default=1, help='Weight of classifier loss.')
    parser.add_argument('--alpha4', type=float, default=1, help='Weight of discriminator loss.')
    parser.add_argument('--lamda', type=float, default=1, help='Weight of GRL.')
    parser.add_argument('--Q', type=float, default=10, help='Weight negative loss for sage losss.')
    params, unknown=parser.parse_known_args()

    if(adata.shape[0] < 256):
        params.batch_size = int(adata.shape[0]/2)

    SPII=SPIRAL_integration(params,feat_file,edge_file,meta_file)
    SPII.train()
    if not os.path.exists(temp_dir+"model/"):
        os.makedirs(temp_dir+"/model/")
    model_file=temp_dir+"/model/SPIRAL"+flags+"_model_"+str(SPII.params.batch_size)+".pt"
    torch.save(SPII.model.state_dict(),model_file)

    SPII.model.eval()
    all_idx=np.arange(SPII.feat.shape[0])
    all_layer,all_mapping=layer_map(all_idx.tolist(),SPII.adj,len(SPII.params.GSdims))
    all_rows=SPII.adj.tolil().rows[all_layer[0]]
    all_feature=torch.Tensor(SPII.feat.iloc[all_layer[0],:].values).float().cuda()
    all_embed,ae_out,clas_out,disc_out=SPII.model(all_feature,all_layer,all_mapping,all_rows,SPII.params.lamda,SPII.de_act,SPII.cl_act)
    [ae_embed,gs_embed,embed]=all_embed
    [x_bar,x]=ae_out
    embed=embed.cpu().detach()
    names=['GTT_'+str(i) for i in range(embed.shape[1])]
    embed1=pd.DataFrame(np.array(embed),index=SPII.feat.index,columns=names)
    if not os.path.exists(temp_dir+"/gtt_output/"):
        os.makedirs(temp_dir+"/gtt_output/")

    embed_file=temp_dir+"/gtt_output/SPIRAL"+flags+"_embed_"+str(SPII.params.batch_size)+".csv"
    embed1.to_csv(embed_file)

    ann=ad.AnnData(SPII.feat)
    ann.obsm['spiral']=embed1.iloc[:,SPII.params.znoise_dim:].values
    sc.pp.neighbors(ann,use_rep='spiral')
    ann.obs[batch_key]=SPII.meta.loc[:,batch_key].values
    ann.obs[batch_key] = [str(i) for i in list(ann.obs[batch_key])]
    ub=np.unique(ann.obs['batch'])
    sc.tl.umap(ann)
    coord=pd.read_csv(coord_file[0],header=0,index_col=0)
    for i in np.arange(1,len(adata_list)):
        coord=pd.concat((coord,pd.read_csv(coord_file[i],header=0,index_col=0)))
    coord.columns=['y','x']
    ann.obsm['spatial']=coord.loc[ann.obs_names,:].values

    if(cluster_method=="mclust"):
        ann.obs["cluster"] = mclust(ann.obsm["spiral"], n_cluster=n_cluster)
    elif(cluster_method=="leiden"):
        res = search_resolution(ann, fixed_clus_count=n_cluster)
        sc.tl.leiden(ann, resolution=res, key_added="cluster")

    embed = pd.DataFrame(ann.obsm["spiral"])
    embed.index = list(ann.obs_names)
    embed.to_csv(save_path+f"/embed_{save_key}.csv")
    umap = pd.DataFrame(ann.obsm["X_umap"])
    umap.index = list(ann.obs_names)
    umap.to_csv(save_path+"/umap_"+save_key+".csv")
    batch = pd.DataFrame(ann.obs[batch_key])
    batch.index = list(ann.obs_names)
    batch.to_csv(save_path+"/batch_"+save_key+".csv")
    cluster = pd.DataFrame(ann.obs["cluster"])
    cluster.index = list(ann.obs_names)
    cluster.to_csv(save_path+"/cluster_"+save_key+".csv")

    if(os.path.exists(temp_dir)):
        shutil.rmtree(temp_dir)