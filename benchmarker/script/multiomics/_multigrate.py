import sys
sys.path.append('/mnt/datadisk/lizhongzhan/SpaMultiOmics/Benchmark/external/Multigrate/')
import os
import sys 

import torch
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import multigrate as mtg

from utils import search_resolution, h5_to_h5ad

args = sys.argv

n_cluster = int(args[1])
rna_file_path = args[2]
atac_file_path = args[3]
adt_file_path = args[4]
save_path = args[5]
save_key = args[6]
hvg_num = int(args[7])
batch_key = args[8]

def call_multigrate(rna_file_path: str,
                 atac_file_path: str,
                 adt_file_path: str,
                 save_path: str,
                 n_cluster: int,
                 hvg_num: int,
                 save_key: str,
):
    
    torch.set_num_threads(5)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if rna_file_path!="NULL":
        rna = h5_to_h5ad(rna_file_path, batch_key=batch_key)
        rna.var_names_make_unique()
    else:
        rna = None
    if atac_file_path!="NULL":
        atac = h5_to_h5ad(atac_file_path, batch_key=batch_key)
        atac.var_names_make_unique()
    else:
        atac = None
    if adt_file_path!="NULL":
        adt = h5_to_h5ad(adt_file_path, batch_key=batch_key)
        adt.var_names_make_unique()
    else:
        adt = None

    rna.layers['counts'] = rna.X.copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=hvg_num, flavor="seurat_v3", subset=True, layer="counts")
    rna_hvg = rna[:, rna.var.highly_variable].copy()

    if atac is None and adt is not None:
        adt.var_names_make_unique()
        adt.layers['counts'] = adt.X.copy()
        adt.X = adt.X.astype(float)
        mu.prot.pp.clr(adt)
        adt.layers['clr'] = adt.X.copy()
        adata = mtg.data.organize_multimodal_anndatas(
            adatas = [[rna], [adt]],            # a list of anndata objects per modality, RNA-seq always goes first
            layers = [['counts'], ['clr']],     # if need to use data from .layers, if None use .X
        )
    elif adt is None and atac is not None:
        atac.layers['counts'] = atac.X.copy()
        mu.atac.pp.tfidf(atac, scale_factor=1e4)
        atac.layers['tf-idf'] = atac.X.copy()
        atac.X = atac.layers['counts'].copy()
        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)
        sc.pp.highly_variable_genes(atac, n_top_genes=hvg_num*10, subset=True, flavor="seurat_v3", layer="counts")
        atac.layers['log-norm'] = atac.X.copy()
        atac_hvf = atac[:, atac.var.highly_variable].copy()

        adata = mtg.data.organize_multimodal_anndatas(
            adatas = [[rna_hvg], [atac_hvf]],           # a list of anndata objects per modality, RNA-seq always goes first
            layers = [['counts'], ['log-norm']],        # if need to use data from .layers, if None use .X
        )
    else:
        raise ValueError("Invalid data.")

    # Add Metadata
    
    # raise ValueError(adata)
    mtg.model.MultiVAE.setup_anndata(
        adata,
        rna_indices_end=hvg_num,          # how many features in the rna-seq modality
        categorical_covariate_keys=[batch_key] if batch_key in rna.obs else None
    )
    if atac is None and adt is not None:
        model = mtg.model.MultiVAE(
            adata,
            losses=['nb', 'mse'],
            # n_layers_encoders=[2, 2],
            # n_layers_decoders=[2, 2],
        )
    else:
        model = mtg.model.MultiVAE(
            adata,
            losses=['nb', 'mse'],
            # loss_coefs={
            #     "integ": 500,
            # },
            alignment_type="marginal",
            modality_alignment="MMD",
        )

    model.train() # default lr = 0.0005

    # Save Results
    ## save latent
    model.get_model_output()
    latent = pd.DataFrame(data=adata.obsm['X_multigrate'],
                          index=adata.obs_names)
    latent.to_csv(os.path.join(save_path, save_key + "_latent.csv"))

    # Visulize Data
    sc.pp.neighbors(adata, use_rep="X_multigrate")
    sc.tl.umap(adata)
    res = search_resolution(adata, fixed_clus_count=n_cluster)
    sc.tl.leiden(adata, resolution=res, key_added="cluster")

    ## save UMAP
    umap = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names)
    umap.insert(2, "cluster", adata.obs['cluster'].values)
    umap.to_csv(os.path.join(save_path, save_key + ".csv"))


call_multigrate(rna_file_path=rna_file_path,
                atac_file_path=atac_file_path,
                adt_file_path=adt_file_path,
                save_path=save_path,
                save_key=save_key,
                n_cluster=n_cluster,
                hvg_num=hvg_num)

