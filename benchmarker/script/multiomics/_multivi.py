import os
import sys 
import scvi
import torch
import muon as mu
import anndata
import numpy as np
import pandas as pd
import scanpy as sc

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

def call_multivi(rna_file_path: str,
                 atac_file_path: str,
                 adt_file_path: str,
                 save_path: str,
                 n_cluster: int,
                 hvg_num: int,
                 save_key: str,
                 batch_key: str
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

    sc.pp.filter_genes(rna, min_cells=int(rna.shape[0] * 0.001))
    sc.pp.highly_variable_genes(rna,
                                flavor="seurat_v3",
                                n_top_genes=hvg_num,
                                subset=True,
                                # batch_key=batch_key 
    )
    
    if atac is None and adt is not None:
        mdata = mu.MuData({'rna': rna, 'protein': adt})
        mdata.obs = rna.obs.copy()
        scvi.model.MULTIVI.setup_mudata(
            mdata,
            modalities={
                "rna_layer": "rna",
                "protein_layer": "protein",
            },
            batch_key = batch_key if batch_key in rna.obs else None
        )
    elif adt is None and atac is not None:
        sc.pp.filter_genes(atac, min_cells=int(rna.shape[0] * 0.001))
        sc.pp.highly_variable_genes(atac,
                                    flavor="seurat_v3",
                                    n_top_genes=hvg_num*10,
                                    subset=True,
                                    # batch_key=batch_key 
        )
        mdata = mu.MuData({'rna': rna, 'atac': atac})
        mdata.obs = rna.obs.copy()
        # raise ValueError(batch_key in rna.obs)
        scvi.model.MULTIVI.setup_mudata(
            mdata,
            modalities={
                "rna_layer": "rna",
                "atac_layer": "atac",
            },
            batch_key = batch_key if batch_key in rna.obs else None
        )
    else:
        raise ValueError("Invalid data.")

    # Add Metadata
    
    
    mvi = scvi.model.MULTIVI(
        mdata,
    )
    # mvi.view_anndata_setup()
    mvi.train()

    # Save Results
    ## save latent
    latent = mvi.get_latent_representation() # 20 dimension
    latent = pd.DataFrame(latent, index=mdata.obs_names)
    latent.to_csv(os.path.join(save_path, save_key + "_latent.csv"))

    # Visulize Data
    mdata.obsm["X_multi_vi"] = latent
    sc.pp.neighbors(mdata, use_rep="X_multi_vi")
    sc.tl.umap(mdata)
    res = search_resolution(mdata, fixed_clus_count=n_cluster)
    sc.tl.leiden(mdata, resolution=res, key_added="cluster")

    ## save UMAP
    umap = pd.DataFrame(mdata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=mdata.obs_names)
    umap.insert(2, "cluster", mdata.obs['cluster'].values)
    umap.to_csv(os.path.join(save_path, save_key + ".csv"))


call_multivi(rna_file_path=rna_file_path,
             atac_file_path=atac_file_path,
             adt_file_path=adt_file_path,
             save_path=save_path,
             save_key=save_key,
             n_cluster=n_cluster,
             hvg_num=hvg_num,
             batch_key=batch_key)

