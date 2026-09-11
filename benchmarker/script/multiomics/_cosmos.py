import os
import argparse
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import sys

warnings.filterwarnings("ignore")
import sys
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/Benchmark/external/COSMOS")

from COSMOS import cosmos

random_seed = 20

from utils import h5_to_h5ad, search_resolution

def parse_args():
    parser = argparse.ArgumentParser("COSMOS unified runner")

    parser.add_argument("n_cluster", type=int)
    parser.add_argument("rna_file_path", type=str)
    parser.add_argument("atac_file_path", type=str)
    parser.add_argument("adt_file_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("save_key", type=str)
    parser.add_argument("hvg_num", type=int)
    parser.add_argument("batch_key", type=str)

    return parser.parse_args()


def is_null_string(x):
    return str(x).upper() == "NULL"


def infer_mode(atac_file_path, adt_file_path):
    atac_is_null = is_null_string(atac_file_path)
    adt_is_null = is_null_string(adt_file_path)

    if (not atac_is_null) and adt_is_null:
        return "atac"
    elif atac_is_null and (not adt_is_null):
        return "adt"
    else:
        raise ValueError(
            "Invalid input."
        )

def prepare_spatial_info(adata):
    if "spatial" not in adata.obsm:
        raise ValueError("spatial not exist in 'adata.obsm'")

    adata.obs["x_pos"] = np.array(adata.obsm["spatial"])[:, 0]
    adata.obs["y_pos"] = np.array(adata.obsm["spatial"])[:, 1]
    adata.X = adata.X.astype("float64")
    return adata


def run_cosmos(
    n_cluster,
    rna_file_path,
    atac_file_path,
    adt_file_path,
    save_path,
    save_key,
    hvg_num,
    batch_key
):
    os.makedirs(save_path, exist_ok=True)

    mode = infer_mode(atac_file_path, adt_file_path)
    print(f"[Info] inferred mode = {mode}")

    print(f"[Info] hvg_num = {hvg_num} (not used, kept for interface consistency)")
    print(f"[Info] batch_key = {batch_key} (not used, kept for interface consistency)")

    print("[1/6] Loading data ...")
    rna = h5_to_h5ad(rna_file_path)
    rna.layers["counts"] =  rna.X.copy()
    rna.X = rna.X.astype(float)
    sc.pp.normalize_per_cell(rna)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna,
                                flavor="seurat_v3",
                                n_top_genes=hvg_num,
                                subset=True,
                                layer="counts"
                                # batch_key=batch_key 
    )

    rna.var_names_make_unique()

    if mode == "atac":
        omics2 = h5_to_h5ad(atac_file_path)
        omics2.layers["counts"] =  omics2.X.copy()
        omics2.X = omics2.X.astype(float)
        sc.pp.normalize_per_cell(omics2)
        sc.pp.log1p(omics2)
        sc.pp.highly_variable_genes(omics2,
                                    flavor="seurat_v3",
                                    n_top_genes=hvg_num*10,
                                    subset=True,
                                    layer="counts"
                                    # batch_key=batch_key 
        )
    else:
        omics2 = h5_to_h5ad(adt_file_path)
        omics2.X = omics2.X.astype(float)
        sc.pp.log1p(omics2)

    omics2.var_names_make_unique()

    # 按 RNA 对齐
    omics2 = omics2[rna.obs_names].copy()

    print(f"RNA shape: {rna.shape}")
    print(f"Omics2 shape: {omics2.shape}")

    print("[2/6] Preparing spatial coordinates ...")
    rna = prepare_spatial_info(rna)
    omics2 = prepare_spatial_info(omics2)

    print("[3/6] Training COSMOS ...")
    cosmos_comb = cosmos.Cosmos(adata1=rna, adata2=omics2)
    cosmos_comb.preprocessing_data(n_neighbors=10)

    gpu_id = 0 if torch.cuda.is_available() else 0
    cosmos_comb.train(
        spatial_regularization_strength=0.01, #if mode=="atac" else 0.05,
        z_dim=50,
        lr=1e-3,
        wnn_epoch=500,
        total_epoch=1000,
        max_patience_bef=10,
        max_patience_aft=30,
        min_stop=200,
        random_seed=random_seed,
        gpu=gpu_id,
        regularization_acceleration=True,
        edge_subset_sz=1000000
    )

    embed = cosmos_comb.embedding
    df_embedding = pd.DataFrame(embed, index=rna.obs_names)
    df_embedding.to_csv(os.path.join(save_path, save_key + "_latent.csv"))

    rna.obsm["latent"] = np.array(embed)
    sc.pp.neighbors(rna, n_neighbors=50, use_rep="latent")
    sc.tl.umap(rna)
    res = search_resolution(rna, fixed_clus_count=n_cluster)
    sc.tl.leiden(rna, resolution=res, key_added="cluster")

    ## save UMAP
    umap = pd.DataFrame(rna.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=rna.obs_names)
    umap.insert(2, "cluster", rna.obs['cluster'].values)
    umap.to_csv(os.path.join(save_path, save_key + ".csv"))


if __name__ == "__main__":
    args = parse_args()

    run_cosmos(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )