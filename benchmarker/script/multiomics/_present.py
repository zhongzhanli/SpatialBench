import sys
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/Benchmark/external/PRESENT")
import os
import argparse
import warnings
import random

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from PRESENT import PRESENT_function

warnings.filterwarnings("ignore")

seed = 100

from utils import h5_to_h5ad, search_resolution


def parse_args():
    parser = argparse.ArgumentParser("PRESENT unified runner")

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
        return "protein"
    else:
        raise ValueError(
            "Invalid input."
        )


def set_seed(seed=100):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_present(
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
    print(f"[Info] hvg_num = {hvg_num}")
    print(f"[Info] batch_key = {batch_key} (not used, kept for interface consistency)")

    set_seed(seed)
    device = get_device()
    print(f"[Info] device = {device}")

    print("[1/6] Loading data ...")
    adata_rna = h5_to_h5ad(rna_file_path, batch_key=batch_key)
    adata_rna.var_names_make_unique()

    if mode == "atac":
        adata_omics2 = h5_to_h5ad(atac_file_path, batch_key=batch_key)
    else:
        adata_omics2 = h5_to_h5ad(adt_file_path, batch_key=batch_key)
    adata_omics2.var_names_make_unique()

    adata_omics2 = adata_omics2[adata_rna.obs_names].copy()

    print(f"RNA shape: {adata_rna.shape}")
    print(f"Omics2 shape: {adata_omics2.shape}")

    if "spatial" not in adata_rna.obsm:
        raise ValueError("spatial not exist in rna")
    if "spatial" not in adata_omics2.obsm:
        raise ValueError("spatial not exist in omics2")

    print("[2/6] Running PRESENT ...")
    if mode == "protein":
        adata = PRESENT_function(
            spatial_key="spatial",
            adata_rna=adata_rna,
            batch_key=batch_key if batch_key in adata_rna.obs else None,
            gene_min_cells=1,
            num_hvg=hvg_num,
            adata_adt=adata_omics2,
            protein_min_cells=1,
            nclusters=n_cluster,
            device=device
        )
    else:
        adata = PRESENT_function(
            spatial_key="spatial",
            adata_rna=adata_rna,
            batch_key=batch_key if batch_key in adata_rna.obs else None,
            gene_min_cells=1,
            num_hvg=hvg_num,
            adata_atac=adata_omics2,
            peak_min_cells_fraction=0.03,
            nclusters=n_cluster,
            device=device
        )
    adata = adata[adata_rna.obs_names,]
    print(adata)

    print("[3/6] Building graph and UMAP ...")
    sc.pp.neighbors(adata, use_rep="embeddings")
    df_embedding = pd.DataFrame(adata.obsm["embeddings"], index=adata.obs_names)
    df_embedding.to_csv(os.path.join(save_path, save_key + "_latent.csv"))
    sc.tl.umap(adata)

    print("[4/6] Standardizing output fields ...")
    # if "LeidenClusters" in adata.obs.columns:
    #     adata.obs[save_key] = adata.obs["LeidenClusters"].astype("category")
    # else:
    ## recluster
    res = search_resolution(adata, fixed_clus_count=n_cluster)
    sc.tl.leiden(adata, resolution=res, key_added="cluster")

    umap = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names)
    umap.insert(2, "cluster", adata.obs['cluster'].values)
    umap.to_csv(os.path.join(save_path, save_key + ".csv"))


if __name__ == "__main__":
    args = parse_args()

    run_present(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )