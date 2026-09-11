import os
import argparse
import warnings

import torch
import scanpy as sc
import pandas as pd

from utils import search_resolution, h5_to_h5ad
from SpatialGlue.preprocess import (
    pca,
    lsi,
    clr_normalize_each_cell,
)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser("SpatialGlue unified runner")

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


def infer_mode_and_datatype(atac_file_path, adt_file_path):
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

def get_hvg_kwargs(hvg_num, batch_key):
    kwargs = dict(flavor="seurat_v3", n_top_genes=hvg_num)
    # if not is_null_string(batch_key):
    #     kwargs["batch_key"] = batch_key
    return kwargs


def preprocess_rna(adata_rna, hvg_num, batch_key):
    hvg_kwargs = get_hvg_kwargs(hvg_num, batch_key)

    sc.pp.highly_variable_genes(adata_rna, **hvg_kwargs)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    # sc.pp.scale(adata_rna)

    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]].copy()
    adata_rna.obsm["feat"] = pca(adata_rna_high, n_comps=50)

    return adata_rna


def preprocess_atac(adata_atac, hvg_num, batch_key):
    hvg_kwargs = get_hvg_kwargs(hvg_num, batch_key)
    sc.pp.highly_variable_genes(adata_atac, **hvg_kwargs)
    lsi(adata_atac, use_highly_variable=False, n_components=51)
    adata_atac.obsm["feat"] = adata_atac.obsm["X_lsi"].copy()

    return adata_atac


def preprocess_adt(adata_adt):
    adata_adt = clr_normalize_each_cell(adata_adt)

    n_adt_pcs = max(2, adata_adt.n_vars - 1)
    adata_adt.obsm["feat"] = pca(adata_adt, n_comps=n_adt_pcs)

    return adata_adt


def run_single_modal(
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

    mode = infer_mode_and_datatype(atac_file_path, adt_file_path)
    adata_rna = h5_to_h5ad(rna_file_path)
    adata_rna.var_names_make_unique()

    if mode == "atac":
        adata_omics2 = h5_to_h5ad(atac_file_path)
    else:
        adata_omics2 = h5_to_h5ad(adt_file_path)

    adata_omics2.var_names_make_unique()

    print(f"RNA shape: {adata_rna.shape}")
    print(f"Omics2 shape: {adata_omics2.shape}")
    
    adata_rna = preprocess_rna(
            adata_rna=adata_rna,
            hvg_num=hvg_num,
            batch_key=batch_key
    )
    print("[2/7] Preprocessing ...")
    if mode == "atac":
        adata_omics2 = preprocess_atac(
            adata_atac=adata_omics2,
            hvg_num=hvg_num,
            batch_key=batch_key
        )

    else:
        adata_omics2 = preprocess_adt(
            adata_adt=adata_omics2,
        )

    print("RNA feat shape:", adata_rna.obsm["feat"].shape)
    print("Omics2 feat shape:", adata_omics2.obsm["feat"].shape)

    sc.pp.neighbors(adata_rna, use_rep="feat")
    sc.tl.umap(adata_rna)
    sc.pp.neighbors(adata_omics2, use_rep="feat")
    sc.tl.umap(adata_omics2)

    latent = pd.DataFrame(adata_rna.obsm["feat"], index=adata_rna.obs_names)
    latent.to_csv(os.path.join(save_path, "rna_latent.csv"))
    latent = pd.DataFrame(adata_omics2.obsm["feat"], index=adata_omics2.obs_names)
    latent.to_csv(os.path.join(save_path, f"{mode}_latent.csv"))

    res = search_resolution(adata_rna, fixed_clus_count=n_cluster)
    sc.tl.leiden(adata_rna, resolution=res, key_added="cluster")
    umap = pd.DataFrame(adata_rna.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata_rna.obs_names)
    umap.insert(2, "cluster", adata_rna.obs['cluster'].values)
    umap.to_csv(os.path.join(save_path, "rna.csv"))

    res = search_resolution(adata_omics2, fixed_clus_count=n_cluster)
    sc.tl.leiden(adata_omics2, resolution=res, key_added="cluster")
    umap = pd.DataFrame(adata_omics2.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata_omics2.obs_names)
    umap.insert(2, "cluster", adata_omics2.obs['cluster'].values)
    umap.to_csv(os.path.join(save_path, f"{mode}.csv"))


if __name__ == "__main__":
    args = parse_args()

    run_single_modal(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )