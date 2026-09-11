import os
import argparse
import warnings

import torch
import scanpy as sc
import pandas as pd

from SpatialGlue.preprocess import (
    pca,
    lsi,
    clr_normalize_each_cell,
    construct_neighbor_graph
)
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
# from SpatialGlue.utils import clustering
from utils import search_resolution, h5_to_h5ad

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
        return "atac", "Spatial-ATAC-RNA"
    elif atac_is_null and (not adt_is_null):
        return "adt", "Stereo-CITE-seq"
    else:
        raise ValueError(
            "Invalid input."
        )


def get_hvg_kwargs(hvg_num, batch_key):
    kwargs = dict(flavor="seurat_v3", n_top_genes=hvg_num)
    # if not is_null_string(batch_key):
    #     kwargs["batch_key"] = batch_key
    return kwargs


def preprocess_rna_for_atac(adata_rna, hvg_num, batch_key):
    hvg_kwargs = get_hvg_kwargs(hvg_num, batch_key)

    sc.pp.highly_variable_genes(adata_rna, **hvg_kwargs)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)

    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]].copy()
    adata_rna.obsm["feat"] = pca(adata_rna_high, n_comps=50)

    return adata_rna


def preprocess_rna_for_adt(adata_rna, adata_adt, hvg_num, batch_key):
    hvg_kwargs = get_hvg_kwargs(hvg_num, batch_key)

    sc.pp.filter_genes(adata_rna, min_cells=0)
    sc.pp.filter_genes(adata_adt, min_cells=0)

    # 过滤后重新按 RNA 对齐
    adata_adt = adata_adt[adata_rna.obs_names].copy()

    sc.pp.highly_variable_genes(adata_rna, **hvg_kwargs)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)

    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]].copy()

    n_rna_pcs = max(2, adata_adt.n_vars - 1)
    adata_rna.obsm["feat"] = pca(adata_rna_high, n_comps=n_rna_pcs)

    return adata_rna, adata_adt


def preprocess_atac(adata_atac, adata_rna, hvg_num, batch_key):
    hvg_kwargs = get_hvg_kwargs(hvg_num, batch_key)

    adata_atac = adata_atac[adata_rna.obs_names].copy()
    sc.pp.highly_variable_genes(adata_atac, **hvg_kwargs)
    lsi(adata_atac, use_highly_variable=False, n_components=51)
    adata_atac.obsm["feat"] = adata_atac.obsm["X_lsi"].copy()

    return adata_atac


def preprocess_adt(adata_adt, adata_rna):
    adata_adt = adata_adt[adata_rna.obs_names].copy()
    adata_adt = clr_normalize_each_cell(adata_adt)

    n_adt_pcs = max(2, adata_adt.n_vars - 1)
    adata_adt.obsm["feat"] = pca(adata_adt, n_comps=n_adt_pcs)

    return adata_adt


def run_spatialglue(
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

    mode, data_type = infer_mode_and_datatype(atac_file_path, adt_file_path)
    print(f"[Info] inferred mode = {mode}")
    print(f"[Info] inferred data_type = {data_type}")

    print("[1/7] Loading data ...")
    adata_rna = h5_to_h5ad(rna_file_path)

    if adata_rna.shape[0] > 5000 and mode=="atac":
        data_type = 'Spatial-epigenome-transcriptome'

    adata_rna.var_names_make_unique()

    if mode == "atac":
        adata_omics2 = h5_to_h5ad(atac_file_path)
    else:
        adata_omics2 = h5_to_h5ad(adt_file_path)
    adata_omics2.var_names_make_unique()

    print(f"RNA shape: {adata_rna.shape}")
    print(f"Omics2 shape: {adata_omics2.shape}")

    print("[2/7] Preprocessing ...")
    if mode == "atac":
        adata_rna = preprocess_rna_for_atac(
            adata_rna=adata_rna,
            hvg_num=hvg_num,
            batch_key=batch_key
        )
        adata_omics2 = preprocess_atac(
            adata_atac=adata_omics2,
            adata_rna=adata_rna,
            hvg_num=hvg_num,
            batch_key=batch_key
        )
        cluster_method = "leiden"

    else:
        adata_rna, adata_omics2 = preprocess_rna_for_adt(
            adata_rna=adata_rna,
            adata_adt=adata_omics2,
            hvg_num=hvg_num,
            batch_key=batch_key
        )
        adata_omics2 = preprocess_adt(
            adata_adt=adata_omics2,
            adata_rna=adata_rna
        )
        cluster_method = "mclust"

    print("RNA feat shape:", adata_rna.obsm["feat"].shape)
    print("Omics2 feat shape:", adata_omics2.obsm["feat"].shape)

    print("[3/7] Constructing neighbor graph ...")
    data = construct_neighbor_graph(adata_rna, adata_omics2, datatype=data_type)

    print("[4/7] Training SpatialGlue ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Train_SpatialGlue(data, datatype=data_type, device=device)
    output = model.train()

    print("[5/7] Saving embeddings ...")
    adata_out = adata_rna.copy()
    adata_out.obsm[f"{save_key}_emb_latent_omics1"] = output["emb_latent_omics1"]
    adata_out.obsm[f"{save_key}_emb_latent_omics2"] = output["emb_latent_omics2"]
    adata_out.obsm["SpatialGlue"] = output["SpatialGlue"]
    adata_out.obsm[f"{save_key}_alpha"] = output["alpha"]
    adata_out.obsm[f"{save_key}_alpha_omics1"] = output["alpha_omics1"]
    adata_out.obsm[f"{save_key}_alpha_omics2"] = output["alpha_omics2"]

    print("[6/7] Clustering ...")
    latent = pd.DataFrame(adata_out.obsm["SpatialGlue"], index=adata_out.obs_names)
    latent.to_csv(os.path.join(save_path, save_key + "_latent.csv"))

    sc.pp.neighbors(adata_out, n_neighbors=50, use_rep="SpatialGlue")
    sc.tl.umap(adata_out)
    res = search_resolution(adata_out, fixed_clus_count=n_cluster)
    sc.tl.leiden(adata_out, resolution=res, key_added="cluster")

    ## save UMAP
    umap = pd.DataFrame(adata_out.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata_out.obs_names)
    umap.insert(2, "cluster", adata_out.obs['cluster'].values)
    umap.to_csv(os.path.join(save_path, save_key + ".csv"))


if __name__ == "__main__":
    args = parse_args()

    run_spatialglue(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )