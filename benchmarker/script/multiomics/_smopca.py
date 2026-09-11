import sys
sys.path.append('/mnt/datadisk/lizhongzhan/SpaMultiOmics/Benchmark/external/SMOPCA/')

import os
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
from SMOPCA.model import SMOPCA
from sklearn.cluster import KMeans

from SpatialGlue.preprocess import (
    pca,
    clr_normalize_each_cell,
)
from utils import h5_to_h5ad, search_resolution


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


def preprocess_rna(adata_rna, hvg_num, batch_key):
    hvg_kwargs = get_hvg_kwargs(hvg_num, batch_key)

    sc.pp.highly_variable_genes(adata_rna, **hvg_kwargs)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)

    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]].copy()

    return adata_rna_high


def preprocess_adt(adata_adt, adata_rna):
    adata_adt = adata_adt[adata_rna.obs_names].copy()
    adata_adt = clr_normalize_each_cell(adata_adt)

    n_adt_pcs = max(2, adata_adt.n_vars - 1)
    adata_adt.obsm["feat"] = pca(adata_adt, n_comps=n_adt_pcs)

    return adata_adt


def run_smopca(
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

    print("[1/5] Loading data ...")
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

    print("[2/5] Preprocessing ...")
    if mode == "atac":
        adata_rna = preprocess_rna(
            adata_rna=adata_rna,
            hvg_num=hvg_num,
            batch_key=batch_key
        )
        adata_omics2 = preprocess_rna(
            adata_rna=adata_omics2,
            hvg_num=hvg_num*10,
            batch_key=batch_key
        )

    else:
        adata_rna = preprocess_rna(
            adata_rna=adata_rna,
            hvg_num=hvg_num,
            batch_key=batch_key
        )
        adata_omics2 = preprocess_adt(
            adata_adt=adata_omics2,
            adata_rna=adata_rna
        )

    print("RNA feat shape:", adata_rna.X.shape)
    print("Omics2 feat shape:", adata_omics2.X.shape)

    X1 = np.array(adata_rna.X)
    X2 = np.array(adata_omics2.X)
    pos = np.array(adata_rna.obsm['spatial'])
   
    print("[3/5] Training SMOPCA ...")
    smopca = SMOPCA(Y_list=[X1.T, X2.T], Z_dim=20, pos=pos, intercept=False, omics_weight=False)
    smopca.estimateParams(sigma_init_list=(1, 1), tol_sigma=2e-5, sigma_xtol_list=(1e-6, 1e-6), gamma_init=1, estimate_gamma=True)
    z = smopca.calculatePosterior()
    if smopca.use_gpu:
        z = z.get()
    y_pred = KMeans(n_clusters=n_cluster, n_init=100).fit_predict(z)

    print("[4/5] Saving embeddings ...")
    adata_out = adata_rna.copy()
    adata_out.obsm["latent"] = z.copy()
    adata_out.obs["cluster"] = y_pred.copy()

    print("[5/5] Clustering ...")
    latent = pd.DataFrame(adata_out.obsm["latent"], index=adata_out.obs_names)
    latent.to_csv(os.path.join(save_path, save_key + "_latent.csv"))

    sc.pp.neighbors(adata_out, n_neighbors=100, use_rep="latent", metric='euclidean')
    sc.tl.umap(adata_out)

    ## save UMAP
    umap = pd.DataFrame(adata_out.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata_out.obs_names)
    umap.insert(2, "cluster", adata_out.obs['cluster'].values)
    umap.to_csv(os.path.join(save_path, save_key + ".csv"))


if __name__ == "__main__":
    args = parse_args()

    run_smopca(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )