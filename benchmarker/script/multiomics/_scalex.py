import os
import argparse
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import scalex
from scalex import SCALEX

from utils import h5_to_h5ad, search_resolution

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser("SCALEX unified runner")

    parser.add_argument("n_cluster", type=int)
    parser.add_argument("rna_file_path", type=str)
    parser.add_argument("atac_file_path", type=str)   # 统一接口保留；本方法当前不直接使用
    parser.add_argument("adt_file_path", type=str)    # 这里表示 gene activity 文件
    parser.add_argument("save_path", type=str)
    parser.add_argument("save_key", type=str)
    parser.add_argument("hvg_num", type=int)
    parser.add_argument("batch_key", type=str)

    return parser.parse_args()


def is_null_string(x):
    return str(x).upper() == "NULL"


def ensure_dense(adata):
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    adata.X = np.asarray(adata.X)
    return adata


def run_scalex(
    n_cluster,
    rna_file_path,
    atac_file_path,
    adt_file_path,
    save_path,
    save_key,
    hvg_num,
    batch_key
):
    if is_null_string(adt_file_path):
        raise ValueError(
            "SCALEX runner requires gene activity input in adt_file_path under the unified interface."
        )

    os.makedirs(save_path, exist_ok=True)

    # 为每次运行创建独立工作目录，避免不同任务互相覆盖
    work_dir = os.path.join(save_path, f"{save_key}_scalex_workdir")
    os.makedirs(work_dir, exist_ok=True)

    print("[1/5] Loading data ...")
    rna = h5_to_h5ad(rna_file_path)
    gene_activity = h5_to_h5ad(adt_file_path)

    rna.var_names_make_unique()
    rna.obs_names_make_unique()
    gene_activity.var_names_make_unique()
    gene_activity.obs_names_make_unique()

    # common_obs = rna.obs_names.intersection(gene_activity.obs_names)
    # if len(common_obs) == 0:
    #     raise ValueError("RNA and gene activity have no overlapping obs_names!")

    # rna = rna[common_obs].copy()
    # gene_activity = gene_activity[common_obs].copy()
    # gene_activity = gene_activity[rna.obs_names, :].copy()

    rna = ensure_dense(rna)
    gene_activity = ensure_dense(gene_activity)

    print(f"RNA shape: {rna.shape}")
    print(f"Gene activity shape: {gene_activity.shape}")

    print("[Info] atac_file_path is kept for unified interface but is not directly used in this SCALEX runner.")

    print("[2/5] Preparing input ...")
    rna_input = rna.copy()
    ga_input = gene_activity.copy()

    rna_input.obs["batch"] = "rna"
    ga_input.obs["batch"] = "atac"

    adata_input = sc.concat([rna_input, ga_input], join="inner")
    temp_h5ad = os.path.join(work_dir, "temp_adata.h5ad")
    adata_input.write(temp_h5ad)

    print("[3/5] Running SCALEX ...")
    use_gpu = 1 if sc.settings.n_jobs != 0 else 0
    try:
        adata = SCALEX(
            data_list=[temp_h5ad],
            min_features=0,
            min_cells=0,
            outdir=work_dir,
            show=False,
            n_top_features=hvg_num,
            gpu=1
        )
    except Exception:
        adata = SCALEX(
            data_list=[temp_h5ad],
            min_features=0,
            min_cells=0,
            outdir=work_dir,
            show=False,
            n_top_features=hvg_num,
            gpu=0
        )

    print("[4/5] Saving embeddings ...")
    adata_out = adata.copy()
    # adata_out = rna.copy()
    # adata_out.obsm["GLUE"] = rna.obsm["X_glue"].copy()
    # adata_out.obsm[f"{save_key}_emb_latent_omics1"] = rna.obsm["X_glue"].copy()
    # adata_out.obsm[f"{save_key}_emb_latent_omics2"] = atac.obsm["X_glue"].copy()
    latent = pd.DataFrame(
        adata_out.obsm["latent"],
        index=adata_out.obs_names
    )
    latent.to_csv(os.path.join(save_path, save_key + "_latent.csv"))

    print("[6/6] Clustering ...")
    sc.pp.neighbors(adata_out, n_neighbors=30, use_rep="latent")
    sc.tl.umap(adata_out)
    res = search_resolution(adata_out, fixed_clus_count=n_cluster)
    sc.tl.leiden(adata_out, resolution=res, key_added="cluster")

    umap = pd.DataFrame(
        adata_out.obsm["X_umap"],
        columns=["UMAP1", "UMAP2"],
        index=adata_out.obs_names
    )
    umap.insert(2, "cluster", adata_out.obs["cluster"].values)
    umap.to_csv(os.path.join(save_path, save_key + ".csv"))


if __name__ == "__main__":
    args = parse_args()

    run_scalex(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )