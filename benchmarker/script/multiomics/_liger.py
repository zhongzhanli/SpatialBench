import os
import argparse
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import pyliger

from utils import h5_to_h5ad, search_resolution

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser("LIGER unified runner")

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


def run_liger(
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
            "LIGER runner requires gene activity input in adt_file_path under the unified interface."
        )

    os.makedirs(save_path, exist_ok=True)

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
    print("[Info] atac_file_path is kept for unified interface but is not directly used in this LIGER runner.")

    print("[2/5] Preparing LIGER inputs ...")
    # pyliger 需要不同数据集的 cell 名不要冲突
    rna_liger = rna.copy()
    ga_liger = gene_activity.copy()

    original_rna_obs = rna.obs_names.copy()
    original_ga_obs = gene_activity.obs_names.copy()

    rna_liger.obs_names = pd.Index(["RNA#" + x for x in rna_liger.obs_names.astype(str)])
    ga_liger.obs_names = pd.Index(["GS#" + x for x in ga_liger.obs_names.astype(str)])

    rna_liger.obs.index.name = "cell"
    rna_liger.var.index.name = "gene"
    ga_liger.obs.index.name = "cell"
    ga_liger.var.index.name = "gene"

    rna_liger.uns["sample_name"] = "RNA"
    ga_liger.uns["sample_name"] = "GeneActivity"

    rna_liger.var_names_make_unique()
    ga_liger.var_names_make_unique()

    adata_list = [rna_liger, ga_liger]
    liger_obj = pyliger.create_liger(adata_list)

    print("[3/5] Running LIGER ...")
    pyliger.normalize(liger_obj)
    pyliger.select_genes(liger_obj)
    pyliger.scale_not_center(liger_obj)
    pyliger.optimize_ALS(liger_obj, k=20)
    pyliger.quantile_norm(liger_obj)

    # 原脚本跑了 UMAP，但统一 benchmark 中后处理要自己统一跑
    # 所以这里不依赖 pyliger.run_umap 的结果

    print("[4/5] Saving embeddings ...")
    rna.obsm["embed"] = np.asarray(liger_obj.adata_list[0].obsm["H_norm"]).copy()
    gene_activity.obsm["embed"] = np.asarray(liger_obj.adata_list[1].obsm["H_norm"]).copy()

    adata_out = sc.concat([rna, gene_activity])
    # adata_out = rna.copy()
    # adata_out.obsm["GLUE"] = rna.obsm["X_glue"].copy()
    # adata_out.obsm[f"{save_key}_emb_latent_omics1"] = rna.obsm["X_glue"].copy()
    # adata_out.obsm[f"{save_key}_emb_latent_omics2"] = atac.obsm["X_glue"].copy()

    latent = pd.DataFrame(
        adata_out.obsm["embed"],
        index=adata_out.obs_names
    )
    latent.to_csv(os.path.join(save_path, save_key + "_latent.csv"))

    print("[6/6] Clustering ...")
    sc.pp.neighbors(adata_out, n_neighbors=30, use_rep="embed")
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

    run_liger(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )