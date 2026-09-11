import sys
# 按你的项目结构保留
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/")

import os
import argparse
import random
import warnings
from itertools import chain

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scglue

from utils import h5_to_h5ad, search_resolution

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser("GLUE unified runner")

    parser.add_argument("n_cluster", type=int)
    parser.add_argument("rna_file_path", type=str)
    parser.add_argument("atac_file_path", type=str)
    parser.add_argument("adt_file_path", type=str)   # 这里保留统一接口；对 GLUE 暂不使用
    parser.add_argument("save_path", type=str)
    parser.add_argument("save_key", type=str)
    parser.add_argument("hvg_num", type=int)
    parser.add_argument("batch_key", type=str)

    return parser.parse_args()


def is_null_string(x):
    return str(x).upper() == "NULL"


def seed_everything(seed=111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_peak_bed_like_names(atac):
    """
    将 peak 名解析为 chrom / chromStart / chromEnd
    期望格式类似:
        chr1:1000-2000
        chr1-1000-2000
    """
    peak_names = pd.Index(atac.var_names.astype(str))

    chrom = []
    chrom_start = []
    chrom_end = []

    for name in peak_names:
        name_std = name.replace(":", "-")
        parts = name_std.split("-")
        if len(parts) < 3:
            chrom.append(np.nan)
            chrom_start.append(np.nan)
            chrom_end.append(np.nan)
            continue

        chrom.append(parts[0])
        try:
            chrom_start.append(int(parts[1]))
            chrom_end.append(int(parts[2]))
        except Exception:
            chrom_start.append(np.nan)
            chrom_end.append(np.nan)

    atac.var["chrom"] = chrom
    atac.var["chromStart"] = chrom_start
    atac.var["chromEnd"] = chrom_end

    valid = (
        atac.var["chrom"].notna()
        & atac.var["chromStart"].notna()
        & atac.var["chromEnd"].notna()
    )
    atac = atac[:, valid].copy()

    return atac


def preprocess_rna(rna, hvg_num, batch_key):
    rna.layers["counts"] = rna.X.copy()

    # 如需 batch_key，可自行打开
    hvg_kwargs = dict(
        n_top_genes=hvg_num,
        flavor="seurat_v3",
        layer="counts"
    )
    # if not is_null_string(batch_key):
    #     hvg_kwargs["batch_key"] = batch_key

    scglue.data.get_gene_annotation(
        rna,
        gtf="/mnt/datadisk/lizhongzhan/SpaMultiOmics/DATA/Mouse_embryo/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
        gtf_by="gene_name"
    )

    # 去掉无法注释到染色体的基因
    rna = rna[:, ~rna.var["chrom"].isna()].copy()

    sc.pp.highly_variable_genes(rna, **hvg_kwargs)
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=100, svd_solver="auto")

    return rna


def preprocess_atac(atac, hvg_num):
    split = atac.var_names.str.split(r"[:-]")
    atac.var["chrom"] = split.map(lambda x: x[0])
    atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
    atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)

    # # 给 ATAC 标高变 feature，便于 guidance_hvf 和 configure_dataset 使用
    # # 这里采用方差筛选，保持和 GLUE 原代码意图一致
    # if atac.n_vars <= hvg_num:
    #     atac.var["highly_variable"] = True
    # else:
    #     x = atac.X
    #     if hasattr(x, "toarray"):
    #         mean = np.asarray(x.mean(axis=0)).ravel()
    #         sq_mean = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    #         var = sq_mean - mean ** 2
    #     else:
    #         var = np.var(np.asarray(x), axis=0)

    #     hv_idx = np.argsort(var)[::-1][:hvg_num]
    #     highly_variable = np.zeros(atac.n_vars, dtype=bool)
    #     highly_variable[hv_idx] = True
    #     atac.var["highly_variable"] = highly_variable

    # scglue.data.lsi(atac, n_components=100, n_iter=15)

    return atac


def run_glue(
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
    seed_everything(111)

    print("[1/6] Loading data ...")
    rna = h5_to_h5ad(rna_file_path)
    atac = h5_to_h5ad(atac_file_path)

    rna.var_names_make_unique()
    rna.obs_names_make_unique()
    atac.var_names_make_unique()
    atac.obs_names_make_unique()

    # # 统一按 RNA 对齐
    # common_obs = rna.obs_names.intersection(atac.obs_names)
    # if len(common_obs) == 0:
    #     raise ValueError("RNA and ATAC have no overlapping obs_names!")

    # rna = rna[common_obs].copy()
    # atac = atac[common_obs].copy()
    # atac = atac[rna.obs_names].copy() if False else atac  # 占位，避免误解
    # atac = atac[rna.obs_names, :].copy()

    print(f"RNA shape: {rna.shape}")
    print(f"ATAC shape: {atac.shape}")

    if not is_null_string(adt_file_path):
        print("[Info] gene activity file is provided but not used in GLUE.")

    print("[2/6] Preprocessing ...")
    rna = preprocess_rna(rna, hvg_num=hvg_num, batch_key=batch_key)
    atac = preprocess_atac(atac, hvg_num=hvg_num)

    print("[3/6] Building guidance graph ...")
    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)

    scglue.data.lsi(atac, n_components=100, n_iter=15)

    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index
    )).copy()

    print("[4/6] Configuring and training GLUE ...")
    scglue.models.configure_dataset(
        rna,
        "NB",
        use_highly_variable=True,
        use_layer="counts",
        use_rep="X_pca",
    )
    scglue.models.configure_dataset(
        atac,
        "NB",
        use_highly_variable=True,
        use_rep="X_lsi",
    )

    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac},
        guidance_hvf,
        model=scglue.models.SCGLUEModel,
        fit_kws={"directory": os.path.join(save_path, f"{save_key}_glue_train")}
    )

    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)

    print("[5/6] Saving embeddings ...")
    # 统一以 RNA 对应的 embedding 作为输出结果
    adata_out = sc.concat([rna, atac])
    # adata_out = rna.copy()
    # adata_out.obsm["GLUE"] = rna.obsm["X_glue"].copy()
    # adata_out.obsm[f"{save_key}_emb_latent_omics1"] = rna.obsm["X_glue"].copy()
    # adata_out.obsm[f"{save_key}_emb_latent_omics2"] = atac.obsm["X_glue"].copy()

    latent = pd.DataFrame(
        adata_out.obsm["X_glue"],
        index=adata_out.obs_names
    )
    latent.to_csv(os.path.join(save_path, save_key + "_latent.csv"))

    print("[6/6] Clustering ...")
    sc.pp.neighbors(adata_out, n_neighbors=30, use_rep="X_glue")
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

    run_glue(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )