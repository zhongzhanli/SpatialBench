import os
import sys
sys.path.append('/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/')
import argparse
import warnings
from itertools import chain

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scglue



from monae.src.config import configure_dataset
from monae.src.train import covel_train
from utils import h5_to_h5ad, search_resolution

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser("Monae unified runner")

    parser.add_argument("n_cluster", type=int)
    parser.add_argument("rna_file_path", type=str)
    parser.add_argument("atac_file_path", type=str)
    parser.add_argument("adt_file_path", type=str)   # 统一接口保留；此方法当前不使用
    parser.add_argument("save_path", type=str)
    parser.add_argument("save_key", type=str)
    parser.add_argument("hvg_num", type=int)
    parser.add_argument("batch_key", type=str)

    return parser.parse_args()


def is_null_string(x):
    return str(x).upper() == "NULL"


def parse_peak_bed_like_names(atac):
    """
    将 peak 名解析为 chrom / chromStart / chromEnd
    支持格式:
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


def ensure_dense(adata):
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    return adata


def preprocess_rna(rna, hvg_num, batch_key):
    rna = ensure_dense(rna)
    rna.layers["counts"] = rna.X.copy()

    scglue.data.get_gene_annotation(
        rna,
        gtf="/mnt/datadisk/lizhongzhan/SpaMultiOmics/DATA/Mouse_embryo/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
        gtf_by="gene_name"
    )

    rna = rna[:, ~rna.var["chrom"].isna()].copy()

    hvg_kwargs = dict(
        n_top_genes=hvg_num,
        flavor="seurat_v3",
        layer="counts",
        subset=True,
    )
    # if not is_null_string(batch_key):
    #     hvg_kwargs["batch_key"] = batch_key

    sc.pp.highly_variable_genes(rna, **hvg_kwargs)

    # Monae 原始脚本直接用原始矩阵作为 X_pre
    rna.obsm["X_pre"] = np.asarray(rna.X).copy()

    return rna


def preprocess_atac(atac, hvg_num):
    atac = ensure_dense(atac)
    atac = parse_peak_bed_like_names(atac)

    # # 给 ATAC 补 highly_variable，保证后续 graph 子图与 configure_dataset 可用
    # if atac.n_vars <= hvg_num:
    #     atac.var["highly_variable"] = True
    # else:
    #     var = np.var(np.asarray(atac.X), axis=0)
    #     hv_idx = np.argsort(var)[::-1][:hvg_num]
    #     highly_variable = np.zeros(atac.n_vars, dtype=bool)
    #     highly_variable[hv_idx] = True
    #     atac.var["highly_variable"] = highly_variable

    atac.obsm["X_pre"] = np.asarray(atac.X).copy()

    return atac


def run_monae(
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

    print("[1/6] Loading data ...")
    rna = h5_to_h5ad(rna_file_path)
    atac = h5_to_h5ad(atac_file_path)

    rna.var_names_make_unique()
    rna.obs_names_make_unique()
    atac.var_names_make_unique()
    atac.obs_names_make_unique()

    # common_obs = rna.obs_names.intersection(atac.obs_names)
    # if len(common_obs) == 0:
    #     raise ValueError("RNA and ATAC have no overlapping obs_names!")

    # rna = rna[common_obs].copy()
    # atac = atac[common_obs].copy()
    # atac = atac[rna.obs_names, :].copy()

    print(f"RNA shape: {rna.shape}")
    print(f"ATAC shape: {atac.shape}")

    if not is_null_string(adt_file_path):
        print("[Info] gene activity file is provided but not used in Monae.")

    print("[2/6] Preprocessing ...")
    rna = preprocess_rna(rna, hvg_num=hvg_num, batch_key=batch_key)
    atac = preprocess_atac(atac, hvg_num=hvg_num)

    print("[3/6] Building guidance graph ...")
    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    atac = atac[:, atac.var["highly_variable"]]
    graph = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index
    )).copy()

    print("[4/6] Training Monae ...")
    adatas = [rna, atac]
    modal_names = ["RNA", "ATAC"]
    prob = ["NB", "NB"]
    rep = ["X_pre", "X_pre"]

    for idx, adata_i in enumerate(adatas):
        configure_dataset(
            adata_i,
            prob[idx],
            use_highly_variable=True,
            use_rep=rep[idx],
        )

    covel = covel_train(
        adatas,
        graph,
        fit_kws={"directory": os.path.join(save_path, f"{save_key}_monae_train")},
        config=[modal_names, prob, rep],
        result_path=os.path.join(save_path, f"{save_key}_monae_train")
    )

    for modal_name, adata_i in zip(modal_names, adatas):
        adata_i.obsm["embedding"] = covel.encode_data(modal_name, adata_i)

    print("[5/6] Saving embeddings ...")
    # 与统一 benchmark 保持一致：输出与 RNA 对齐的一份结果
    adata_out = sc.concat([rna, atac])
    # adata_out = rna.copy()
    # adata_out.obsm["GLUE"] = rna.obsm["X_glue"].copy()
    # adata_out.obsm[f"{save_key}_emb_latent_omics1"] = rna.obsm["X_glue"].copy()
    # adata_out.obsm[f"{save_key}_emb_latent_omics2"] = atac.obsm["X_glue"].copy()

    latent = pd.DataFrame(
        adata_out.obsm["embedding"],
        index=adata_out.obs_names
    )
    latent.to_csv(os.path.join(save_path, save_key + "_latent.csv"))

    print("[6/6] Clustering ...")
    sc.pp.neighbors(adata_out, n_neighbors=30, use_rep="embedding")
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

    run_monae(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )