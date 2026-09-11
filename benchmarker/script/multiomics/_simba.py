import os
import sys
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/simba/")
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import simba as si



from utils import h5_to_h5ad, search_resolution


def parse_args():
    parser = argparse.ArgumentParser("SIMBA unified runner")

    parser.add_argument("n_cluster", type=int)
    parser.add_argument("rna_file_path", type=str)
    parser.add_argument("atac_file_path", type=str)
    parser.add_argument("adt_file_path", type=str)   # 这里表示 gene activity 文件
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
    adata.X = np.asarray(adata.X).astype(float)
    return adata


def parse_peak_bed_like_names(atac):
    """
    将 peak 名解析为 chrom / chromStart / chromEnd
    支持:
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
    atac.var["chr"] = atac.var["chrom"].values
    atac.var["start"] = atac.var["chromStart"].values
    atac.var["end"] = atac.var["chromEnd"].values

    valid = (
        atac.var["chrom"].notna()
        & atac.var["chromStart"].notna()
        & atac.var["chromEnd"].notna()
    )
    atac = atac[:, valid].copy()
    return atac


def preprocess_rna(rna, hvg_num):
    rna = ensure_dense(rna)

    si.pp.filter_genes(rna, min_n_cells=3)
    si.pp.normalize(rna, method="lib_size")
    si.pp.log_transform(rna)
    si.pp.select_variable_genes(rna, n_top_genes=hvg_num)

    return rna


def preprocess_atac(atac):
    atac = ensure_dense(atac)

    si.pp.filter_peaks(atac, min_n_cells=atac.shape[0]*0.02)
    si.pp.pca(atac, n_components=50)
    si.pp.select_pcs(atac, n_pcs=40)
    si.pp.select_pcs_features(atac)

    atac = parse_peak_bed_like_names(atac)
    return atac


def preprocess_gene_activity(gene_activity):
    gene_activity = ensure_dense(gene_activity)

    si.pp.filter_genes(gene_activity, min_n_cells=3)
    si.pp.cal_qc_rna(gene_activity)
    si.pp.normalize(gene_activity, method="lib_size")
    si.pp.log_transform(gene_activity)

    return gene_activity


def run_simba(
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
            "SIMBA requires gene activity input. "
            "Under the unified interface, please provide gene activity file in adt_file_path."
        )

    os.makedirs(save_path, exist_ok=True)

    # 给每次运行单独目录，避免不同任务互相覆盖
    workdir = os.path.join(save_path, f"{save_key}_simba_workdir")
    os.makedirs(workdir, exist_ok=True)
    si.settings.set_workdir(workdir)
    si.settings.pbg_params["workers"] = 10

    print("[1/6] Loading data ...")
    rna = h5_to_h5ad(rna_file_path)
    atac = h5_to_h5ad(atac_file_path)
    gene_activity = h5_to_h5ad(adt_file_path)

    rna.var_names_make_unique()
    rna.obs_names_make_unique()
    atac.var_names_make_unique()
    atac.obs_names_make_unique()
    gene_activity.var_names_make_unique()
    gene_activity.obs_names_make_unique()

    # # 三者统一按共同细胞对齐
    # common_obs = rna.obs_names.intersection(atac.obs_names)
    # common_obs = common_obs.intersection(gene_activity.obs_names)
    # if len(common_obs) == 0:
    #     raise ValueError("RNA, ATAC and gene activity have no overlapping obs_names!")

    # rna = rna[common_obs].copy()
    # atac = atac[common_obs].copy()
    # gene_activity = gene_activity[common_obs].copy()

    # # 统一顺序
    # atac = atac[rna.obs_names, :].copy()
    gene_activity = gene_activity[atac.obs_names, :].copy()

    print(f"RNA shape: {rna.shape}")
    print(f"ATAC shape: {atac.shape}")
    print(f"Gene activity shape: {gene_activity.shape}")

    print("[2/6] Preprocessing ...")
    rna = preprocess_rna(rna, hvg_num=hvg_num)
    atac = preprocess_atac(atac)
    gene_activity = preprocess_gene_activity(gene_activity)

    # 保存原始 cell id，便于最后恢复
    original_obs_names = rna.obs_names.copy()

    # SIMBA 内部需要区分不同模态 cell 节点
    rna.obs_names = pd.Index([f"{x}_rna" for x in rna.obs_names])
    atac.obs_names = pd.Index([f"{x}_atac" for x in atac.obs_names])
    gene_activity.obs_names = pd.Index([f"{x}_atac" for x in gene_activity.obs_names])

    print("[3/6] Inferring cross-modal edges ...")
    adata_CrnaCatac = si.tl.infer_edges(rna, gene_activity, n_components=15, k=15)
    si.tl.trim_edges(adata_CrnaCatac, cutoff=0.6)

    print("[4/6] Training SIMBA ...")
    si.tl.gen_graph(
        list_CP=[atac],
        list_CG=[rna],
        list_CC=[adata_CrnaCatac],
        copy=False,
        use_highly_variable=True,
        use_top_pcs=True,
        dirname="graph0"
    )

    si.tl.pbg_train(auto_wd=False, save_wd=True)
    dict_adata = si.read_embedding()

    adata_C = dict_adata["C"]    # ATAC cells
    adata_C2 = dict_adata["C2"]  # RNA cells

    atac.obsm["embed"] = adata_C[atac.obs_names].X
    rna.obsm["embed"] = adata_C2[rna.obs_names].X

    print("[5/6] Saving embeddings ...")
    # 统一输出：只保留与 RNA 对齐的一份结果
    adata_out = sc.concat([rna, atac])
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

    run_simba(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )