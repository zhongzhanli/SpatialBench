import os
import argparse
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import muon as mu
import anndata as ad
import scanpy as sc
import scconfluence

from utils import h5_to_h5ad, search_resolution


def parse_args():
    parser = argparse.ArgumentParser("scConfluence unified runner")

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
    adata.X = np.asarray(adata.X)
    return adata


def run_scconfluence(
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
            "scConfluence runner requires gene activity input in adt_file_path under the unified interface."
        )

    os.makedirs(save_path, exist_ok=True)

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

    # common_obs = rna.obs_names.intersection(atac.obs_names)
    # common_obs = common_obs.intersection(gene_activity.obs_names)
    # if len(common_obs) == 0:
    #     raise ValueError("RNA, ATAC and gene activity have no overlapping obs_names!")

    # rna = rna[common_obs].copy()
    # atac = atac[common_obs].copy()
    # gene_activity = gene_activity[common_obs].copy()

    # atac = atac[rna.obs_names, :].copy()
    gene_activity = gene_activity[atac.obs_names, :].copy()

    rna = ensure_dense(rna)
    atac = ensure_dense(atac)
    gene_activity = ensure_dense(gene_activity)

    print(f"RNA shape: {rna.shape}")
    print(f"ATAC shape: {atac.shape}")
    print(f"Gene activity shape: {gene_activity.shape}")

    print("[2/6] Preprocessing ...")
    sc.pp.filter_genes(rna, min_cells=20)
    sc.pp.filter_genes(atac, min_cells=atac.shape[0]*0.02)

    cm_genes = sorted(list(set(rna.var_names) & set(gene_activity.var_names)))
    if len(cm_genes) == 0:
        raise ValueError("RNA and gene activity have no overlapping genes!")

    cm_features_rna = rna[:, cm_genes].copy()
    cm_features_atac = gene_activity[:, cm_genes].copy()

    sc.pp.normalize_total(cm_features_rna, target_sum=10000.0)
    sc.pp.log1p(cm_features_rna)

    sc.pp.normalize_total(cm_features_atac, target_sum=10000.0)
    sc.pp.log1p(cm_features_atac)

    cm_hvg_genes = sc.pp.highly_variable_genes(
        cm_features_rna,
        n_top_genes=min(hvg_num, cm_features_rna.n_vars),
        subset=False,
        inplace=False
    )
    hv_mask = np.asarray(cm_hvg_genes["highly_variable"]).astype(bool)
    if hv_mask.sum() == 0:
        raise ValueError("No highly variable genes retained for cross-modal features!")

    cm_features_rna = cm_features_rna[:, hv_mask].copy()
    cm_features_atac = cm_features_atac[:, hv_mask].copy()

    sc.pp.scale(cm_features_rna)
    sc.pp.scale(cm_features_atac)

    print("[3/6] Building MuData ...")
    mdata = mu.MuData({"rna": rna.copy(), "atac": atac.copy()})
    mdata.uns["cross_rna+atac"] = cdist(
        cm_features_rna.X,
        cm_features_atac.X,
        metric="correlation"
    )
    mdata.uns["cross_keys"] = ["cross_rna+atac"]

    mdata["rna"].layers["counts"] = mdata["rna"].X.copy()

    sc.pp.normalize_total(mdata["rna"], target_sum=10000.0)
    sc.pp.log1p(mdata["rna"])

    raw_hvg = sc.pp.highly_variable_genes(
        mdata["rna"],
        layer="counts",
        n_top_genes=min(hvg_num, mdata["rna"].n_vars),
        subset=False,
        inplace=False,
        flavor="seurat_v3"
    )["highly_variable"].values

    norm_hvg = sc.pp.highly_variable_genes(
        mdata["rna"],
        n_top_genes=min(hvg_num, mdata["rna"].n_vars),
        subset=False,
        inplace=False
    )["highly_variable"].values

    rna_hv_mask = np.logical_or(raw_hvg, norm_hvg)
    mdata.mod["rna"] = mdata["rna"][:, rna_hv_mask].copy()

    sc.tl.pca(mdata["rna"], n_comps=100, zero_center=None)

    mu.atac.pp.tfidf(mdata["atac"], log_tf=True, log_idf=True)
    sc.tl.pca(mdata["atac"], n_comps=100, zero_center=None)

    print("[4/6] Training scConfluence ...")
    torch.manual_seed(1792)

    autoencoders = {
        "rna": scconfluence.unimodal.AutoEncoder(
            mdata["rna"],
            modality="rna",
            rep_in="X_pca",
            rep_out="counts",
            batch_key=None,
            n_hidden=64,
            n_latent=16,
            type_loss="zinb"
        ),
        "atac": scconfluence.unimodal.AutoEncoder(
            mdata["atac"],
            modality="atac",
            rep_in="X_pca",
            rep_out=None,
            batch_key=None,
            n_hidden=64,
            n_latent=16,
            type_loss="l2",
            reconstruction_weight=5.0
        )
    }

    model = scconfluence.model.ScConfluence(
        mdata=mdata,
        unimodal_aes=autoencoders,
        mass=0.5,
        reach=0.3,
        iot_loss_weight=0.01,
        sinkhorn_loss_weight=0.1
    )

    model_dir = os.path.join(save_path, f"{save_key}_scconfluence_train")
    os.makedirs(model_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    model.fit(
        save_path=model_dir,
        use_cuda=use_cuda,
        max_epochs=1000
    )

    print("[5/6] Saving embeddings ...")
    latent_all = model.get_latent(use_cuda=use_cuda)
    latent_all = np.asarray(latent_all)

    n_rna = rna.n_obs
    n_atac = atac.n_obs

    if latent_all.shape[0] != (n_rna + n_atac):
        raise ValueError(
            f"Unexpected latent shape {latent_all.shape}; expected first dimension {n_rna + n_atac}."
        )

    rna.obsm["embed"] = latent_all[:n_rna].copy()
    atac.obsm["embed"] = latent_all[n_rna:n_rna + n_atac].copy()

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

    run_scconfluence(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )