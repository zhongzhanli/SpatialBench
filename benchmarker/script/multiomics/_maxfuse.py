import os
import sys

# sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/")
# import scglue

import argparse
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/MaxFuse_devo/09302022V/")
import match

from utils import h5_to_h5ad, search_resolution

warnings.filterwarnings("ignore")
import sklearn
from sklearn.preprocessing import normalize
def tfidf(X) -> np.ndarray:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)

    Parameters
    ----------
    X
        Input matrix

    Returns
    -------
    np.ndarray
        TF-IDF normalized matrix
    """
    idf = X.shape[0] / X.sum(axis=0)
    if sp.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

def lsi(
        adata: ad.AnnData,
        n_components: int=51,
        use_highly_variable = None,
        **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi

def parse_args():
    parser = argparse.ArgumentParser("MaxFuse unified runner")

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


def to_dense_array(x):
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def preprocess_shared_rna(rna_shared, hvg_num, batch_key):
    """
    对 shared gene 空间中的 RNA 做预处理，用于 shared_arr1 / active_arr1
    """
    rna_shared = ad.AnnData(to_dense_array(rna_shared.X).copy())
    hvg_kwargs = dict(flavor="seurat_v3", n_top_genes=hvg_num)
    # if not is_null_string(batch_key):
    #     hvg_kwargs["batch_key"] = batch_key

    sc.pp.normalize_total(rna_shared)
    sc.pp.log1p(rna_shared)
    sc.pp.highly_variable_genes(rna_shared, **hvg_kwargs)
    sc.pp.scale(rna_shared)

    return rna_shared


def preprocess_shared_ga(activity_shared):
    """
    对 gene activity 的 shared gene 空间做预处理，用于 shared_arr2
    """
    activity_shared = ad.AnnData(to_dense_array(activity_shared.X).copy())
    sc.pp.normalize_total(activity_shared)
    sc.pp.log1p(activity_shared)
    sc.pp.scale(activity_shared)
    return activity_shared


def preprocess_atac_lsi(atac):
    """
    对 ATAC 做 LSI，作为 active_arr2
    """
    lsi(atac, n_components=50, n_iter=15)
    atac_lsi = ad.AnnData(np.asarray(atac.obsm["X_lsi"], dtype=np.float32))
    atac_lsi.obs_names = atac.obs_names.copy()
    return atac_lsi


def run_maxfuse(
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
            "MaxFuse runner requires gene activity input in adt_file_path under the unified interface."
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

    print(f"RNA shape: {rna.shape}")
    print(f"ATAC shape: {atac.shape}")
    print(f"Gene activity shape: {gene_activity.shape}")

    print("[2/6] Building MaxFuse inputs ...")
    shared_genes = np.intersect1d(rna.var_names, gene_activity.var_names)
    if len(shared_genes) == 0:
        raise ValueError("RNA and gene activity have no overlapping genes!")

    rna_shared = rna[:, shared_genes].copy()
    activity_shared = gene_activity[:, shared_genes].copy()

    rna_shared = preprocess_shared_rna(
        rna_shared=rna_shared,
        hvg_num=min(hvg_num, len(shared_genes)),
        batch_key=batch_key
    )
    activity_shared = preprocess_shared_ga(activity_shared)

    vgenes = np.asarray(rna_shared.var.highly_variable).astype(bool)
    if vgenes.sum() == 0:
        raise ValueError("No highly variable genes retained in shared RNA space!")

    # shared features
    rnaC_shared = to_dense_array(rna_shared[:, vgenes].X)
    atac_shared = to_dense_array(activity_shared[:, vgenes].X)

    # active features
    rnaC_active = to_dense_array(rna_shared[:, vgenes].X)
    atac_lsi = preprocess_atac_lsi(atac)
    atac_active = to_dense_array(atac_lsi.X)

    print("[3/6] Running MaxFuse ...")
    spm = match.MaxFuse(
        shared_arr1=np.array(rnaC_shared),
        shared_arr2=np.array(atac_shared),
        active_arr1=np.array(rnaC_active),
        active_arr2=np.array(atac_active),
        method="centroid_shrinkage",
        labels1=None,
        labels2=None
    )

    spm.split_into_batches(
        max_outward_size=5000,
        matching_ratio=5,
        metacell_size=2,
        method="binning",
        verbose=True,
        seed=42
    )

    spm.construct_graphs(
        n_neighbors1=15,
        n_neighbors2=15,
        svd_components1=30,
        svd_components2=15,
        resolution1=2,
        resolution2=2,
        randomized_svd=False,
        svd_runs=1,
        resolution_tol=0.1,
        leiden_runs=1,
        leiden_seed=None,
        verbose=True
    )

    spm.find_initial_pivots(
        wt1=0.7,
        wt2=0.7,
        svd_components1=20,
        svd_components2=20,
        randomized_svd=False,
        svd_runs=1,
        verbose=True
    )

    spm.refine_pivots(
        wt1=0.7,
        wt2=0.7,
        svd_components1=100,
        svd_components2=None,
        cca_components=20,
        filter_prop=0.0,
        n_iters=8,
        randomized_svd=False,
        svd_runs=1,
        verbose=True
    )

    spm.filter_bad_matches(
        target="pivot",
        filter_prop=0.4,
        verbose=True
    )

    spm.propagate(
        wt1=0.7,
        wt2=0.7,
        svd_components1=30,
        svd_components2=None,
        randomized_svd=False,
        svd_runs=1,
        verbose=True
    )

    spm.filter_bad_matches(
        target="propagated",
        filter_prop=0.0,
        verbose=True
    )

    print("[4/6] Extracting embeddings ...")
    dim_use = 15
    rna_cca, atac_cca = spm.get_embedding(
        active_arr1=spm.active_arr1,
        active_arr2=spm.active_arr2,
        refit=False,
        matching=None,
        order=None,
        cca_components=20,
        cca_max_iter=None
    )

    rna.obsm["embed"] = np.asarray(rna_cca[:, :dim_use]).copy()
    atac.obsm["embed"] = np.asarray(atac_cca[:, :dim_use]).copy()

    print("[5/6] Saving embeddings ...")
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

    run_maxfuse(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )