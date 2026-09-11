#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/Benchmark/external/sciPENN/")

import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import scanpy as sc
import anndata as ad

from sciPENN.sciPENN_API import sciPENN_API
from utils import h5_to_h5ad, search_resolution

# -------------------------
# Helpers
# -------------------------
def is_null_path(x: str) -> bool:
    return x is None or str(x).strip().upper() == "NULL"

def decode_h5_str(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr], dtype=str)
    return arr.astype(str)

def build_anndata_from_h5(h5_path: str, batch_value: str):
    """
    Build AnnData for sciPENN:
      - X: cells×features (csr)
      - obs_names: barcodes if available else cell_*
      - var_names: features if available else {feature_prefix}_*
      - obs['batch']: str
    """
    adata = h5_to_h5ad(h5_path)
    adata.obs["batch"] = str(batch_value)
    return adata

def ensure_string_obs_var(adata: ad.AnnData):
    """
    Your original code tries to decode bytes in obs/var. We keep a safe version.
    """
    # obs/var columns (rarely used by sciPENN, but keep consistent)
    for key in list(adata.obs.columns):
        if adata.obs[key].dtype == "O":
            adata.obs[key] = adata.obs[key].apply(lambda x: x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x)
    for key in list(adata.var.columns):
        if adata.var[key].dtype == "O":
            adata.var[key] = adata.var[key].apply(lambda x: x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x)

    # obs_names / var_names
    adata.obs_names = adata.obs_names.map(str)
    adata.var_names = adata.var_names.map(str)
    return adata

# -------------------------
# Core sciPENN run
# -------------------------
def run_scipenn_single(rna_path: str, adt_path: str, seed: int = 1):
    """
    Single pair (vertical integration): RNA+ADT
    Returns: embedding AnnData or numpy array depending on sciPENN version.
    """
    random.seed(seed)
    np.random.seed(seed)

    # build per-modality AnnData
    rna = build_anndata_from_h5(rna_path, batch_value="0")
    adt = build_anndata_from_h5(adt_path, batch_value="0")

    rna = ensure_string_obs_var(rna)
    adt = ensure_string_obs_var(adt)

    # sanity: same n_obs
    if rna.n_obs != adt.n_obs:
        raise ValueError(f"Cell number mismatch: RNA={rna.n_obs} vs ADT={adt.n_obs}")

    # if barcodes exist and differ, warn (sciPENN expects matched)
    if not np.array_equal(rna.obs_names.values, adt.obs_names.values):
        print("[WARN] RNA/ADT barcodes not identical in order; proceeding by row order. "
              "If results look wrong, align cells before running sciPENN.")

    # sciPENN expects lists of trainsets
    gene_trainsets = [rna]
    protein_trainsets = [adt]
    train_batchkeys = ["batch"]  # IMPORTANT: this is the obs column name

    model = sciPENN_API(
        gene_trainsets=gene_trainsets,
        protein_trainsets=protein_trainsets,
        train_batchkeys=train_batchkeys,
        min_cells=0,
        min_genes=0,
    )

    # keep defaults close to original (very long training)
    model.train(
        quantiles=[0.1, 0.25, 0.75, 0.9],
        n_epochs=10000,
        ES_max=12,
        decay_max=6,
        decay_step=0.1,
        lr=10 ** (-3),
        weights_dir="pbmc_to_pbmc",
        load=False,
    )

    emb = model.embed()  # usually returns AnnData with .X as embedding
    return emb, rna.obs_names.values


def save_embedding_csv(emb, barcodes, out_csv: str):
    """
    sciPENN embed() often returns AnnData with .X as array.
    Handle both cases.
    """
    if hasattr(emb, "X"):
        X = emb.X
    else:
        X = emb

    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    df = pd.DataFrame(X)
    if barcodes is not None and len(barcodes) == df.shape[0]:
        df.insert(0, "barcode", barcodes.astype(str))
    df.to_csv(out_csv, index=False)


# -------------------------
# CLI (your 8 args)
# -------------------------
def main():
    parser = argparse.ArgumentParser("sciPENN unified CLI (RNA+ADT only)")

    parser.add_argument("n_cluster", type=int)
    parser.add_argument("rna_file_path", type=str)
    parser.add_argument("atac_file_path", type=str)
    parser.add_argument("adt_file_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("save_key", type=str)
    parser.add_argument("hvg_num", type=int)
    parser.add_argument("batch_key", type=str)

    # optional
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    # scenario check (sciPENN only supports RNA+ADT)
    if is_null_path(args.adt_file_path):
        raise ValueError("sciPENN only supports RNA+ADT. You provided adt_file_path='NULL'.")
    if not is_null_path(args.atac_file_path):
        raise ValueError("sciPENN only supports RNA+ADT. You provided atac_file_path != 'NULL' (RNA+ATAC not supported).")

    os.makedirs(args.save_path, exist_ok=True)

    t0 = time.time()
    emb, barcodes = run_scipenn_single(args.rna_file_path, args.adt_file_path, seed=args.seed)
    adata = h5_to_h5ad(args.rna_file_path)
    adata.obsm["latent"] = np.array(emb.X)
    sc.pp.neighbors(adata, use_rep="latent")
    sc.tl.umap(adata)
    res = search_resolution(adata, fixed_clus_count=args.n_cluster)
    sc.tl.leiden(adata, resolution=res, key_added="cluster")

    ## save UMAP
    umap = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names)
    umap.insert(2, "cluster", adata.obs['cluster'].values)
    umap.to_csv(os.path.join(args.save_path, args.save_key + ".csv"))

    out_csv = os.path.join(args.save_path, f"{args.save_key}_latent.csv")
    save_embedding_csv(emb, barcodes, out_csv)

    # elapsed = time.time() - t0
    # time_csv = os.path.join(args.save_path, f"{args.save_key}_time.csv")
    # pd.DataFrame({"time_sec": [elapsed]}).to_csv(time_csv, index=False)

    # print("[Done] latent:", out_csv)
    # print("[Done] time:", time_csv)
    # print(f"[Done] elapsed={elapsed:.2f}s")

if __name__ == "__main__":
    main()