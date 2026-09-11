import sys
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/Benchmark/external/MISO/")

import os
import argparse
import warnings
import random
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import torch

from PIL import Image
from miso.utils import *
from miso import Miso

from utils import h5_to_h5ad, search_resolution

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

seed = 100


def parse_args():
    parser = argparse.ArgumentParser("MISO unified runner")

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


def infer_mode(atac_file_path, adt_file_path):
    atac_is_null = is_null_string(atac_file_path)
    adt_is_null = is_null_string(adt_file_path)

    if (not atac_is_null) and adt_is_null:
        return "atac"
    elif atac_is_null and (not adt_is_null):
        return "protein"
    else:
        raise ValueError(
            "Invalid input"
        )


def set_seed(seed=100):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = "cuda:1"
            print(f"CUDA is available. GPU: {torch.cuda.get_device_name(1)}")
        else:
            device = "cuda:0"
            print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA is not available. Using CPU.")
    return device


def run_miso(
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

    mode = infer_mode(atac_file_path, adt_file_path)
    print(f"[Info] inferred mode = {mode}")

    print(f"[Info] hvg_num = {hvg_num} (not used, kept for interface consistency)")
    print(f"[Info] batch_key = {batch_key} (not used, kept for interface consistency)")

    set_seed(seed)
    device = get_device()

    print("[1/6] Loading data ...")
    rna = h5_to_h5ad(rna_file_path)
    sc.pp.highly_variable_genes(rna,
                                flavor="seurat_v3",
                                n_top_genes=hvg_num,
                                subset=True,
                                # batch_key=batch_key 
    )
    rna.var_names_make_unique()

    if mode == "atac":
        omics2 = h5_to_h5ad(atac_file_path)
        sc.pp.highly_variable_genes(rna,
                                    flavor="seurat_v3",
                                    n_top_genes=hvg_num*10,
                                    subset=True,
                                    # batch_key=batch_key 
        )
        omics2_modality = "atac"
    else:
        omics2 = h5_to_h5ad(adt_file_path)
        omics2_modality = "protein"

    omics2.var_names_make_unique()

    omics2 = omics2[rna.obs_names].copy()

    print(f"RNA shape: {rna.shape}")
    print(f"Omics2 shape: {omics2.shape}")
    print(f"Omics2 modality for preprocess: {omics2_modality}")

    print("[2/6] Preprocessing ...")
    rna = preprocess(rna, modality="rna")
    omics2 = preprocess(omics2, modality=omics2_modality)

    print("[3/6] Training MISO ...")
    model = Miso(
        [rna, omics2],
        ind_views="all",
        combs="all",
        sparse=False,
        device=device
    )
    model.train()

    adata_rna = h5_to_h5ad(args.rna_file_path)
    emb = pd.DataFrame(model.emb, index=adata_rna.obs_names)
    emb.to_csv(os.path.join(save_path, f"{save_key}_latent.csv"))
    adata_rna.obsm["latent"] = np.array(model.emb)
    sc.pp.neighbors(adata_rna, use_rep="latent")
    sc.tl.umap(adata_rna)
    res = search_resolution(adata_rna, fixed_clus_count=n_cluster)
    sc.tl.leiden(adata_rna, resolution=res, key_added="cluster")

    ## save UMAP
    umap = pd.DataFrame(adata_rna.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata_rna.obs_names)
    umap.insert(2, "cluster", adata_rna.obs['cluster'].values)
    umap.to_csv(os.path.join(save_path, save_key + ".csv"))

if __name__ == "__main__":
    args = parse_args()

    run_miso(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )