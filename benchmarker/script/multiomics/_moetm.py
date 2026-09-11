#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import torch

from _moETM_helpper.utils import calc_weight
from _moETM_helpper.moETM.build_model import build_moETM
from _moETM_helpper.moETM.train import Trainer_moETM
from _moETM_helpper.dataloader import prepare_nips_dataset
from utils import h5_to_h5ad, search_resolution

# -------------------------
# Your NULL rule helpers
# -------------------------
def is_null_path(x: str) -> bool:
    return x is None or str(x).strip().upper() == "NULL"

def infer_mode(atac_file_path: str, adt_file_path: str) -> str:
    atac_null = is_null_path(atac_file_path)
    adt_null  = is_null_path(adt_file_path)
    if (not atac_null) and adt_null:
        return "RNA_ATAC"
    if atac_null and (not adt_null):
        return "RNA_ADT"
    raise ValueError(
        f"Invalid inputs under NULL rule: atac={atac_file_path}, adt={adt_file_path}. "
        f"Exactly ONE of them must be 'NULL'."
    )

def load_pair_as_adata(mod_path: str, rna_path: str, batch_name: str):
    adata_mod = h5_to_h5ad(mod_path)
    adata_rna = h5_to_h5ad(rna_path)

    if adata_mod.n_obs != adata_rna.n_obs:
        raise ValueError(f"Cell number mismatch: rna={adata_rna.n_obs}, mod={adata_mod.n_obs}")

    # if barcodes exist and not equal, warn (still proceed)
    if "barcode" in adata_mod.obs.columns and "barcode" in adata_rna.obs.columns:
        if not np.array_equal(adata_mod.obs["barcode"].values, adata_rna.obs["barcode"].values):
            print("[WARN] RNA and modality barcodes are not identical in order. "
                  "moETM vertical integration usually assumes matched cell order.")

    adata_mod.obs["batch"] = batch_name
    adata_rna.obs["batch"] = batch_name
    return adata_mod, adata_rna

def preprocess_hvg(adata: ad.AnnData, hvg_num: int, is_adt: bool):
    """
    Follow original pipeline:
      normalize_total -> log1p -> highly_variable_genes -> subset to HVGs
    """
    # copy for safe selection
    adata0 = adata.copy()
    adata0.layers["counts"] = adata.X.copy()

    if not is_adt:
        sc.pp.normalize_total(adata0, target_sum=1e4)
        sc.pp.log1p(adata0)

        if hvg_num is not None and int(hvg_num) > 0:
            sc.pp.highly_variable_genes(adata0, n_top_genes=int(hvg_num), flavor="seurat_v3", subset=True, layer="counts")
        else:
            sc.pp.highly_variable_genes(adata0)

    return adata0

def get_adata_mod_single(rna_path: str, mod_path: str, mode: str, hvg_num: int):
    """
    Single dataset case (your 8-arg interface gives single file path).
    """
    batch_names = ["batch0"]
    # mod_list, rna_list = [], []

    mod_adata, rna_adata = load_pair_as_adata(mod_path, rna_path, batch_names[0])
    # mod_list.append(mod_adata)
    # rna_list.append(rna_adata)

    # concat (keep behavior similar to original)
    adata_mod = mod_adata # sc.concat(mod_list, axis=0, join="outer")
    adata_rna = rna_adata # sc.concat(rna_list, axis=0, join="outer")

    # HVG
    # 原脚本：ATAC/RNA 都做 normalize+log+hvg 再用原始矩阵按 hvg subset
    # 这里同样做
    adata_rna_hvg = preprocess_hvg(adata_rna, hvg_num=hvg_num, is_adt=False)
    adata_mod_hvg = preprocess_hvg(adata_mod, hvg_num=hvg_num*10, is_adt=(mode == "RNA_ADT"))

    # prepare_nips_dataset expects (RNA, other_mod)
    adata_mod1, adata_mod2 = prepare_nips_dataset(adata_rna_hvg, adata_mod_hvg)
    return adata_mod1, adata_mod2

# -------------------------
# Training loop (fix hard-coded cuda; keep full-batch coverage)
# -------------------------
def train_moetm(trainer, total_epoch, batch_size, train_set, device):
    X_mod1, X_mod2, batch_index = train_set
    n = X_mod1.shape[0]
    idx = np.arange(n)

    best_embed = None

    for epoch in range(total_epoch):
        np.random.shuffle(idx)
        KL_weight = calc_weight(epoch, total_epoch, 0, 1 / 3, 0, 1e-4)

        loss_all = 0.0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            mb = idx[start:end]

            x1 = X_mod1[mb, :].to(device)
            x2 = X_mod2[mb, :].to(device)
            b  = batch_index[mb].to(device)

            loss, nll1, nll2, kl = trainer.train(x1, x2, b, KL_weight)
            loss_all += float(loss)

        # every 10 epochs compute embedding on full set
        if epoch % 10 == 0:
            trainer.encoder_mod1.to('cpu')
            trainer.encoder_mod2.to('cpu')
            embed = trainer.get_embed(X_mod1, X_mod2)
            trainer.encoder_mod1.cuda()
            trainer.encoder_mod2.cuda()
            best_embed = embed

    if best_embed is None:
        best_embed = trainer.get_embed(X_mod1.to(device), X_mod2.to(device))
    return best_embed

def run_moetm(rna_path, mod_path, mode, save_path, save_key, hvg_num, n_cluster,
             total_epoch=500, batch_size=2000, device="cuda", ):
    adata_mod1, adata_mod2 = get_adata_mod_single(rna_path, mod_path, mode, hvg_num=hvg_num)

    # to dense numpy (as original)
    X1 = adata_mod1.X.toarray() if sp.issparse(adata_mod1.X) else np.asarray(adata_mod1.X)
    X2 = adata_mod2.X.toarray() if sp.issparse(adata_mod2.X) else np.asarray(adata_mod2.X)

    # avoid division by zero
    X1_sum = X1.sum(1)
    X2_sum = X2.sum(1)
    X1_sum[X1_sum == 0] = 1
    X2_sum[X2_sum == 0] = 1

    X1 = X1 / X1_sum[:, None]
    X2 = X2 / X2_sum[:, None]

    X1_T = torch.from_numpy(X1).float()
    X2_T = torch.from_numpy(X2).float()

    # batch indices: from prepare_nips_dataset output obs['batch_indices'] (original behavior)
    batch_index = np.asarray(adata_mod1.obs["batch_indices"]).astype(np.int64)
    batch_T = torch.from_numpy(batch_index).to(torch.int64)

    num_batch = len(torch.unique(batch_T))
    input_dim_mod1 = X1_T.shape[1]
    input_dim_mod2 = X2_T.shape[1]

    num_topic = 100
    emd_dim = 400

    encoder_mod1, encoder_mod2, decoder, optimizer = build_moETM(
        input_dim_mod1, input_dim_mod2, num_batch, num_topic=num_topic, emd_dim=emd_dim
    )
    trainer = Trainer_moETM(encoder_mod1, encoder_mod2, decoder, optimizer)

    # device
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    dev = torch.device("cuda" if use_cuda else "cpu")
    trainer.encoder_mod1.to(dev)
    trainer.encoder_mod2.to(dev)
    trainer.decoder.to(dev) if hasattr(trainer, "decoder") else None

    embed = train_moetm(
        trainer,
        total_epoch=total_epoch,
        batch_size=batch_size,
        train_set=(X1_T, X2_T, batch_T),
        device=dev,
    )

    # embed expected dict with key 'delta' (as your original print(result['delta'].shape))
    if not isinstance(embed, dict) or "delta" not in embed:
        raise ValueError("trainer.get_embed() did not return dict with key 'delta'.")

    delta = embed["delta"]
    if torch.is_tensor(delta):
        delta = delta.detach().cpu().numpy()
    else:
        delta = np.asarray(delta)
    
    adata_mod1.obsm["latent"] = delta
    sc.pp.neighbors(adata_mod1, use_rep="latent")
    sc.tl.umap(adata_mod1)
    res = search_resolution(adata_mod1, fixed_clus_count=n_cluster)
    sc.tl.leiden(adata_mod1, resolution=res, key_added="cluster")

    ## save UMAP
    umap = pd.DataFrame(adata_mod1.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata_mod1.obs_names)
    umap.insert(2, "cluster", adata_mod1.obs['cluster'].values)
    umap.to_csv(os.path.join(save_path, save_key + ".csv"))

    # add barcode (from RNA side)
    barcodes = None
    if "barcode" in adata_mod1.obs.columns:
        barcodes = adata_mod1.obs["barcode"].astype(str).values

    os.makedirs(save_path, exist_ok=True)

    out_csv = os.path.join(save_path, f"{save_key}_latent.csv")
    df = pd.DataFrame(delta)
    if barcodes is not None and len(barcodes) == df.shape[0]:
        df.insert(0, "barcode", barcodes)
    df.to_csv(out_csv, index=False)

    return out_csv

# -------------------------
# CLI (your 8 args)
# -------------------------
def main():
    parser = argparse.ArgumentParser("moETM unified CLI (RNA+ATAC or RNA+ADT)")

    parser.add_argument("n_cluster", type=int)
    parser.add_argument("rna_file_path", type=str)
    parser.add_argument("atac_file_path", type=str)
    parser.add_argument("adt_file_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("save_key", type=str)
    parser.add_argument("hvg_num", type=int)
    parser.add_argument("batch_key", type=str)

    # optional knobs (keep close to original defaults)
    parser.add_argument("--total_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    t0 = time.time()
    mode = infer_mode(args.atac_file_path, args.adt_file_path)
    print("Mode =", mode)

    if mode == "RNA_ATAC":
        mod_path = args.atac_file_path
    else:
        mod_path = args.adt_file_path

    out_csv = run_moetm(
        rna_path=args.rna_file_path,
        mod_path=mod_path,
        mode=mode,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        n_cluster=args.n_cluster,
        total_epoch=args.total_epoch,
        batch_size=args.batch_size,
        device=args.device,
    )

    # elapsed = time.time() - t0
    # time_csv = os.path.join(args.save_path, f"{args.save_key}_time.csv")
    # pd.DataFrame({"time_sec": [elapsed]}).to_csv(time_csv, index=False)

    print("[Done] embedding csv:", out_csv)
    # print("[Done] time csv:", time_csv)
    # print(f"[Done] elapsed={elapsed:.2f}s")

if __name__ == "__main__":
    main()