#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import h5py
import scipy.sparse as sp
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scanpy as sc

from _matilda_learn.model import CiteAutoencoder_CITEseq, CiteAutoencoder_SHAREseq, CiteAutoencoder_TEAseq
from _matilda_learn.train import train_model
from _matilda_learn.util import (
    setup_seed, MyDataset,
    read_h5_data, read_fs_label,
    compute_zscore, compute_log2,
    get_vae_simulated_data_from_sampling,
    get_encodings,
)
from utils import h5_to_h5ad, search_resolution

# -------------------------
# Helpers
# -------------------------
def is_null_path(x: str) -> bool:
    return x is None or str(x).strip().upper() == "NULL"

def infer_mode(atac_file_path: str, adt_file_path: str) -> str:
    atac_null = is_null_path(atac_file_path)
    adt_null  = is_null_path(adt_file_path)
    if atac_null and (not adt_null):
        return "CITEseq"    # RNA + ADT
    if adt_null and (not atac_null):
        return "SHAREseq"   # RNA + ATAC
    raise ValueError(
        f"Invalid inputs: atac_file_path={atac_file_path}, adt_file_path={adt_file_path}. "
        f"Exactly ONE of them must be 'NULL'."
    )

def decode_h5_str(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr], dtype=str)
    return arr.astype(str)

def maybe_load_cty_auto(rna_file_path: str, save_path: str, device_str: str):
    """
    Auto-find label file:
      - same dir as rna: cty.csv / celltype.csv
      - save_path/cty.csv
    Return torch label or None if not found.
    """
    candidates = [
        os.path.join(os.path.dirname(rna_file_path), "cty.csv"),
        os.path.join(os.path.dirname(rna_file_path), "celltype.csv"),
        os.path.join(save_path, "cty.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            label = read_fs_label(p, device_str)  # util.read_fs_label
            return label, p
    return None, None

def load_barcodes_if_any(h5_path: str):
    try:
        with h5py.File(h5_path, "r") as f:
            if "matrix" in f and "barcodes" in f["matrix"]:
                return decode_h5_str(f["matrix/barcodes"][:])
    except Exception:
        pass
    return None

def load_features(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        return decode_h5_str(f["matrix/features"][:])

def standardize_latent_to_cells_by_dim(latent):
    """
    get_encodings may return shape (n_cells, z_dim) or (z_dim, n_cells) depending on implementation.
    We convert to (n_cells, z_dim) with a safe heuristic.
    """
    if torch.is_tensor(latent):
        arr = latent.detach().cpu().numpy()
    else:
        arr = np.asarray(latent)

    # heuristic: if first dim is small (<=512) and less than second dim -> likely z_dim x n_cells
    if arr.ndim == 2 and (arr.shape[0] <= 512) and (arr.shape[0] < arr.shape[1]):
        arr = arr.T
    return arr

def auto_find_test_files(train_rna, train_atac, train_adt):
    """
    No extra args -> auto-detect:
      same folder contains:
        test_rna.h5
        and paired modality:
          - CITEseq: test_adt.h5
          - SHAREseq: test_atac.h5
    """
    folder = os.path.dirname(train_rna)
    test_rna  = os.path.join(folder, "test_rna.h5")
    test_atac = os.path.join(folder, "test_atac.h5")
    test_adt  = os.path.join(folder, "test_adt.h5")

    if not os.path.exists(test_rna):
        return None, None, None

    if is_null_path(train_atac) and (not is_null_path(train_adt)):
        # CITEseq
        if os.path.exists(test_adt):
            return test_rna, "NULL", test_adt
        return None, None, None

    if is_null_path(train_adt) and (not is_null_path(train_atac)):
        # SHAREseq
        if os.path.exists(test_atac):
            return test_rna, test_atac, "NULL"
        return None, None, None

    return None, None, None

def build_data_and_dl(rna_path, atac_path, adt_path, cty_path, batch_size, device_str, hvg_num):
    """
    Read + preprocess + DataLoader (shuffle=False for stable output order).
    """
    rna = read_h5_data_v2(rna_path, device_str, hvg_num=hvg_num)

    # label
    if (cty_path is not None) and (not is_null_path(cty_path)) and os.path.exists(cty_path):
        label = read_fs_label(cty_path, device_str)
    else:
        label = torch.zeros(rna.shape[0], dtype=torch.long).to(device_str)

    if (not is_null_path(adt_path)) and (not os.path.exists(adt_path)):
        raise FileNotFoundError(f"ADT file not found: {adt_path}")
    if (not is_null_path(atac_path)) and (not os.path.exists(atac_path)):
        raise FileNotFoundError(f"ATAC file not found: {atac_path}")

    if (not is_null_path(adt_path)) and is_null_path(atac_path):
        # CITEseq
        adt = read_h5_data_v2(adt_path, device_str, hvg_num=-1)
        nfeatures_rna = rna.shape[1]
        nfeatures_mod = adt.shape[1]
        rna = compute_zscore(compute_log2(rna))
        adt = compute_zscore(compute_log2(adt))
        data = torch.cat((rna, adt), 1)
        mode = "CITEseq"

    elif (not is_null_path(atac_path)) and is_null_path(adt_path):
        # SHAREseq
        atac = read_h5_data_v2(atac_path, device_str, hvg_num=hvg_num*10)
        nfeatures_rna = rna.shape[1]
        nfeatures_mod = atac.shape[1]
        rna = compute_zscore(compute_log2(rna))
        atac = compute_zscore(compute_log2(atac))
        data = torch.cat((rna, atac), 1)
        mode = "SHAREseq"

    else:
        raise ValueError("Unsupported modality combination under NULL rule.")

    ds = MyDataset(data, label)
    train_dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    eval_dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    return data, label, train_dl, eval_dl, mode, nfeatures_rna, nfeatures_mod

def build_model(mode, nfeatures_rna, nfeatures_mod, hidden_rna, hidden_mod, z_dim, classify_dim, device):
    if mode == "CITEseq":
        model = CiteAutoencoder_CITEseq(nfeatures_rna, nfeatures_mod, hidden_rna, hidden_mod, z_dim, classify_dim)
    elif mode == "SHAREseq":
        model = CiteAutoencoder_SHAREseq(nfeatures_rna, nfeatures_mod, hidden_rna, hidden_mod, z_dim, classify_dim)
    else:
        # not used under NULL rule
        model = CiteAutoencoder_TEAseq(...)
    model = model.to(device)
    return model

def load_best_checkpoint_if_any(model, model_save_path, device):
    ckpt = os.path.join(model_save_path, "model_best.pth.tar")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["state_dict"], strict=True)
        return True
    return False

def read_h5_data_v2(data_path: str, device: str, hvg_num: int = 0):
    cuda = True if device == "cuda" and torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    adata = h5_to_h5ad(data_path)  # X: cells × features (CSR or sparse)
    if hvg_num > 0:
        sc.pp.highly_variable_genes(adata,
                                flavor="seurat_v3",
                                n_top_genes=hvg_num,
                                subset=True,
    )

    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()

        X = X.toarray()
    else:
        X = np.asarray(X)

    X = X.astype(np.float32, copy=False)
    t = torch.from_numpy(X)
    t = Variable(t.type(FloatTensor))
    return t

parser = argparse.ArgumentParser("Matilda All-in-One CLI")

parser.add_argument("n_cluster", type=int)
parser.add_argument("rna_file_path", type=str)
parser.add_argument("atac_file_path", type=str)
parser.add_argument("adt_file_path", type=str)
parser.add_argument("save_path", type=str)
parser.add_argument("save_key", type=str)
parser.add_argument("hvg_num", type=int)
parser.add_argument("batch_key", type=str)

# training params (optional)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument("--augmentation", type=bool, default=True)

# model params
parser.add_argument("--z_dim", type=int, default=100)
parser.add_argument("--hidden_rna", type=int, default=185)
parser.add_argument("--hidden_adt", type=int, default=30)   # for CITEseq
parser.add_argument("--hidden_atac", type=int, default=185) # for SHAREseq

# optional explicit label path (still not in 8 args)
parser.add_argument("--cty", type=str, default="NULL")

args = parser.parse_args()

# -------------------------
# Setup
# -------------------------
setup_seed(args.seed)
cuda = True if (args.device == "cuda" and torch.cuda.is_available()) else False
device = torch.device("cuda" if cuda else "cpu")

os.makedirs(args.save_path, exist_ok=True)

begin_time = time.time()
mode = infer_mode(args.atac_file_path, args.adt_file_path)
print("Mode =", mode)

# -------------------------
# Determine label source
# -------------------------
cty_path = None
cty_used = None
if (args.cty is not None) and (not is_null_path(args.cty)) and os.path.exists(args.cty):
    cty_path = args.cty
    cty_used = args.cty
else:
    auto_label, auto_path = maybe_load_cty_auto(args.rna_file_path, args.save_path, args.device)
    if auto_path is not None:
        cty_path = auto_path
        cty_used = auto_path

# -------------------------
# Build TRAIN data & loader
# -------------------------
train_data, train_label, train_dl, eval_dl, mode2, nfeatures_rna, nfeatures_mod = build_data_and_dl(
    args.rna_file_path, args.atac_file_path, args.adt_file_path,
    cty_path, args.batch_size, args.device, args.hvg_num
)
assert mode2 == mode, "Internal mode mismatch"

# classify_dim
if torch.is_tensor(train_label):
    classify_dim = int((train_label.max() + 1).detach().cpu().numpy()) if train_label.numel() > 0 else 1
else:
    classify_dim = 1
if classify_dim <= 0:
    classify_dim = 1

# -------------------------
# Train (stage-1)
# -------------------------
model_save_path = os.path.join("..", "trained_model", mode)
os.makedirs(model_save_path, exist_ok=True)

hidden_mod = args.hidden_adt if mode == "CITEseq" else args.hidden_atac
model = build_model(mode, nfeatures_rna, nfeatures_mod, args.hidden_rna, hidden_mod, args.z_dim, classify_dim, device)

test_dl = "NULL"
model, acc1, num1, train_num = train_model(
    model, train_dl, test_dl,
    lr=args.lr, epochs=args.epochs,
    classify_dim=classify_dim,
    best_top1_acc=0,
    save_path=model_save_path,
    feature_num=train_data.shape[1],
    device=device,
)

# -------------------------
# Optional augmentation (only meaningful when classify_dim>1)
# -------------------------
if args.augmentation and classify_dim > 1:
    stage1_df = pd.DataFrame([[i, train_num[i]] for i in range(classify_dim)])
    if classify_dim % 2 == 0:
        train_median = np.sort(train_num)[int(classify_dim / 2) - 1]
    else:
        train_median = np.median(train_num)

    median_anchor = stage1_df[stage1_df[1] == train_median][0]
    train_major = stage1_df[stage1_df[1] > train_median]
    train_minor = stage1_df[stage1_df[1] < train_median]

    if len(train_minor) > 0 and float(train_median) > 0:
        anchor_fold = np.array((train_median) / (train_minor[:][1]))
        minor_anchor_cts = train_minor[0].to_numpy()
        major_anchor_cts = train_major[0].to_numpy()

        # anchor (median)
        idx = (train_label == int(np.array(median_anchor)[0])).nonzero(as_tuple=True)[0]
        new_data = train_data[idx.tolist(), :]
        new_label = train_label[idx.tolist()]

        # downsample major
        j = 0
        for anchor in major_anchor_cts:
            anchor_num = int(np.array(train_major[1])[j])
            ds_index = random.sample(list(range(anchor_num)), int(train_median))
            idx = (train_label == anchor).nonzero(as_tuple=True)[0]
            a_data = train_data[idx.tolist(), :][ds_index, :]
            a_label = train_label[idx.tolist()][ds_index]
            new_data = torch.cat((new_data, a_data), 0)
            new_label = torch.cat((new_label, a_label.to(device)), 0)
            j += 1

        # augment minor
        j = 0
        for anchor in minor_anchor_cts:
            aug_fold = int(anchor_fold[j])
            remaining_cell = int(train_median - aug_fold * int(np.array(train_minor[1])[j]))

            idx = (train_label == anchor).nonzero(as_tuple=True)[0]
            a_data = train_data[idx.tolist(), :]
            a_label = train_label[idx.tolist()]

            a_ds = MyDataset(a_data, a_label)
            a_dl = DataLoader(a_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

            for _ in range(max(aug_fold, 1)):
                rec_data, rec_label, real_data = get_vae_simulated_data_from_sampling(model, a_dl, args.device)
                rec_data[rec_data > torch.max(real_data)] = torch.max(real_data)
                rec_data[rec_data < torch.min(real_data)] = torch.min(real_data)
                rec_data[torch.isnan(rec_data)] = torch.max(real_data)
                new_data = torch.cat((new_data, rec_data), 0)
                new_label = torch.cat((new_label.to(device), rec_label.to(device)), 0)

            if remaining_cell > 0:
                rec_data, rec_label, real_data = get_vae_simulated_data_from_sampling(model, a_dl, args.device)
                rec_data[rec_data > torch.max(real_data)] = torch.max(real_data)
                rec_data[rec_data < torch.min(real_data)] = torch.min(real_data)
                rec_data[torch.isnan(rec_data)] = torch.max(real_data)

                ds_index = random.sample(list(range(int(rec_data.size(0)))), remaining_cell)
                rec_data = rec_data[ds_index, :]
                rec_label = rec_label[ds_index]
                new_data = torch.cat((new_data, rec_data), 0)
                new_label = torch.cat((new_label.to(device), rec_label.to(device)), 0)

            j += 1

        # stage-2 re-train on augmented
        model = build_model(mode, nfeatures_rna, nfeatures_mod, args.hidden_rna, hidden_mod, args.z_dim, classify_dim, device)
        aug_ds = MyDataset(new_data, new_label)
        aug_dl = DataLoader(aug_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

        model, acc2, num1, train_num = train_model(
            model, aug_dl, test_dl,
            lr=args.lr, epochs=int(args.epochs / 2),
            classify_dim=classify_dim,
            best_top1_acc=0,
            save_path=model_save_path,
            feature_num=new_data.shape[1],
            device=device,
        )

        # load best then fine-tune
        loaded = load_best_checkpoint_if_any(model, model_save_path, device)
        model, acc2, num1, train_num = train_model(
            model, aug_dl, test_dl,
            lr=args.lr / 10, epochs=int(args.epochs / 2),
            classify_dim=classify_dim,
            best_top1_acc=0,
            save_path=model_save_path,
            feature_num=new_data.shape[1],
            device=device,
        )
    else:
        print("[Info] augmentation requested but cannot compute minority/median properly; skip augmentation.")
else:
    if args.augmentation and classify_dim <= 1:
        print("[Info] augmentation=True but classify_dim<=1 (no labels). Skip augmentation.")

# -------------------------
# Load best checkpoint for embedding export (recommended)
# -------------------------
loaded_ckpt = load_best_checkpoint_if_any(model, model_save_path, device)
if loaded_ckpt:
    print("[Info] Loaded best checkpoint for embedding export.")

model.eval()

# -------------------------
# Export TRAIN embedding (CSV)
# -------------------------
latent_train, _, _ = get_encodings(model, eval_dl, args.device)
latent_train = standardize_latent_to_cells_by_dim(latent_train)

barcodes_train = load_barcodes_if_any(args.rna_file_path)
df_train = pd.DataFrame(latent_train)
if barcodes_train is not None and len(barcodes_train) == df_train.shape[0]:
    df_train.insert(0, "barcode", barcodes_train)

out_train_csv = os.path.join(args.save_path, f"{args.save_key}_latent.csv")
df_train.to_csv(out_train_csv, index=False)
print(f"[Done] Saved train latent: {out_train_csv}")

rna = h5_to_h5ad(args.rna_file_path)
rna.obsm["latent"] = np.array(latent_train)
sc.pp.neighbors(rna, use_rep="latent")
sc.tl.umap(rna)
res = search_resolution(rna, fixed_clus_count=args.n_cluster)
sc.tl.leiden(rna, resolution=res, key_added="cluster")

## save UMAP
umap = pd.DataFrame(rna.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=rna.obs_names)
umap.insert(2, "cluster", rna.obs['cluster'].values)
umap.to_csv(os.path.join(args.save_path, args.save_key + ".csv"))

# # -------------------------
# # Export TEST embedding (CSV) if auto-detected
# # -------------------------
# test_rna, test_atac, test_adt = auto_find_test_files(args.rna_file_path, args.atac_file_path, args.adt_file_path)
# out_test_csv = None
# if test_rna is not None:
#     test_data, test_label, test_dl2, test_dl3, mode3, _, _ = build_data_and_dl(
#         test_rna, test_atac, test_adt, cty_path, args.batch_size, args.device
#     )
#     latent_test, _, _ = get_encodings(model, test_dl2, args.device)
#     latent_test = standardize_latent_to_cells_by_dim(latent_test)

#     barcodes_test = load_barcodes_if_any(test_rna)
#     df_test = pd.DataFrame(latent_test)
#     if barcodes_test is not None and len(barcodes_test) == df_test.shape[0]:
#         df_test.insert(0, "barcode", barcodes_test)

#     out_test_csv = os.path.join(args.save_path, f"{args.save_key}_test_matilda_latent.csv")
#     df_test.to_csv(out_test_csv, index=False)
#     print(f"[Done] Saved test latent: {out_test_csv}")
# else:
#     print("[Info] No test files detected (need test_rna.h5 + paired modality). Skip test export.")

# -------------------------
# Save meta/time/classify_dim (CSV)
# -------------------------
# elapsed = time.time() - begin_time

# pd.DataFrame({"time_sec": [elapsed]}).to_csv(os.path.join(args.save_path, f"{args.save_key}_time.csv"), index=False)
# pd.DataFrame({"classify_dim": [int(classify_dim)]}).to_csv(os.path.join(args.save_path, f"{args.save_key}_classify_dim.csv"), index=False)

# meta = {
#     "mode": mode,
#     "rna_file_path": args.rna_file_path,
#     "atac_file_path": args.atac_file_path,
#     "adt_file_path": args.adt_file_path,
#     "save_path": args.save_path,
#     "save_key": args.save_key,
#     "n_cluster": int(args.n_cluster),
#     "hvg_num": int(args.hvg_num),
#     "batch_key": str(args.batch_key),
#     "seed": int(args.seed),
#     "device": str(args.device),
#     "batch_size": int(args.batch_size),
#     "epochs": int(args.epochs),
#     "lr": float(args.lr),
#     "z_dim": int(args.z_dim),
#     "hidden_rna": int(args.hidden_rna),
#     "hidden_mod": int(args.hidden_adt if mode == "CITEseq" else args.hidden_atac),
#     "feature_num": int(train_data.shape[1]),
#     "n_cells_train": int(train_data.shape[0]),
#     "classify_dim": int(classify_dim),
#     "cty_used": cty_used if cty_used is not None else "NULL",
#     "checkpoint_loaded_for_export": bool(loaded_ckpt),
#     "train_latent_csv": out_train_csv,
#     "test_latent_csv": out_test_csv if out_test_csv is not None else "NULL",
#     "time_sec": float(elapsed),
# }
# pd.DataFrame([meta]).to_csv(os.path.join(args.save_path, f"{args.save_key}_matilda_meta.csv"), index=False)

# print(f"[Done] All finished. time={elapsed:.2f}s | mode={mode} | classify_dim={classify_dim}")