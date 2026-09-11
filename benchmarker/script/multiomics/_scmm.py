#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scMM runner (RNA+ADT or RNA+ATAC) with YOUR 8 positional args and H5 inputs.

Usage (8 positional args):
  python main_scMM_cli.py \
      <n_cluster> <rna_file_path> <atac_file_path> <adt_file_path> \
      <save_path> <save_key> <hvg_num> <batch_key>

Rules:
- atac_file_path and adt_file_path: exactly ONE must be "NULL" (string, case-insensitive)
  - RNA + ATAC:  adt_file_path == "NULL" and atac_file_path is a valid .h5
  - RNA + ADT:   atac_file_path == "NULL" and adt_file_path is a valid .h5

Notes:
- n_cluster / batch_key are accepted for compatibility and will be saved to metadata,
  but scMM itself does not use them in this script.
- hvg_num: optional HVG selection for RNA only (variance-based on log1p, works even without batch labels).
"""

import os
import sys
import time
import json
import argparse
import datetime
from pathlib import Path
from tempfile import mkdtemp
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

import torch
from torch import optim

import scipy.sparse as sp
import scipy.sparse as sparse
from scipy.io import mmwrite

import anndata as ad
import scanpy as sc

import _scmm_models as models
import _scmm_obj as objectives
from utils import search_resolution
from torch.utils.data import Dataset
import scipy
from glob import glob
from scipy.io import mmread

class RNA_Dataset(Dataset):
    """
    Single-cell RNA/ADT dataset
    """

    def __init__(self, path, transpose=False):
        
        self.data, self.genes, self.barcode = load_data(path, transpose)
        self.indices = None
        self.n_cells, self.n_peaks = self.data.shape
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        if type(data) is not np.ndarray:
            data = data.toarray().squeeze()
        return torch.tensor(data)
    
    def info(self):
        print("\n===========================")
        print("Dataset Info")
        print('Cell number: {}\nGene number: {}'.format(self.n_cells, self.n_peaks))
        print('===========================\n')

def load_data(path, transpose=False):
    print("Loading  data ...")
    t0 = time.time()
    if os.path.isdir(path):
        count, peaks, barcode = read_mtx(path)
    elif os.path.isfile(path):
        count, peaks, barcode = read_csv(path)
    else:
        raise ValueError("File {} not exists".format(path))
        
    if transpose: 
        count = count.transpose()
    print('Original data contains {} cells x {} peaks'.format(*count.shape))
    assert (len(barcode), len(peaks)) == count.shape
    print("Finished loading takes {:.2f} min".format((time.time()-t0)/60))
    return count, peaks, barcode

def read_mtx(path):
    for filename in glob(path+'/*'):
        basename = os.path.basename(filename)
        if (('count' in basename) or ('matrix' in basename)) and ('mtx' in basename):
            count = mmread(filename).T.tocsr().astype('float32')
        elif 'barcode' in basename:
            if ('.txt' in basename) or ('tsv' in basename):
                sep = '\t'
            elif '.csv' in basename:
                sep = ','
            barcode = pd.read_csv(filename, sep=sep, header=None)[0].values
        elif 'gene' in basename or 'peak' in basename or 'protein' in basename:
            if ('.txt' in basename) or ('tsv' in basename):
                sep = '\t'
            elif '.csv' in basename:
                sep = ','
            feature = pd.read_csv(filename, sep=sep, header=None).iloc[:, -1].values

    return count, feature, barcode
    
def read_csv(path):
    if ('.txt' in path) or ('tsv' in path):
        sep = '\t'
    elif '.csv' in path:
        sep = ','
    else:
        raise ValueError("File {} not in format txt or csv".format(path))
    data = pd.read_csv(path, sep=sep, index_col=0).T.astype('float32')
    genes = data.columns.values
    barcode = data.index.values
    return scipy.sparse.csr_matrix(data.values), genes, barcode

class ATAC_Dataset(Dataset):
    """
    Single-cell ATAC dataset
    """

    def __init__(self, path, transpose=False):
        
        self.data, self.peaks, self.barcode = load_data(path, transpose)
        self.indices = None
        self.n_cells, self.n_peaks = self.data.shape
        self.shape = self.data.shape


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        if type(data) is not np.ndarray:
            data = data.toarray().squeeze()
        return torch.tensor(data)
    
    def info(self):
        print("\n===========================")
        print("Dataset Info")
        print('Cell number: {}\nPeak number: {}'.format(self.n_cells, self.n_peaks))
        print('===========================\n')

class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))
        
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, runPath):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, runPath)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, runPath) #runPath追加
            self.counter = 0

    def save_checkpoint(self, val_loss, model, runPath):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), 'checkpoint.pt')
        save_model(model, runPath + '/model.rar') #mmvaeより移植
        self.val_loss_min = val_loss
# -------------------------
# Args
# -------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="scMM (RNA+ADT / RNA+ATAC) CLI wrapper with H5 inputs")

    # 8 positional arguments (as requested)
    parser.add_argument("n_cluster", type=int)
    parser.add_argument("rna_file_path", type=str)
    parser.add_argument("atac_file_path", type=str)
    parser.add_argument("adt_file_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("save_key", type=str)
    parser.add_argument("hvg_num", type=int)
    parser.add_argument("batch_key", type=str)

    # Keep scMM hyperparams as optional flags (same defaults as original)
    parser.add_argument("--experiment", type=str, default="test", help="experiment name")
    parser.add_argument("--obj", type=str, default="m_elbo_naive_warmup", help="objective to use")
    parser.add_argument("--llik_scaling", type=float, default=1.0, help="likelihood scaling")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--latent_dim", type=int, default=10, help="latent dimensionality")
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of hidden layers")
    parser.add_argument("--r_hidden_dim", type=int, default=100, help="hidden dim for gene")
    parser.add_argument("--p_hidden_dim", type=int, default=20, help="hidden dim for protein/peak")
    parser.add_argument("--pre_trained", type=str, default="", help="path to pre-trained model (unused here)")
    parser.add_argument("--learn_prior", action="store_true", default=False, help="learn model prior parameters")
    parser.add_argument("--analytics", action="store_true", default=True, help="enable analytics/output embedding")
    parser.add_argument("--print_freq", type=int, default=0, help="print frequency")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disable CUDA")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--deterministic_warmup", type=int, default=50, help="deterministic warmup")

    return parser


# -------------------------
# Scenario inference
# -------------------------
def is_null_path(x: str) -> bool:
    return x is None or str(x).strip().upper() == "NULL"

def infer_model(atac_file_path: str, adt_file_path: str) -> str:
    atac_null = is_null_path(atac_file_path)
    adt_null = is_null_path(adt_file_path)

    if atac_null and (not adt_null):
        return "rna_protein"  # RNA + ADT
    if adt_null and (not atac_null):
        return "rna_atac"     # RNA + ATAC
    raise ValueError(
        f"Invalid inputs: atac_file_path={atac_file_path}, adt_file_path={adt_file_path}. "
        f"Exactly ONE of them must be 'NULL'."
    )


# -------------------------
# H5 -> AnnData (your function, with required imports)
# -------------------------
def h5_to_h5ad(h5_path: str) -> "ad.AnnData":
    def _as_str_array(x):
        arr = np.asarray(x)
        if arr.dtype.kind in ("S", "O"):
            return np.array(
                [v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in arr],
                dtype=str,
            )
        return arr.astype(str)

    with h5py.File(h5_path, "r") as f:
        if "matrix" not in f:
            raise ValueError("H5 file must contain group 'matrix'.")
        g = f["matrix"]

        # ---- sparse CSR case ----
        if all(k in g for k in ("data", "indices", "indptr", "shape")) and ("barcodes" in g) and ("features" in g):
            data = np.asarray(g["data"])
            indices = np.asarray(g["indices"])
            indptr = np.asarray(g["indptr"])
            shape = tuple(np.asarray(g["shape"]).tolist())  # (n_cells, n_features)

            barcodes = _as_str_array(g["barcodes"])
            features = _as_str_array(g["features"])

            X = sparse.csr_matrix((data, indices, indptr), shape=shape)

        # ---- dense case ----
        else:
            if not all(k in g for k in ("data", "barcodes", "features")):
                raise ValueError("Dense H5 format requires: matrix/data, matrix/barcodes, matrix/features.")

            X = np.asarray(g["data"])
            barcodes = _as_str_array(g["barcodes"])
            features = _as_str_array(g["features"])

            # If stored as features×cells -> convert to cells×features
            if X.shape[0] == len(features) and X.shape[1] == len(barcodes):
                X = X.T

            X = sparse.csr_matrix(X)

    adata = ad.AnnData(
        X=X,
        obs={"barcode": barcodes},
        var={"feature": features},
    )
    adata.obs_names = barcodes
    adata.var_names = features
    return adata


# -------------------------
# Optional HVG (variance-based) for RNA only
# -------------------------
def select_hvg_variance(adata: "ad.AnnData", hvg_num: int) -> "ad.AnnData":
    """
    Simple, dependency-light HVG selection:
    - log1p transform (on a copy)
    - compute per-gene variance
    - keep top hvg_num genes

    Works without batch info.
    """
    if hvg_num is None or hvg_num <= 0:
        return adata
    n_genes = adata.n_vars
    if hvg_num >= n_genes:
        return adata

    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()
        # log1p on sparse: operate on data
        X_log = X.copy()
        X_log.data = np.log1p(X_log.data)
        # variance per gene on sparse matrix:
        # var = E[x^2] - (E[x])^2
        mean = np.asarray(X_log.mean(axis=0)).ravel()
        mean_sq = np.asarray(X_log.multiply(X_log).mean(axis=0)).ravel()
        var = mean_sq - mean**2
    else:
        X_log = np.log1p(X.astype(np.float64))
        var = X_log.var(axis=0)

    top_idx = np.argsort(var)[::-1][:hvg_num]
    top_idx = np.sort(top_idx)  # keep original order
    return adata[:, top_idx].copy()


# -------------------------
# AnnData -> scMM directory (count.mtx + barcode.txt + gene/protein/peak.txt)
# -------------------------
def adata_to_scmm_dir(adata: "ad.AnnData", out_dir: str, feature_kind: str):
    """
    Creates:
      out_dir/count.mtx
      out_dir/barcode.txt
      out_dir/gene.txt (if feature_kind='gene')
      out_dir/protein.txt + peak.txt (if feature_kind!='gene')

    We write mtx as features×cells (10X style), i.e. adata.X.T.
    """
    os.makedirs(out_dir, exist_ok=True)

    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    # features × cells
    mtx = X.T.tocoo()
    mmwrite(os.path.join(out_dir, "count.mtx"), mtx)

    barcodes = np.asarray(adata.obs_names).astype(str)
    feats = np.asarray(adata.var_names).astype(str)

    np.savetxt(os.path.join(out_dir, "barcode.txt"), barcodes, fmt="%s")
    if feature_kind == "gene":
        np.savetxt(os.path.join(out_dir, "gene.txt"), feats, fmt="%s")
    else:
        np.savetxt(os.path.join(out_dir, "protein.txt"), feats, fmt="%s")
        # For ATAC_Dataset variants that expect "peak.txt"
        np.savetxt(os.path.join(out_dir, "peak.txt"), feats, fmt="%s")


# -------------------------
# Training / Testing
# -------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    # infer scenario/model
    args.model = infer_model(args.atac_file_path, args.adt_file_path)
    if args.model == "rna_atac":
        modal_name = "ATAC-seq"
        modal_h5 = args.atac_file_path
    else:
        modal_name = "CITE-seq"
        modal_h5 = args.adt_file_path

    # seed & device
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # run path
    runId = datetime.datetime.now().isoformat()
    experiment_dir = Path("../experiments") / args.experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
    print(runPath)

    # -------------------------
    # Prepare scMM input dirs from H5 (using your reader)
    # -------------------------
    rna_h5 = args.rna_file_path
    if not os.path.exists(rna_h5):
        raise FileNotFoundError(f"RNA h5 not found: {rna_h5}")
    if not is_null_path(modal_h5) and (not os.path.exists(modal_h5)):
        raise FileNotFoundError(f"Modality h5 not found: {modal_h5}")

    # Make temp folders next to source files (same style as your original)
    rna_dir_base = os.path.join(os.path.dirname(rna_h5), "scMM_data", "rna0")
    mod_dir_base = os.path.join(os.path.dirname(modal_h5), "scMM_data", "another_modality0")

    # Read -> AnnData
    adata_rna = h5_to_h5ad(rna_h5)
    # Optional HVG for RNA only
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=args.hvg_num, subset=True)

    adata_mod = h5_to_h5ad(modal_h5)
    if modal_name=="ATAC-seq":
        sc.pp.highly_variable_genes(adata_mod, flavor="seurat_v3", n_top_genes=args.hvg_num*10, subset=True)


    # Write to scMM dirs
    adata_to_scmm_dir(adata_rna, rna_dir_base, feature_kind="gene")
    # ATAC/ADT both written as protein/peak style (compatible with your original script)
    adata_to_scmm_dir(adata_mod, mod_dir_base, feature_kind="protein")

    rna_path_list = [rna_dir_base]
    modal_path_list = [mod_dir_base]

    # -------------------------
    # Build datasets
    # -------------------------
    r_dataset_list = []
    for rp in rna_path_list:
        r_dataset = RNA_Dataset(rp)
        args.r_dim = r_dataset.data.shape[1]
        r_dataset_list.append(r_dataset)

    modal_dataset_list = []
    for mp in modal_path_list:
        modal_dataset = ATAC_Dataset(mp) if args.model == "rna_atac" else RNA_Dataset(mp)
        args.p_dim = modal_dataset.data.shape[1]
        modal_dataset_list.append(modal_dataset)

    train_dataset = [r_dataset_list[0], modal_dataset_list[0]]
    test_dataset_list = []  # your CLI inputs only provide one pair; keep empty

    # -------------------------
    # Load model
    # -------------------------
    modelC = getattr(models, f"VAE_{args.model}")
    print(args)

    model = modelC(args).to(device)

    # Save args snapshot (json for readability)
    args_dump = vars(args).copy()
    args_dump["modal_name"] = modal_name
    args_dump["runPath"] = runPath
    with open(os.path.join(runPath, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dump, f, indent=2, ensure_ascii=False)

    # Dataloaders
    train_loader = model.getDataLoaders(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, device=device
    )
    test_loader_list = []
    for test_dataset in test_dataset_list:
        test_loader_list.append(
            model.getDataLoaders(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)
        )

    # Optimizer & objective
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True)
    objective = getattr(objectives, args.obj)
    s_objective = getattr(objectives, args.obj)

    def train_epoch(epoch, agg, W):
        model.train()
        b_loss = 0.0
        for i, dataT in enumerate(train_loader):
            beta = (epoch - 1) / W if epoch <= W else 1.0
            if dataT[0].size()[0] == 1:
                continue
            data = [d.to(device) for d in dataT]
            optimizer.zero_grad()
            loss = -objective(model, data, beta)
            loss.backward()
            optimizer.step()
            b_loss += float(loss.item())
            if args.print_freq > 0 and i % args.print_freq == 0:
                print(f"iteration {i:04d}: loss: {loss.item() / args.batch_size:6.3f}")
        agg["train_loss"].append(b_loss / len(train_loader.dataset))
        print(f"====> Epoch: {epoch:03d} Train loss: {agg['train_loss'][-1]:.4f}")
        return b_loss

    def test_epoch(epoch, agg, W, test_loader):
        model.eval()
        b_loss = 0.0
        with torch.no_grad():
            for i, dataT in enumerate(test_loader):
                beta = (epoch - 1) / W if epoch <= W else 1.0
                if dataT[0].size()[0] == 1:
                    continue
                data = [d.to(device) for d in dataT]
                loss = -s_objective(model, data, beta)
                b_loss += float(loss.item())
        agg["test_loss"].append(b_loss / len(test_loader.dataset))
        print(f"====>             Test loss: {agg['test_loss'][-1]:.4f}")

    # -------------------------
    # Train
    # -------------------------
    with Timer("MM-VAE") as _t:
        agg = defaultdict(list)
        early_stopping = EarlyStopping(patience=10, verbose=True)
        W = args.deterministic_warmup

        for epoch in range(1, args.epochs + 1):
            b_loss = train_epoch(epoch, agg, W)
            if torch.isnan(torch.tensor([b_loss])):
                print("NaN loss detected, stopping.")
                break

            # optional: run tests if provided
            for tl in test_loader_list:
                test_epoch(epoch, agg, W, tl)

    def get_latent(dataloader, train_test, runPath):
        model.eval()
        with torch.no_grad():
            modal_tags = ["rna", "atac"] if args.model == "rna_atac" else ["rna", "protein"]
            pred = None

            for i, dataT in enumerate(dataloader):
                data = [d.to(device) for d in dataT]
                lats = model.latents(data, sampling=False)  # list of tensors, each: (bs, latent_dim)

                if pred is None:
                    pred = [lat.detach().clone() for lat in lats]
                else:
                    for m, lat in enumerate(lats):
                        pred[m] = torch.cat([pred[m], lat.detach()], dim=0)

            # save each modality latent
            for m, lat in enumerate(pred):
                lat_np = lat.cpu().numpy()
                pd.DataFrame(lat_np).to_csv(
                    os.path.join(runPath, f"lat_{train_test}_{modal_tags[m]}.csv"),
                    index=False
                )

            # save mean latent
            mean_lats = sum(pred) / len(pred)
            mean_np = mean_lats.cpu().numpy()
            pd.DataFrame(mean_np).to_csv(
                os.path.join(runPath, f"lat_{train_test}_mean.csv"),
                index=False
            )
            return mean_np

    train_loader_eval = model.getDataLoaders(
        train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device
    )

    train_result = get_latent(train_loader_eval, "train", runPath)
    adata_rna.obsm["latent"] = train_result.copy()
    sc.pp.neighbors(adata_rna, use_rep="latent")
    sc.tl.umap(adata_rna)
    res = search_resolution(adata_rna, fixed_clus_count=args.n_cluster)
    sc.tl.leiden(adata_rna, resolution=res, key_added="cluster")

    ## save UMAP
    umap = pd.DataFrame(adata_rna.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata_rna.obs_names)
    umap.insert(2, "cluster", adata_rna.obs['cluster'].values)
    umap.to_csv(os.path.join(args.save_path, args.save_key + ".csv"))
    latent = pd.DataFrame(train_result, index=adata_rna.obs_names)
    latent.to_csv(os.path.join(args.save_path, args.save_key + "_latent.csv"))


if __name__ == "__main__":
    main()