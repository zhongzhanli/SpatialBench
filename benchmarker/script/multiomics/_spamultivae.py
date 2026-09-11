import sys
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/Benchmark/external/spaVAE/spaMultiVAE")

import os
from time import time
import argparse

import torch
import numpy as np
import scanpy as sc
import pandas as pd

from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from spaMultiVAE import SPAMULTIVAE
from preprocess import normalize, geneSelection

from utils import h5_to_h5ad, search_resolution


def read_any_adata(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith(".h5ad"):
        adata = sc.read_h5ad(path)
    elif path.endswith(".h5"):
        adata = h5_to_h5ad(path)
    else:
        raise ValueError(f"Unsupported file format: {path}. Only .h5ad and .h5 are supported.")

    if adata.X is None:
        raise ValueError(f"No expression matrix found in: {path}")

    if "spatial" not in adata.obsm:
        raise ValueError(f"'spatial' not found in adata.obsm for file: {path}")

    return adata


def to_dense_float64(X):
    if sparse.issparse(X):
        return X.toarray().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


def make_output_paths(save_path, save_key):
    os.makedirs(save_path, exist_ok=True)
    return {
        "model_file": os.path.join(save_path, f"{save_key}_model.pt"),
        "final_latent_file": os.path.join(save_path, f"{save_key}_final_latent.txt"),
        "gene_denoised_counts_file": os.path.join(save_path, f"{save_key}_gene_denoised_counts.txt"),
        "protein_denoised_counts_file": os.path.join(save_path, f"{save_key}_protein_denoised_counts.txt"),
        "protein_sigmoid_file": os.path.join(save_path, f"{save_key}_protein_sigmoid.txt"),
        "selected_genes_file": os.path.join(save_path, f"{save_key}_selected_genes.txt"),
        "selected_proteins_file": os.path.join(save_path, f"{save_key}_selected_proteins.txt"),
        "location_centroids_file": os.path.join(save_path, f"{save_key}_location_centroids.txt"),
        "location_kmeans_labels_file": os.path.join(save_path, f"{save_key}_location_kmeans_labels.txt"),
    }


if __name__ == "__main__":
    # ==========================================================
    # 只保留你要求的 8 个输入参数
    # ==========================================================
    parser = argparse.ArgumentParser(
        description="spaMultiVAE runner for RNA + ADT only",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("n_cluster", type=int)
    parser.add_argument("rna_file_path", type=str)
    parser.add_argument("atac_file_path", type=str)
    parser.add_argument("adt_file_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("save_key", type=str)
    parser.add_argument("hvg_num", type=int)
    parser.add_argument("batch_key", type=str)
    args = parser.parse_args()

    # ==========================================================
    # 固定默认参数（保持原方法默认）
    # ==========================================================
    batch_size = "auto"
    maxiter = 5000
    train_size = 0.95
    patience = 200
    lr = 5e-3
    weight_decay = 1e-6
    gene_noise = 0.0
    protein_noise = 0.0
    dropoutE = 0.0
    dropoutD = 0.0
    encoder_layers = [128, 64]
    GP_dim = 2
    Normal_dim = 18
    gene_decoder_layers = [128]
    protein_decoder_layers = [128]
    dynamicVAE = True
    init_beta = 10.0
    min_beta = 4.0
    max_beta = 25.0
    KL_loss = 0.025
    num_samples = 1
    fix_inducing_points = True
    grid_inducing_points = True
    inducing_point_steps = 10
    inducing_point_nums = None
    fixed_gp_params = False
    loc_range = 20.0
    kernel_scale = 20.0
    device = "cuda"

    # ==========================================================
    # 该方法只允许 RNA + ADT
    # ==========================================================
    if args.atac_file_path != "NULL":
        raise ValueError(
            "spaMultiVAE in this script only supports RNA + ADT integration. "
            "Therefore atac_file_path must be 'NULL'."
        )

    if args.adt_file_path == "NULL":
        raise ValueError(
            "spaMultiVAE in this script requires ADT input. "
            "Therefore adt_file_path cannot be 'NULL'."
        )

    out_files = make_output_paths(args.save_path, args.save_key)

    # ==========================================================
    # 读取 RNA / ADT
    # ==========================================================
    adata_rna = read_any_adata(args.rna_file_path)
    adata_adt = read_any_adata(args.adt_file_path)

    # cell number check
    if adata_rna.n_obs != adata_adt.n_obs:
        raise ValueError(
            f"Cell number mismatch: RNA has {adata_rna.n_obs} cells, "
            f"ADT has {adata_adt.n_obs} cells."
        )

    # spatial check
    spatial_rna = np.asarray(adata_rna.obsm["spatial"], dtype=np.float64)
    spatial_adt = np.asarray(adata_adt.obsm["spatial"], dtype=np.float64)

    if spatial_rna.shape != spatial_adt.shape:
        raise ValueError(
            f"Spatial shape mismatch: RNA spatial {spatial_rna.shape}, "
            f"ADT spatial {spatial_adt.shape}."
        )

    if not np.allclose(spatial_rna, spatial_adt):
        raise ValueError(
            "RNA and ADT spatial coordinates are not identical. "
            "This script expects matched RNA + ADT measured on the same spots/cells."
        )

    # expr matrices
    x1 = to_dense_float64(adata_rna.X)   # RNA
    x2 = to_dense_float64(adata_adt.X)   # ADT
    loc = spatial_rna.copy()

    # ==========================================================
    # batch_size auto
    # ==========================================================
    if batch_size == "auto":
        if x1.shape[0] <= 1024:
            batch_size = 128
        elif x1.shape[0] <= 2048:
            batch_size = 256
        else:
            batch_size = 512
    else:
        batch_size = int(batch_size)

    print("===== Input arguments =====")
    print(args)
    print("===== Fixed default parameters =====")
    print({
        "batch_size": batch_size,
        "maxiter": maxiter,
        "train_size": train_size,
        "patience": patience,
        "lr": lr,
        "weight_decay": weight_decay,
        "gene_noise": gene_noise,
        "protein_noise": protein_noise,
        "dropoutE": dropoutE,
        "dropoutD": dropoutD,
        "encoder_layers": encoder_layers,
        "GP_dim": GP_dim,
        "Normal_dim": Normal_dim,
        "gene_decoder_layers": gene_decoder_layers,
        "protein_decoder_layers": protein_decoder_layers,
        "dynamicVAE": dynamicVAE,
        "init_beta": init_beta,
        "min_beta": min_beta,
        "max_beta": max_beta,
        "KL_loss": KL_loss,
        "num_samples": num_samples,
        "fix_inducing_points": fix_inducing_points,
        "grid_inducing_points": grid_inducing_points,
        "inducing_point_steps": inducing_point_steps,
        "inducing_point_nums": inducing_point_nums,
        "fixed_gp_params": fixed_gp_params,
        "loc_range": loc_range,
        "kernel_scale": kernel_scale,
        "device": device,
        "n_cluster_unused": args.n_cluster,
        "batch_key_unused": args.batch_key,
    })

    # ==========================================================
    # 基因筛选：hvg_num 对应 select_genes
    # protein 不筛选，保持默认 0
    # ==========================================================
    if args.hvg_num > 0 and args.hvg_num < x1.shape[1]:
        importantGenes = geneSelection(x1, n=args.hvg_num, plot=False)
        x1 = x1[:, importantGenes]
        np.savetxt(out_files["selected_genes_file"], importantGenes, delimiter=",", fmt="%i")

    # 原默认 select_proteins = 0，不筛选
    select_proteins = 0
    if select_proteins > 0 and select_proteins < x2.shape[1]:
        importantProteins = geneSelection(x2, n=select_proteins, plot=False)
        x2 = x2[:, importantProteins]
        np.savetxt(out_files["selected_proteins_file"], importantProteins, delimiter=",", fmt="%i")

    # ==========================================================
    # spatial normalization
    # ==========================================================
    scaler = MinMaxScaler()
    loc = scaler.fit_transform(loc) * loc_range

    print("X_gene shape:", x1.shape)
    print("X_protein shape:", x2.shape)
    print("pos shape:", loc.shape)

    # ==========================================================
    # inducing points
    # ==========================================================
    if grid_inducing_points:
        if inducing_point_steps is None or inducing_point_steps <= 0:
            raise ValueError("inducing_point_steps must be a positive integer when grid_inducing_points=True.")
        eps = 1e-5
        initial_inducing_points = np.mgrid[
            0:(1 + eps):(1.0 / inducing_point_steps),
            0:(1 + eps):(1.0 / inducing_point_steps)
        ].reshape(2, -1).T * loc_range
        print("initial_inducing_points shape:", initial_inducing_points.shape)
    else:
        if inducing_point_nums is None or inducing_point_nums <= 0:
            raise ValueError("inducing_point_nums must be a positive integer when grid_inducing_points=False.")
        loc_kmeans = KMeans(n_clusters=inducing_point_nums, n_init=100).fit(loc)
        np.savetxt(out_files["location_centroids_file"], loc_kmeans.cluster_centers_, delimiter=",")
        np.savetxt(out_files["location_kmeans_labels_file"], loc_kmeans.labels_, delimiter=",", fmt="%i")
        initial_inducing_points = loc_kmeans.cluster_centers_

    # ==========================================================
    # normalize
    # ==========================================================
    adata1 = sc.AnnData(x1, dtype="float64")
    adata1 = normalize(
        adata1,
        size_factors=True,
        normalize_input=True,
        logtrans_input=True
    )

    adata2 = sc.AnnData(x2, dtype="float64")
    adata2 = normalize(
        adata2,
        size_factors=False,
        normalize_input=True,
        logtrans_input=True
    )

    adata2_no_scale = sc.AnnData(x2, dtype="float64")
    adata2_no_scale = normalize(
        adata2_no_scale,
        size_factors=False,
        normalize_input=False,
        logtrans_input=True
    )

    # ==========================================================
    # protein background prior
    # ==========================================================
    gm = GaussianMixture(n_components=2, covariance_type="diag", n_init=20).fit(adata2_no_scale.X)
    back_idx = np.argmin(gm.means_, axis=0)
    protein_log_back_mean = np.log(np.expm1(gm.means_[back_idx, np.arange(adata2_no_scale.n_vars)]))
    protein_log_back_scale = np.sqrt(gm.covariances_[back_idx, np.arange(adata2_no_scale.n_vars)])
    print("protein_back_mean shape:", protein_log_back_mean.shape)

    # ==========================================================
    # model
    # ==========================================================
    model = SPAMULTIVAE(
        gene_dim=adata1.n_vars,
        protein_dim=adata2.n_vars,
        GP_dim=GP_dim,
        Normal_dim=Normal_dim,
        encoder_layers=encoder_layers,
        gene_decoder_layers=gene_decoder_layers,
        protein_decoder_layers=protein_decoder_layers,
        gene_noise=gene_noise,
        protein_noise=protein_noise,
        encoder_dropout=dropoutE,
        decoder_dropout=dropoutD,
        fixed_inducing_points=fix_inducing_points,
        initial_inducing_points=initial_inducing_points,
        fixed_gp_params=fixed_gp_params,
        kernel_scale=kernel_scale,
        N_train=adata1.n_obs,
        KL_loss=KL_loss,
        dynamicVAE=dynamicVAE,
        init_beta=init_beta,
        min_beta=min_beta,
        max_beta=max_beta,
        protein_back_mean=protein_log_back_mean,
        protein_back_scale=protein_log_back_scale,
        dtype=torch.float64,
        device=device
    )

    print(model)

    # ==========================================================
    # train / load model
    # ==========================================================
    if not os.path.isfile(out_files["model_file"]):
        t0 = time()
        model.train_model(
            pos=loc,
            gene_ncounts=adata1.X,
            gene_raw_counts=adata1.raw.X,
            gene_size_factors=adata1.obs.size_factors,
            protein_ncounts=adata2.X,
            protein_raw_counts=adata2.raw.X,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_samples=num_samples,
            train_size=train_size,
            maxiter=maxiter,
            patience=patience,
            save_model=True,
            model_weights=out_files["model_file"]
        )
        print("Training time: %d seconds." % int(time() - t0))
    else:
        model.load_model(out_files["model_file"])

    # ==========================================================
    # outputs
    # ==========================================================
    final_latent = model.batching_latent_samples(
        X=loc,
        gene_Y=adata1.X,
        protein_Y=adata2.X,
        batch_size=batch_size
    )
    adata1.obsm["latent"] = np.array(final_latent)
    sc.pp.neighbors(adata1, use_rep="latent")
    sc.tl.umap(adata1)
    res = search_resolution(adata1, fixed_clus_count=args.n_cluster)
    sc.tl.leiden(adata1, resolution=res, key_added="cluster")

    ## save UMAP
    umap = pd.DataFrame(adata1.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata1.obs_names)
    umap.insert(2, "cluster", adata1.obs['cluster'].values)
    umap.to_csv(os.path.join(args.save_path, args.save_key + ".csv"))
    latent = pd.DataFrame(final_latent, index=adata1.obs_names)
    latent.to_csv(os.path.join(args.save_path, args.save_key + "_latent.csv"))
    # np.savetxt(out_files["final_latent_file"], final_latent, delimiter=",")

    # gene_denoised_counts, protein_denoised_counts, protein_sigmoid = model.batching_denoise_counts(
    #     X=loc,
    #     gene_Y=adata1.X,
    #     protein_Y=adata2.X,
    #     batch_size=batch_size,
    #     n_samples=25
    # )
    # np.savetxt(out_files["gene_denoised_counts_file"], gene_denoised_counts, delimiter=",")
    # np.savetxt(out_files["protein_denoised_counts_file"], protein_denoised_counts, delimiter=",")
    # np.savetxt(out_files["protein_sigmoid_file"], protein_sigmoid, delimiter=",")

    # print("Done.")
    # print("Outputs saved to:", args.save_path)