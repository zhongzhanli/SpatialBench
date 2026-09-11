import os
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import cellcharter as cc
import scvi
import torch
import squidpy as sq

from utils import h5_to_h5ad, search_resolution


def parse_args():
    parser = argparse.ArgumentParser("SpatialGlue unified runner")

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


def infer_mode_and_datatype(atac_file_path, adt_file_path):
    atac_is_null = is_null_string(atac_file_path)
    adt_is_null = is_null_string(adt_file_path)

    if (not atac_is_null) and adt_is_null:
        return "atac", "Spatial-ATAC-RNA"
    elif atac_is_null and (not adt_is_null):
        return "adt", "Stereo-CITE-seq"
    else:
        raise ValueError(
            "Invalid input."
        )


def get_hvg_kwargs(hvg_num, batch_key):
    kwargs = dict(flavor="seurat_v3", n_top_genes=hvg_num)
    # if not is_null_string(batch_key):
    #     kwargs["batch_key"] = batch_key
    return kwargs


def preprocess_rna(adata_rna, hvg_num, batch_key):
    adata_rna.layers["counts"] = adata_rna.X.copy()
    hvg_kwargs = get_hvg_kwargs(hvg_num, batch_key)

    sc.pp.highly_variable_genes(adata_rna, **hvg_kwargs)
    # sc.pp.normalize_total(adata_rna, target_sum=1e4)
    # sc.pp.log1p(adata_rna)
    # sc.pp.scale(adata_rna)

    adata_rna_high = adata_rna[:, adata_rna.var["highly_variable"]].copy()

    return adata_rna_high


def run_cellcharter(
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

    mode, data_type = infer_mode_and_datatype(atac_file_path, adt_file_path)
    print(f"[Info] inferred mode = {mode}")
    print(f"[Info] inferred data_type = {data_type}")

    print("[1/5] Loading data ...")
    adata_rna = h5_to_h5ad(rna_file_path, batch_key=batch_key)

    if adata_rna.shape[0] > 5000 and mode=="atac":
        data_type = 'Spatial-epigenome-transcriptome'

    adata_rna.var_names_make_unique()

    if mode == "atac":
        adata_omics2 = h5_to_h5ad(atac_file_path, batch_key=batch_key)
    else:
        adata_omics2 = h5_to_h5ad(adt_file_path, batch_key=batch_key)
    adata_omics2.var_names_make_unique()

    print(f"RNA shape: {adata_rna.shape}")
    print(f"Omics2 shape: {adata_omics2.shape}")

    print("[2/5] Preprocessing ...")
    if mode == "atac":
        adata_rna = preprocess_rna(
            adata_rna=adata_rna,
            hvg_num=hvg_num,
            batch_key=batch_key
        )
        adata_omics2 = preprocess_rna(
            adata_rna=adata_omics2,
            hvg_num=hvg_num*10,
            batch_key=batch_key
        )

    else:
        adata_rna = preprocess_rna(
            adata_rna=adata_rna,
            hvg_num=hvg_num,
            batch_key=batch_key
        )
        adata_omics2.layers["counts"] = adata_omics2.X.copy()

    print("RNA feat shape:", adata_rna.X.shape)
    print("Omics2 feat shape:", adata_omics2.X.shape)

   
    print("[3/5] Training CellCharter ...")
    scvi.model.SCVI.setup_anndata(adata_rna, layer="counts",
     batch_key=batch_key if batch_key in adata_rna.obs else None)
    model = scvi.model.SCVI(adata_rna, n_layers=2, n_latent=30, gene_likelihood="nb")
    model.train()
    adata_rna.obsm["latent"] = model.get_latent_representation()

    if mode == "atac":
        scvi.model.SCVI.setup_anndata(adata_omics2, layer="counts",
         batch_key=batch_key if batch_key in adata_omics2.obs else None)
        model = scvi.model.SCVI(adata_omics2, n_layers=2, n_latent=30, gene_likelihood="poisson")
        model.train()
        adata_omics2.obsm["latent"] = model.get_latent_representation()
    else:
        adata_omics2.obs["condition"] = "adt"
        adata_omics2.X = adata_omics2.X.astype(np.float32)
        adata_omics2.layers["counts"] = adata_omics2.X.astype(np.float32).copy()
        scvi.model.SCVI.setup_anndata(adata_omics2, layer="counts",)# batch_key="batch")
        model = cc.tl.TRVAE(adata_omics2,
        condition_key= "condition" if batch_key not in adata_omics2.obs else batch_key)
        model.train()
        adata_omics2.obsm["latent"] = model.get_latent(torch.from_numpy(adata_omics2.X.toarray()),
        adata_omics2.obs['condition'] if batch_key not in adata_omics2.obs else [list(adata_omics2.obs[batch_key])[0]] * adata_omics2.shape[0])
    
    adata_out = adata_rna.copy()
    adata_out.obsm["latent"] = np.concatenate([adata_rna.obsm['latent'], adata_omics2.obsm['latent']], axis=1)
    sq.gr.spatial_neighbors(adata_out, coord_type='generic', delaunay=True)
    cc.gr.remove_long_links(adata_out)
    cc.gr.aggregate_neighbors(adata_out, n_layers=4 if mode == "atac" else 3, use_rep='latent')
    gmm = cc.tl.Cluster(
        n_clusters=n_cluster, 
        random_state=12345,
        trainer_params=dict(accelerator='gpu', devices=1)
    )
    gmm.fit(adata_out, use_rep='X_cellcharter')
    adata_out.obs['spatial_cluster'] = gmm.predict(adata_out, use_rep='X_cellcharter')

    print("[4/5] Saving embeddings ...")

    latent = pd.DataFrame(adata_out.obsm['X_cellcharter'], index=adata_out.obs_names)
    latent.to_csv(os.path.join(save_path, save_key + "_latent.csv"))
    
    print("[5/5] Clustering ...")
    

    sc.pp.neighbors(adata_out, n_neighbors=30, use_rep="X_cellcharter")
    sc.tl.umap(adata_out)

    ## save UMAP
    umap = pd.DataFrame(adata_out.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata_out.obs_names)
    umap.insert(2, "cluster", adata_out.obs['spatial_cluster'].values)
    umap.to_csv(os.path.join(save_path, save_key + ".csv"))


if __name__ == "__main__":
    args = parse_args()

    run_cellcharter(
        n_cluster=args.n_cluster,
        rna_file_path=args.rna_file_path,
        atac_file_path=args.atac_file_path,
        adt_file_path=args.adt_file_path,
        save_path=args.save_path,
        save_key=args.save_key,
        hvg_num=args.hvg_num,
        batch_key=args.batch_key
    )