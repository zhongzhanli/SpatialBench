import os
import time
import argparse
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import anndata as ad
import scvi
import scanpy as sc

from utils import search_resolution, h5_to_h5ad

def is_null_path(x: str) -> bool:
    return x is None or str(x).strip().upper() == "NULL"

def decode_h5_str(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr], dtype=str)
    return arr.astype(str)

def build_totalvi_anndata(rna_h5: str, adt_h5: str, batch_key: str, hvg_num: int):
    """
    Build AnnData suitable for scvi TOTALVI:
      - adata.X : RNA counts (csr)
      - adata.layers["counts"] : RNA counts copy
      - adata.obsm["protein_expression"] : ADT counts (dense np array or csr->dense)
      - adata.obs[batch_key] : batch labels (single batch here)
    """
    adata_rna = h5_to_h5ad(rna_h5, batch_key=batch_key)
    adata_adt = h5_to_h5ad(adt_h5, batch_key=batch_key)

    if adata_rna.shape[0] != adata_adt.shape[0]:
        raise ValueError(f"Cell number mismatch: RNA={adata_rna.shape[0]} vs ADT={adata_adt.shape[0]}")

    # if both have barcodes, try align/order-check
    if (adata_rna.obs_names is not None) and (adata_adt.obs_names is not None):
        if not np.array_equal(np.array(adata_rna.obs_names), np.array(adata_adt.obs_names)):
            # totalVI assumes matched cells; we warn loudly and proceed by row order
            print("[WARN] RNA/ADT barcodes not identical in order; proceeding by row order. "
                  "If results look wrong, you need to align cells before running totalVI.")

    adata = adata_rna.copy()
    sc.pp.highly_variable_genes(adata,
                                flavor="seurat_v3",
                                n_top_genes=hvg_num,
                                subset=True,
                                # batch_key=batch_key 
    )

    # counts layer required by TOTALVI
    adata.layers["counts"] = adata.X.copy()

    # protein expression stored in obsm: must be cells×proteins
    # scvi expects numpy array; csr ok in some versions but dense is safest
    adt_dense = adata_adt.X.toarray().astype(np.float32) if sp.issparse(adata_adt.X) else adata_adt.X.astype(np.float32)
    adata.obsm["protein_expression"] = adt_dense
    proteins = adata_adt.var_names

    # add protein names if available (optional but helpful)
    if proteins is not None:
        adata.uns["protein_names"] = proteins

    # single-batch label (since your 8-arg interface gives single file)
    if batch_key not in adata.obs:
        adata.obs[batch_key] = "batch0"

    return adata

def run_totalvi(adata, batch_key: str):
    scvi.settings.seed = 0
    print("Last run with scvi-tools version:", scvi.__version__)

    scvi.model.TOTALVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        layer="counts",
        batch_key=batch_key if batch_key in adata.obs else None,
    )

    vae = scvi.model.TOTALVI(
        adata,
        empirical_protein_background_prior=False,
        latent_distribution="normal",
    )
    vae.train()
    z = vae.get_latent_representation()  # n_cells × latent_dim
    return z


# -------------------------
# CLI (your 8 args)
# -------------------------
def main():
    parser = argparse.ArgumentParser("totalVI unified CLI (RNA+ADT only)")

    parser.add_argument("n_cluster", type=int)
    parser.add_argument("rna_file_path", type=str)
    parser.add_argument("atac_file_path", type=str)
    parser.add_argument("adt_file_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("save_key", type=str)
    parser.add_argument("hvg_num", type=int)
    parser.add_argument("batch_key", type=str)

    # optional knobs
    parser.add_argument("--device", type=str, default="auto")  # scvi handles device internally

    args = parser.parse_args()

    # --------- scenario check (totalVI only supports RNA+ADT) ----------
    if is_null_path(args.adt_file_path):
        raise ValueError("totalVI only supports RNA+ADT. You provided adt_file_path='NULL'.")
    if not is_null_path(args.atac_file_path):
        raise ValueError("totalVI only supports RNA+ADT. You provided atac_file_path != 'NULL' (RNA+ATAC not supported).")

    os.makedirs(args.save_path, exist_ok=True)

    t0 = time.time()
    adata = build_totalvi_anndata(
        rna_h5=args.rna_file_path,
        adt_h5=args.adt_file_path,
        batch_key=args.batch_key,
        hvg_num=args.hvg_num,
    )

    z = run_totalvi(adata, batch_key=args.batch_key)

    # ---------- save CSV (consistent with your other methods) ----------
    out_csv = os.path.join(args.save_path, f"{args.save_key}_latent.csv")
    df = pd.DataFrame(z)
    df.insert(0, "barcode", np.array(adata.obs_names).astype(str))
    df.to_csv(out_csv, index=False)

    adata.obsm["latent"] = z
    sc.pp.neighbors(adata, use_rep="latent")
    sc.tl.umap(adata)
    res = search_resolution(adata, fixed_clus_count=args.n_cluster)
    sc.tl.leiden(adata, resolution=res, key_added="cluster")

    ## save UMAP
    umap = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names)
    umap.insert(2, "cluster", adata.obs['cluster'].values)
    umap.to_csv(os.path.join(args.save_path, args.save_key + ".csv"))

    # elapsed = time.time() - t0
    # time_csv = os.path.join(args.save_path, f"{args.save_key}_time.csv")
    # pd.DataFrame({"time_sec": [elapsed]}).to_csv(time_csv, index=False)

    # print("[Done] latent:", out_csv)
    # print("[Done] time:", time_csv)
    # print(f"[Done] elapsed={elapsed:.2f}s | shape={df.shape}")


if __name__ == "__main__":
    main()