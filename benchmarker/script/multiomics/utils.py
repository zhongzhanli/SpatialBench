import numpy as np
import scanpy as sc
import pandas as pd
import sklearn
from typing import Literal, Optional
import h5py
import anndata as ad
from scipy import sparse
from scipy.sparse import csc_matrix

def search_resolution(adata, fixed_clus_count, increment=0.02):
    closest_count = np.inf  
    closest_res = None  
    
    for res in sorted(list(np.arange(0.1, 2, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res, key_added="temp_label")
        count_unique_leiden = len(list(set(adata.obs["temp_label"])))
        current_diff = abs(count_unique_leiden - fixed_clus_count)
        if current_diff < closest_count:
            closest_count = current_diff
            closest_res = res
        if count_unique_leiden == fixed_clus_count:
            break

    return closest_res

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
#     coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

import anndata as ad
import h5py
import numpy as np
from scipy import sparse


def h5_to_h5ad(h5_path: str, batch_key: Optional[str] = None):
    def _as_str_array(x):
        arr = np.asarray(x)
        if arr.dtype.kind in ("S", "O"):
            return np.array(
                [v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in arr],
                dtype=str
            )
        return arr.astype(str)

    with h5py.File(h5_path, "r") as f:
        if "matrix" not in f:
            raise ValueError("H5 file must contain group 'matrix'.")

        g = f["matrix"]

        spatial = None

        # ---- sparse case ----
        if all(k in g for k in ("data", "indices", "indptr", "shape")) and ("barcodes" in g) and ("features" in g):
            data = np.asarray(g["data"])
            indices = np.asarray(g["indices"])
            indptr = np.asarray(g["indptr"])
            shape = tuple(np.asarray(g["shape"]).tolist())  # (n_cells, n_genes)

            barcodes = _as_str_array(g["barcodes"])
            features = _as_str_array(g["features"])

            X = sparse.csr_matrix((data, indices, indptr), shape=shape)

        # ---- dense case ----
        else:
            if not all(k in g for k in ("data", "barcodes", "features")):
                raise ValueError("Dense H5 format requires datasets: matrix/data, matrix/barcodes, matrix/features.")

            X = np.asarray(g["data"])
            barcodes = _as_str_array(g["barcodes"])
            features = _as_str_array(g["features"])

            if X.shape[0] == len(features) and X.shape[1] == len(barcodes):
                X = X.T

            X = sparse.csr_matrix(X)

        # ---- read spatial if exists ----
        if "spatial" in g:
            spatial = np.asarray(g["spatial"], dtype=np.float32)

            if spatial.shape[0] != len(barcodes):
                raise ValueError(
                    f"spatial rows ({spatial.shape[0]}) do not match number of cells ({len(barcodes)})."
                )
        
        if batch_key is not None and batch_key in g:
            batches = _as_str_array(g[batch_key])
        else:
            batches = None

    # Build AnnData
    adata = ad.AnnData(
        X=X,
        obs={"barcode": barcodes},
        var={"feature": features},
    )
    adata.obs_names = barcodes
    adata.var_names = features

    # attach spatial
    if spatial is not None:
        adata.obsm["spatial"] = spatial
    if batches is not None:
        adata.obs[batch_key] = batches

    return adata
