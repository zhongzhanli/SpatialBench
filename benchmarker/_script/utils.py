import numpy as np
import scanpy as sc
import pandas as pd
import sklearn
from typing import Literal
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

def mclust(input, n_cluster):
    np.random.seed(0)
    robjects.r.library("mclust")
    np_cv_rules = default_converter + numpy2ri.converter

    r_random_seed = robjects.r['set.seed']
    r_random_seed(0)
    rmclust = robjects.r['Mclust']

    with np_cv_rules.context():
        res = rmclust(np.array(input), n_cluster, "EEE")

    mclust_res = np.array(res[-2])
    return mclust_res

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

def split_adata(adata, batch_key="batch", ):
    adata_list = []
    seen = set()
    batch_list = []
    for item in list(adata.obs[batch_key]):
        if item not in seen:
            seen.add(item)
            batch_list.append(item)
    for i in batch_list:
        index = adata.obs[batch_key]==i
        t_ad = adata[index,]
        adata_list.append(t_ad)
    return adata_list

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

def transform_coord(coords_list, axis: Literal["x", "y"] = "x", margin_size: float = 0.1,
                    angle=None, horizontal=False, vertical=False,
                    align_axis: Literal["x", "y", None] = None):

    transformed_coord = []
    axis_idx = 0 if axis == "x" else 1

    for i, coords in enumerate(coords_list):
        coords = coords.copy()

        # Step 1: rotation / mirror
        if horizontal:
            coords = coords @ np.array([[-1, 0], [0, 1]])
        if vertical:
            coords = coords @ np.array([[1, 0], [0, -1]])
        if angle is not None:
            theta = np.deg2rad(angle)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            coords = coords @ rotation_matrix

        # Step 2: translation to avoid overlap
        if i == 0:
            transformed = coords
        else:
            prev_coords = transformed_coord[-1]
            prev_max = np.max(prev_coords[:, axis_idx])
            curr_min = np.min(coords[:, axis_idx])
            curr_max = np.max(coords[:, axis_idx])
            margin = margin_size * (curr_max - curr_min)
            offset = prev_max - curr_min + margin
            offset_vec = np.array([offset, 0]) if axis_idx == 0 else np.array([0, offset])
            transformed = coords + offset_vec

        transformed_coord.append(transformed)

    # Step 3: align the orthogonal direction
    if align_axis is None:
        align_axis = "y" if axis == "x" else "x"
    align_idx = 0 if align_axis == "x" else 1
    global_min = min(np.min(coords[:, align_idx]) for coords in transformed_coord)

    for i in range(len(transformed_coord)):
        curr_min = np.min(transformed_coord[i][:, align_idx])
        offset = global_min - curr_min
        offset_vec = np.array([offset, 0]) if align_idx == 0 else np.array([0, offset])
        transformed_coord[i] += offset_vec

    return transformed_coord

def list_to_onehot(input_list):
   
    unique_classes = list(dict.fromkeys(input_list))
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    onehot_matrix = np.zeros((len(input_list), len(unique_classes)), dtype=int)
    
    for i, cls in enumerate(input_list):
        onehot_matrix[i, class_to_index[cls]] = 1
    
    return onehot_matrix, class_to_index
