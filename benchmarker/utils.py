import numpy as np
import anndata as ad
import scanpy as sc
import math
import pandas as pd
from collections import defaultdict
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from typing import Callable, Literal
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler

def transform_coord(coords_list, axis: Literal["x", "y"] = "x", margin_size: float = 0.1,
                    angle=None, horizontal=False, vertical=False,
                    align_axis: Literal["x", "y", None] = None, align_mode: Literal["mid","min"] = "mid"):

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

        transformed_coord.append(transformed.astype(float))

    # Step 3: align the orthogonal direction
    if align_axis is None:
        align_axis = "y" if axis == "x" else "x"
    align_idx = 0 if align_axis == "x" else 1
    if(align_mode=="mid"):
        global_mid = np.mean([max(np.max(coords[:, align_idx]) for coords in transformed_coord), 
                            min(np.min(coords[:, align_idx]) for coords in transformed_coord)])

        for i in range(len(transformed_coord)):
            curr_mid = np.mean([max(transformed_coord[i][:, align_idx]), min(transformed_coord[i][:, align_idx])])
            offset = global_mid - curr_mid
            offset_vec = np.array([offset, 0]) if align_idx == 0 else np.array([0, offset])
            transformed_coord[i] += offset_vec
    else:
        global_min = min(np.min(coords[:, align_idx]) for coords in transformed_coord)
        for i in range(len(transformed_coord)):
            curr_min = np.min(transformed_coord[i][:, align_idx])
            offset = global_min - curr_min
            offset_vec = np.array([offset, 0]) if align_idx == 0 else np.array([0, offset])
            transformed_coord[i] += offset_vec



    return transformed_coord

def generate_test_data(n_cell=100, n_gene=100, batch_key="batch"):
    np.random.seed(0)
    mat1 = np.random.randn(n_cell, n_gene)
    mat1 = (mat1 * 10).astype(int)
    mat1[mat1<0] = 0
    mat2 = mat1.copy()
    # mat2 = np.random.randn(n_cell, n_gene)
    # mat2 = (mat2 * 10).astype(int)
    # mat2[mat2<0] = 0
    nx = int(n_cell / 5)
    ny = math.ceil(n_cell / nx)
    obsm = {"spatial": np.array([[i, j]  for j in range(ny) for i in range(nx)])[0:n_cell,:]}
    adata1 = ad.AnnData(X=mat1, obsm=obsm)
    adata2 = ad.AnnData(X=mat2, obsm=obsm)
    adata1.obs_names = [f"cell1_{i}" for i in range(n_cell)]
    adata2.obs_names = [f"cell2_{i}" for i in range(n_cell)]
    adata1.var_names = [f"gene{i}" for i in range(n_gene)]
    adata2.var_names = [f"gene{i}" for i in range(n_gene)]
    adata1.obs[batch_key] = "batch1"
    adata2.obs[batch_key] = "batch2"
    sc.pp.filter_cells(adata1, min_counts=1)
    sc.pp.filter_genes(adata1, min_counts=1)
    sc.pp.filter_cells(adata2, min_counts=1)
    sc.pp.filter_genes(adata2, min_counts=1)
    concat = ad.concat([adata1, adata2])
    concat.obs["spatial1"] = list(concat.obsm["spatial"][:,0])
    concat.obs["spatial2"] = list(concat.obsm["spatial"][:,1])
    return concat

def _shorten_list(items, preview=3):
    if not items:
        return "None"
    if len(items) <= preview:
        return ", ".join(items)
    return ", ".join(items[:preview]) + f", ... ({len(items) - preview} more)"

def prefix_reindex(df, short_index):

    prefix_groups = defaultdict(dict)
    index_list = list(df.index)

    for full_idx in index_list:
        for length in set(len(s) for s in short_index):
            prefix = full_idx[:length]
            prefix_groups[length][prefix] = full_idx

    final_idx = []
    for idx in short_index:
        found = False
        for length in prefix_groups:
            if idx in prefix_groups[length]:
                final_idx.append(prefix_groups[length][idx])
                found = True
                break
        if not found:
            raise KeyError(f"Prefix '{idx}' not found in df.index.")
        
    result = df.reindex(final_idx).copy()
    result.index = short_index
    return result

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

def _min_max_scale(df: pd.DataFrame):
    df = pd.DataFrame(
        MinMaxScaler().fit_transform(df),
        columns=df.columns,
        index=df.index,
        )
    return df

def normed_cmap(
    s: pd.Series, cmap: LinearSegmentedColormap, num_stds: float = 2.5
) -> Callable:
    """Returns a normalized colormap function that takes a float as an argument and
    returns an rgba value.

    Args:
        s (pd.Series):
            a series of numeric values
        cmap (matplotlib.colors.LinearSegmentedColormap):
            matplotlib Colormap
        num_stds (float, optional):
            vmin and vmax are set to the median ± num_stds.
            Defaults to 2.5.

    Returns:
        Callable: Callable that takes a float as an argument and returns an rgba value.
    """
    if(num_stds is not None): # colored by value
        _median = s.median()
        _std = s.std()
        vmin = _median - num_stds * _std
        vmax = _median + num_stds * _std
    else: # colored by rank
        s = pd.Series([i for i in s if i!=""])
        vmin = np.nanmin(s)
        vmax = np.nanmax(s)
    
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba

def apply_string_formatter(fmt: str, val: str) -> str:
    return fmt.format(val)

def apply_formatter(formatter: str , content: str) -> str:
    """Applies a formatter to the content.

    Args:
        formatter (str | Callable):
            the string formatter.
            Can either be a string format, ie "{:2f}" for 2 decimal places.
            Or a Callable that is applied to the content.
        content (str | Number):
            The content to format

    Raises:
        TypeError: when formatter is not of type str or Callable.

    Returns:
        str: a formatted string
    """

    if isinstance(formatter, str):
        return apply_string_formatter(formatter, content)
    elif isinstance(formatter, Callable):
        return formatter(content)
    else:
        raise TypeError("formatter needs to be either a `Callable` or a string.")

def truncate_colormap(cmap, minval=0.2, maxval=0.8, n=256):
    new_colors = cmap(np.linspace(minval, maxval, n))
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})', new_colors)
    return new_cmap

def get_default_cmap():
    cmap1 = truncate_colormap(matplotlib.cm.PRGn, 0.02, 0.95)
    cmap2 = truncate_colormap(matplotlib.cm.YlGnBu, 0.02, 0.95)
    return cmap1, cmap2

def repeat_to_length(lst, n, mode: Literal["tile","repeat"] = "tile"):
    if not len(lst):
        raise ValueError("Input list cannot be empty.")
    
    if mode == "tile":
        return (lst * (n // len(lst) + 1))[:n]
    elif mode == "repeat":
        return [item for item in lst for _ in range(n)]
    else:
        raise ValueError("Mode must be 'tile' or 'repeat'")

def get_scatter_cmap(lst):
    length = len(set(lst))
    seen = set()
    label_list = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            label_list.append(item)
    # label_list = sorted(list(set(lst)))
    
    if len(rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
        cc = rcParams["axes.prop_cycle"]()
        palette = [next(cc)["color"] for _ in range(length)]
    elif length <= 20:
        palette = sc.pl.palettes.default_20
    elif length <= 28:
        palette = sc.pl.palettes.default_28
    elif length <= len(sc.pl.palettes.default_102):  # 103 colors
        palette = sc.pl.palettes.default_102
    else:
        palette = ["grey" for _ in range(length)]
    return {label_list[i]: palette[i] for i in range(len(label_list))}

def map_to_integers(lst):
    unique_vals = sorted(set(lst))
    mapping_dict = {val: str(i) for i, val in enumerate(unique_vals)}
    mapped_list = [mapping_dict[val] for val in lst]
    return mapped_list
