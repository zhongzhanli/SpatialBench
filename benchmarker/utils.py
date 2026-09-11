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
import h5py
import numpy as np
from scipy.sparse import csr_matrix, issparse, csc_matrix
from scib_metrics.nearest_neighbors import NeighborsResults

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


def read_sparse_h5(file, group_name="matrix"):
    with h5py.File(file, "r") as f:
        i = f[f"{group_name}/i"][:]
        p = f[f"{group_name}/p"][:]
        x = f[f"{group_name}/x"][:]
        dim = tuple(f[f"{group_name}/dim"][:])

        rownames = f[f"{group_name}/rownames"][:].astype(str) if f"{group_name}/rownames" in f else None
        colnames = f[f"{group_name}/colnames"][:].astype(str) if f"{group_name}/colnames" in f else None

    mat = csc_matrix((x, i, p), shape=dim)
    return mat, rownames, colnames

def distmat_to_neighbors_results(
    distance_matrix,
    connection_matrix=None,
    sort_by_distance=True,
    validate=True,
):
    """
    Convert sparse distance/connection matrices into NeighborsResults.

    Parameters
    ----------
    distance_matrix : scipy sparse matrix
        Sparse matrix where [i, j] is the distance from cell i to neighbor j.
    connection_matrix : scipy sparse matrix or None
        Sparse binary adjacency matrix. Optional; mainly used for validation.
    sort_by_distance : bool
        Whether to sort each row's neighbors by ascending distance.
    validate : bool
        Whether to check consistency between distance and connection matrices.

    Returns
    -------
    NeighborsResults
    """
    if not issparse(distance_matrix):
        raise TypeError("distance_matrix must be a scipy sparse matrix")

    dist_csr = distance_matrix.tocsr()

    if connection_matrix is not None:
        if not issparse(connection_matrix):
            raise TypeError("connection_matrix must be a scipy sparse matrix")
        conn_csr = connection_matrix.tocsr()
    else:
        conn_csr = None

    n_samples = dist_csr.shape[0]

    indices_list = []
    distances_list = []

    neighbor_counts = []

    for i in range(n_samples):
        start, end = dist_csr.indptr[i], dist_csr.indptr[i + 1]
        neigh_idx = dist_csr.indices[start:end].copy()
        neigh_dist = dist_csr.data[start:end].copy()

        if conn_csr is not None and validate:
            cstart, cend = conn_csr.indptr[i], conn_csr.indptr[i + 1]
            conn_idx = conn_csr.indices[cstart:cend]

            if set(neigh_idx.tolist()) != set(conn_idx.tolist()):
                raise ValueError(
                    f"Row {i}: distance_matrix and connection_matrix have different neighbor sets"
                )

        if sort_by_distance:
            order = np.argsort(neigh_dist)
            neigh_idx = neigh_idx[order]
            neigh_dist = neigh_dist[order]

        indices_list.append(neigh_idx)
        distances_list.append(neigh_dist)
        neighbor_counts.append(len(neigh_idx))

    unique_counts = set(neighbor_counts)
    if len(unique_counts) != 1:
        raise ValueError(
            f"Rows have different numbers of neighbors: {sorted(unique_counts)}. "
            "Cannot form a rectangular NeighborsResults directly."
        )

    k = neighbor_counts[0]
    indices = np.empty((n_samples, k), dtype=np.int64)
    distances = np.empty((n_samples, k), dtype=dist_csr.data.dtype)

    for i in range(n_samples):
        indices[i, :] = indices_list[i]
        distances[i, :] = distances_list[i]

    return NeighborsResults(indices=indices, distances=distances)

import pandas as pd
import numpy as np
import re
from typing import Optional, Dict, List


import pandas as pd
import numpy as np
import re
from typing import Optional, Dict, List


def recompute_aggregate_scores(
    df: pd.DataFrame,
    metric_type_row: str = "Metric Type",
    aggregate_keywords: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    重新计算文献中的 aggregate score，同时保持输出 df 与原始 df 结构一致。

    功能：
    1. 自动从 `metric_type_row` 这一行识别各指标所属类别
    2. 自动识别原表中哪些列是 aggregate 列（如 Bio conservation, Domain continuity, Aggregate score）
    3. 对原始指标列按列做 min-max scaling
    4. 按类别计算 aggregate 分数
    5. 将重算后的 aggregate 分数直接覆盖回原始 df 中对应位置
    6. 输出 df 的行列结构、顺序与原始 df 完全一致

    参数
    ----
    df : pd.DataFrame
        原始数据框。通常行为方法，列为指标；最后一行可为 'Metric Type'。
    metric_type_row : str
        用于标记各列类别的行名，默认 'Metric Type'。
    aggregate_keywords : list[str] or None
        用于识别 aggregate 列的关键词。
        默认会识别包含 'aggregate score' 的类别标记，以及列名本身为类别名的列。

    返回
    ----
    pd.DataFrame
        与原 df 结构一致、但 aggregate 分数已重算覆盖后的新 df。
    """

    if metric_type_row not in df.index:
        raise ValueError(f"在 df.index 中未找到 {metric_type_row!r}。")

    if aggregate_keywords is None:
        aggregate_keywords = ["aggregate score"]

    out = df.copy()
    metric_type = out.loc[metric_type_row].copy()

    # -----------------------------
    # 1. 识别“原始指标列”与“aggregate列”
    # -----------------------------
    metric_groups: Dict[str, List[str]] = {}
    aggregate_cols = []

    for col in out.columns:
        group = metric_type[col]

        if pd.isna(group):
            continue

        group_str = str(group).strip()
        if group_str == "":
            continue

        # 若 Metric Type 行把该列标为 "Aggregate score"，则该列不是原始指标列
        if any(k.lower() in group_str.lower() for k in aggregate_keywords):
            aggregate_cols.append(col)
            continue

        metric_groups.setdefault(group_str, []).append(col)

    if len(metric_groups) == 0:
        raise ValueError("未识别到任何原始指标类别，请检查 Metric Type 行。")

    # 数据部分（去掉 Metric Type 行）
    data = out.drop(index=metric_type_row).copy()

    # -----------------------------
    # 2. 将原始指标列转为数值
    # -----------------------------
    raw_metric_cols = []
    for cols in metric_groups.values():
        raw_metric_cols.extend(cols)
    raw_metric_cols = list(dict.fromkeys(raw_metric_cols))  # 去重并保序

    for col in raw_metric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # -----------------------------
    # 3. 对每个原始指标列做 min-max scaling
    # -----------------------------
    def minmax_scale(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if valid.empty:
            return pd.Series(np.nan, index=series.index)
        smin = valid.min()
        smax = valid.max()
        if smax == smin:
            return pd.Series(0.5, index=series.index)
        return (series - smin) / (smax - smin)

    scaled_metrics = pd.DataFrame(index=data.index)

    for col in raw_metric_cols:
        scaled_metrics[col] = minmax_scale(data[col])

    # -----------------------------
    # 4. 计算每个类别的 aggregate 分数
    # -----------------------------
    group_scores = {}
    for group_name, cols in metric_groups.items():
        group_scores[group_name] = scaled_metrics[cols].mean(axis=1, skipna=True)

    group_scores_df = pd.DataFrame(group_scores, index=data.index)

    # 总 Aggregate score：对所有类别分数取均值
    overall_score = group_scores_df.mean(axis=1, skipna=True)

    # -----------------------------
    # 5. 覆盖回原始 df 中已有的 aggregate 列
    # -----------------------------
    # 规则：
    # - 若某列名恰好等于某个 group 名，则写入对应类别分数
    # - 若列名包含 "Aggregate score"，则写入 overall_score
    #
    # 例如：
    #   "Bio conservation"   <- group_scores["Bio conservation"]
    #   "Domain continuity"  <- group_scores["Domain continuity"]
    #   "Aggregate score"    <- overall_score
    #
    # 其他列不动

    for col in out.columns:
        col_str = str(col).strip()

        # 类别列：例如 "Bio conservation", "Domain continuity"
        if col_str in group_scores_df.columns:
            out.loc[data.index, col] = group_scores_df[col_str]
            continue

        # 总分列：例如 "Aggregate score"
        if any(k.lower() in col_str.lower() for k in aggregate_keywords):
            out.loc[data.index, col] = overall_score
            continue

    return out