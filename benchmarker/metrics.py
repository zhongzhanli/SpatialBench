from scib_metrics.benchmark import Benchmarker as scib_bm
from anndata import AnnData
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score, average_precision_score
from typing import List, Literal
import os
import subprocess
from .utils import split_adata, _min_max_scale

def cal_scib_metrics(adata: AnnData, batch_key: str = "batch", label_key: str = "cell_type",
                     workers: int = 1, embed_dict: dict = {}, verbose: bool = False,
                     min_max_scale: bool = True, n_rep: int = None):
    
    adata = adata.copy()
    methods = sorted(list(embed_dict.keys()))

    if(n_rep is None):
        if(isinstance(embed_dict[methods[0]], list)):
            n_rep = len(embed_dict[methods[0]])
        else:
            n_rep = 1
    
    metric_df = []
    for rep in range(n_rep):
        for m in methods:
            if(not isinstance(embed_dict[methods[0]], list)):
                adata.obsm[f"temp_embed_{m}"] = embed_dict[m]
            else:
                adata.obsm[f"temp_embed_{m}"] = embed_dict[m][rep]

        scib_benchmarker = scib_bm(
            adata,
            batch_key = batch_key,
            label_key = label_key,
            embedding_obsm_keys = [f"temp_embed_{m}" for m in methods],
            n_jobs = workers,
            progress_bar = verbose
        )
        scib_benchmarker.benchmark()
        df = scib_benchmarker.get_results(min_max_scale=min_max_scale)
        df.index = [i.replace("temp_embed_", "", 1) for i in df.index]
        metric_df.append(df)
    return metric_df

def cal_ari_nmi(adata, label_key: str = "cell_type", cluster_dict: dict = {}, min_max_scale: bool = False,
            n_rep: int = None):
    
    adata = adata.copy()
    methods = sorted(list(cluster_dict.keys()))

    if(n_rep is None):
        if(isinstance(cluster_dict[methods[0]], list)):
            n_rep = len(cluster_dict[methods[0]])
        else:
            n_rep = 1
    
    groud_truth = np.array(list(adata.obs[label_key]))
    metric_df = []
    for rep in range(n_rep):
        ari_col = []
        nmi_col = []
        for m in methods:
            if(not isinstance(cluster_dict[methods[0]], list)):
                cluster = np.array(cluster_dict[m]).flatten()
            else:
                cluster = np.array(cluster_dict[m][rep]).flatten()
            ari = adjusted_rand_score(groud_truth, cluster)
            nmi = normalized_mutual_info_score(groud_truth, cluster)
            ari_col.append(ari)
            nmi_col.append(nmi)

        df = pd.DataFrame({"ARI": ari_col, "NMI": nmi_col}, index=methods)
        if(min_max_scale):
            df = _min_max_scale(df)
        df = df.transpose()
        df['Metric Type'] = ['Bio conservation'] * df.shape[0]
        df = df.transpose()
        metric_df.append(df)
    return metric_df

def fx_1NN(i,location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
    dist_array[i] = np.inf
    return np.min(dist_array)

def fx_kNN(i,location_in,k,cluster_in):

    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)


    dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind]!=cluster_in[i])>(k/2):
        return 1
    else:
        return 0

def _chaos(clusterlabel, location):

    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel==k,:]
        if len(location_cluster)<=2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i,location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1

    return np.sum(dist_val)/len(clusterlabel)

def _pas(clusterlabel,location):
        
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [fx_kNN(i,matched_location,k=10,cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results)/len(clusterlabel)

def cal_chaos_pas(adata, cluster_dict: dict = {}, batch_key: str = "batch", n_rep: int = None,
                  min_max_scale: bool = False):

    adata = adata.copy()
    methods = sorted(list(cluster_dict.keys()))

    if(n_rep is None):
        if(isinstance(cluster_dict[methods[0]], list)):
            n_rep = len(cluster_dict[methods[0]])
        else:
            n_rep = 1
    
    metric_df = []
    for rep in range(n_rep):

        chaos_col, pas_col = [], []
        for m in methods:
            if(not isinstance(cluster_dict[methods[0]], list)):
                cluster = np.array(cluster_dict[m]).flatten()
            else:
                cluster = np.array(cluster_dict[m][rep]).flatten()
            adata.obs[f"{m}_cluster"] = cluster

        adata_list = split_adata(adata, batch_key=batch_key)

        for m in methods:
            chaos_scores = []
            pas_scores = []
            for t_ad in adata_list:
                temp_chaos =  _chaos(t_ad.obs[f"{m}_cluster"], t_ad.obsm["spatial"])
                chaos_scores.append(temp_chaos)
                temp_pas =  _pas(t_ad.obs[f"{m}_cluster"], t_ad.obsm["spatial"])
                pas_scores.append(temp_pas)

            chaos_col.append(1 - np.mean(chaos_scores))
            pas_col.append(1 - np.mean(pas_scores))

        df = pd.DataFrame({"CHAOS": chaos_col, "PAS": pas_col}, index=methods)
        if(min_max_scale):
            df = _min_max_scale(df)
        df = df.transpose()
        df['Metric Type'] = ['Domain continuity'] * df.shape[0]
        df = df.transpose()
        metric_df.append(df)

    return metric_df

def _seurat_marker_score(file_path:str, save_path: str, batch_key: str, conda_env: str = None):
    if not os.path.exists(file_path):
        raise ValueError(f"input file '{file_path}' does not exist")

    cwd = os.path.dirname(__file__)

    script_path = f"{cwd}/cal_marker_score.r"
    if not os.path.exists(script_path):
        raise ValueError(f"'{script_path}' do not exist")

    cmd = f"Rscript {script_path} {file_path} {save_path} {batch_key}"

    if(conda_env is not None):
        cmd = f"conda run -n {conda_env} " + cmd

    try:
        p = subprocess.run(cmd, shell=True,check=True, capture_output=True)
        _ = p.stdout 
        return ""
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode().split("\n")
        return stderr
        