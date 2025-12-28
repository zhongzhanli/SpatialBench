import logging
import sys
import os
from pathlib import Path
from typing import Literal, List, Tuple, Optional, Dict, Callable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.cm
from anndata import AnnData
import subprocess
import pandas as pd
import numpy as np
import pickle
import matplotlib
import shutil
from collections import defaultdict

from .metrics import cal_scib_metrics, cal_ari_nmi, cal_chaos_pas, _seurat_marker_score
from .utils import _shorten_list, generate_test_data, prefix_reindex, _min_max_scale
from .plotting import plot_results_table, _embed_plot, _legend_plot, _heatmap_legend_plot, _box_plot
from .call_methods import call_py_method, call_r_method
from .utils import repeat_to_length, transform_coord, get_scatter_cmap, get_default_cmap, map_to_integers, split_adata

ALL_METHODS = ["Harmony", "LIGER", "PRECAST", "Scanorama", "scVI", "SEDR", "Seurat", \
               "SLAT", "SPIRAL", "DeepST", "STAligner", "STAMP", "Graspot", "BASS",
               "spaVAE", "Unintegrated"]
R_METHODS = ["Seurat",  "LIGER", "PRECAST", "Harmony", "BASS"]
COMM_GENE_METHODS = ["LIGER", "SPIRAL", "STAligner"]
KNN_GRAPH_METHODS = ["SPIRAL", "SLAT", "SEDR", "STAligner"]
RESULT_FILES = ["UMAP", "Cluster", "Embed", "Batch"]
PREFIX_REINDEX_METHOD = ["SEDR"]
SPATIAL_METHODS = ["PRECAST", "SEDR", "SLAT", "SPIRAL", "DeepST", "STAligner", "STAMP", "Graspot", "BASS", "spaVAE"]
METRIC_ORDER = ['CHAOS', 'PAS', 'Isolated labels', 'NMI', 'ARI', 'Silhouette label', 'cLISI', 'Silhouette batch',
                'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison', 'Domain continuity', 'Batch correction', 
                'Bio conservation', 'Total']

class Benchmarker():
    def __init__(self, R_conda_env: str = None):
        cwd = os.path.dirname(__file__)
        script_folder = Path(f"{cwd}/_script/")
        files = [f.name for f in script_folder.iterdir() if f.is_file() and f.name.startswith("_")]
        self.all_methods = sorted(ALL_METHODS)
        self.spatial_methods = sorted(SPATIAL_METHODS)
        self.script_files = [".".join(f.split(".")[0:-1]).replace("_","",1) for f in files]
        self.R_conda_env = R_conda_env

        self.logger = logging.getLogger('bm_logger')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.propagate = False
        self._plot_param_initialized = False
    
    def test(self, temp_dir: str = None, methods:  str|List = "all", conda_envs: dict = None, 
             h5ad2rds_env: str = None, verbose: bool = True, workers: int = None):
        
        remove_temp = False if temp_dir is not None else True
        workers = workers or len(self.all_methods)
        test_ad1 = generate_test_data(batch_key="batch")

        if(temp_dir is None):
            if(not os.path.exists("bm_test")):
                os.mkdir("bm_test")
            temp_dir = "bm_test"
            if(verbose):
                self.logger.info(f"Using temporary directory: '{temp_dir}'")
        test_ad1.write(f"{temp_dir}/bm_test.h5ad")
        in_file = f"{temp_dir}/bm_test.h5ad"

        self.h5ad2rds(in_file=in_file, out_file=f"{temp_dir}/bm_test.rds", temp_dir=temp_dir, conda_env=h5ad2rds_env)
        self.run(RDS_file_path=f"{temp_dir}/bm_test.rds", H5AD_file_path=f"{temp_dir}/bm_test.h5ad", save_path=temp_dir,
                 n_cluster=2, methods=methods, batch_key="batch", knn_cutoff=1, verbose=verbose, hvg_num=80, nFactors=5,
                 workers=workers, conda_envs=conda_envs)
        
        if(remove_temp):
            if(verbose):
                self.logger.info("Removing temporary files")
            shutil.rmtree(temp_dir)
        
    def _run_single_method(self, method: str, RDS_file_path: str, H5AD_file_path: str, save_path: str,
                           n_cluster: int, cluster_method: str, batch_key: str, hvg_num: int,
                           conda_envs: dict, spec_params: dict, verbose: bool):
        try:
            if(verbose):
                logger = logging.getLogger(f'bm_logger.{method}')
                logger.setLevel(logging.DEBUG)
                logger.propagate = False

                if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
                    handler = logging.StreamHandler(sys.stdout)
                    handler.setFormatter(logging.Formatter(
                        '%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'))
                    logger.addHandler(handler)

                logger.info(f"Start running '{method}'")

            if method in COMM_GENE_METHODS:
                hvg_num += 2000

            file_path = RDS_file_path if method in R_METHODS else H5AD_file_path

            conda_env = conda_envs.get(method) if conda_envs and method in conda_envs else (
                self.R_conda_env if method in R_METHODS else None
            )
            if(method in R_METHODS):
                stdout = call_r_method(
                    method=method,
                    file_path=file_path,
                    save_path=save_path,
                    save_key=method.lower(),
                    n_cluster=n_cluster,
                    cluster_method=cluster_method,
                    batch_key=batch_key,
                    hvg_num=hvg_num,
                    conda_env=conda_env,
                    spec_params=spec_params
                )
            else:
                stdout = call_py_method(
                    method=method,
                    file_path=file_path,
                    save_path=save_path,
                    save_key=method.lower(),
                    n_cluster=n_cluster,
                    cluster_method=cluster_method,
                    batch_key=batch_key,
                    hvg_num=hvg_num,
                    conda_env=conda_env,
                    spec_params=spec_params
                )

            return (method, True, stdout)

        except Exception as e:
            return (method, False, [str(e)])

    def run(self, RDS_file_path: str, H5AD_file_path: str, save_path: str = None, n_cluster: int = None,
            methods: str|List = "all", cluster_method: Literal["leiden", "mclust"] = "leiden",
            batch_key: str = "batch", hvg_num: int = 3000, conda_envs: dict = None, workers: int = 1, 
            nFactors: int =20, knn_cutoff: int = 5, verbose: bool = True, rep: int = 1,
            **kwargs):
        
        assert methods=="all" or isinstance(methods, list)

        if(isinstance(methods, list)):
             for i in methods:
                assert i.lower() in [m.lower() for m in self.all_methods] and i.lower() in self.script_files,\
                f"'{i}' does not support"

        to_run_methods = self.all_methods if methods=="all" else methods
        preview = 4
        msg = _shorten_list(to_run_methods, preview)
        if(rep > 1):
            msg = f"About to run {len(to_run_methods)} methods, each repeated {rep} times: {msg}"
        else:
            msg = f"About to run {len(to_run_methods)} methods (no repeats): {msg}"
        self.logger.info(msg)

        max_workers = min(workers, int(len(to_run_methods)*rep))
        if(verbose):
            self.logger.info(f"Number of workers: {max_workers}")

        spec_params = {"liger": [nFactors]}
        for m in KNN_GRAPH_METHODS:
            spec_params[m.lower()] = [knn_cutoff]

        success_list = []
        failure_list = []
        failure_log = defaultdict(list)
        if(rep > 1):
            method2status = defaultdict(list)
            save_paths = []
            for i in range(rep):
                sp = save_path+f"/rep{i+1}/"
                save_paths.append(sp)
                if(not os.path.exists(sp)):
                    os.mkdir(sp)
        else:
            save_paths = [save_path]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_single_method, method, RDS_file_path, H5AD_file_path,
                                save_path, n_cluster, cluster_method, batch_key,
                                hvg_num, conda_envs, spec_params, verbose): (method, save_path)
                for method in to_run_methods for save_path in save_paths
            }

            for future in as_completed(futures):
                method, sp = futures[future]
                try:
                    method, success, result = future.result()
                    if success and (result == "" or result is None):
                        if(verbose):
                            self.logger.info(f"'{method}' completed successfully.")
                        success_list.append(f"{method}")
                        if(rep > 1):
                            method2status[method].append(True)
                    else:
                        failure_log[method].append(result)
                        raise ValueError(f"'{method}' failed.")
                except Exception as exc:
                    self.logger.error(f"{exc}")
                    failure_list.append(method)
                    if(rep > 1):
                        method2status[method].append(False) 
        
        for m in failure_log.keys():
            results = failure_log[m]
            for result in results:
                for i, line in enumerate(result):
                    if line != "":
                        if(i==0):
                            self.logger.error(line)
                        else:
                            print(line)
                    if(i==len(result)-1):
                        print("\n")
        if(rep > 1):
            all_success = []
            partial_success = []
            all_failed = []
            all_methods_num = len(to_run_methods)
            for method, results in method2status.items():
                if all(results):
                    all_success.append(method)
                elif any(results):
                    partial_success.append(method)
                else:
                    all_failed.append(method)

            self.logger.info("===================================")
            self.logger.info("Run Summary:")
            self.logger.info(f"All Success ({len(all_success)}/{all_methods_num}): {_shorten_list(all_success)}")
            self.logger.info(f"Partial Success ({len(partial_success)}/{all_methods_num}): " + \
                             f"{_shorten_list(partial_success, preview=15)}")
            self.logger.info(f"All Failed ({len(all_failed)}/{all_methods_num}): " + \
                             f"{_shorten_list(all_failed, preview=15)}")
            self.logger.info("===================================")
        else:
            all_methods_num = len(to_run_methods)
            self.logger.info("===================================")
            self.logger.info("Run Summary:")
            self.logger.info(f"Success ({len(success_list)}/{all_methods_num}): {_shorten_list(success_list)}")
            self.logger.info(f"Failed ({len(failure_list)}/{all_methods_num}): {_shorten_list(failure_list, preview=15)}")
            self.logger.info("===================================")

    def h5ad2rds(self, in_file: str, out_file: str, temp_dir: str = None, remove_temp: bool = True, 
                 conda_env: str = None, verbose: bool = False):
        remove_raw = False
        if(isinstance(in_file, AnnData)):
            remove_raw = True
            if(temp_dir is None):
                if(not os.path.exists("bm_temp")):
                    os.mkdir("bm_temp")
                temp_dir = "bm_temp"
            if(verbose):
                self.logger.info(f"Using temporary directory: '{temp_dir}'")
            in_file.write(f"{temp_dir}/bm_temp.h5ad")
            in_file = f"{temp_dir}/bm_temp.h5ad"
        cwd = os.path.dirname(__file__)

        conda_env = conda_env or self.R_conda_env
        if(conda_env is not None):
            cmd = f"conda run -n {conda_env} Rscript {cwd}/h5ad2rds.r --infile {in_file} --outfile {out_file}"
        else:
            cmd = f"Rscript {cwd}/h5ad2rds.r --infile {in_file} --outfile {out_file}"
        try:
            p = subprocess.run(cmd, shell=True,check=True, capture_output=True)
            if(verbose):
                self.logger.info(f"Successfully converted the h5ad file to RDS format")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode().split("\n")
            self.logger.error(f"An error occurred while running 'h5ad2rds'. The log message is as follows:")
            for line in stderr:
                if(line!=""):
                    self.logger.error(line)
        if(remove_temp):
            if(verbose):
                self.logger.info("Removing temporary files")
            if(remove_raw):
                os.remove(in_file)
            temp_file = f"{in_file.split('/')[-1].removesuffix('.h5ad')}.h5seurat"
            if(os.path.exists(temp_file)):
                os.remove(temp_file)

    def read_result(self, path: str = None, methods: str|List = "all", reindex: bool = True, index: List = None,
                rep: int|bool = False, save: str = None):
    
        if reindex and index is None:
            raise ValueError(f"When `reindex=True`, `index` must be provided")
        
        assert methods == "all" or isinstance(methods, list)
        methods = self.all_methods if methods == "all" else methods

        res_dict = {rf: {} for rf in RESULT_FILES}

        if rep is False or rep == 1:
            reps = [None]
        elif isinstance(rep, int) and rep > 1:
            reps = [f"rep{i+1}" for i in range(rep)]
        else:
            raise ValueError("rep must be False or a positive integer > 1")

        for rf in RESULT_FILES:
            for m in methods:
                results = []
                for rep_tag in reps:
                    prefix = path or "."
                    subpath = os.path.join(prefix, rep_tag) if rep_tag else prefix
                    if m in self.all_methods:
                        fn = f"{rf.lower()}_{m.lower()}.csv"
                    else:
                        fn = f"{rf.lower()}_{m}.csv"
                    full_path = os.path.join(subpath, fn)

                    if not os.path.exists(full_path):
                        raise FileNotFoundError(f"'{rf}' result for '{m}' not found at: {full_path}")

                    df = pd.read_csv(full_path, index_col=0)

                    if reindex:
                        if m in PREFIX_REINDEX_METHOD:
                            df = prefix_reindex(df, index)
                        else:
                            df = df.reindex(index)
                        assert not df.isna().any().any(), f"'{rf}' result for '{m}' reindex error"

                    results.append(np.array(df))

                res_dict[rf][m] = results if len(results) > 1 else results[0]
        if save is not None:
            assert save[-4:]==".pkl", f"save file name must end with '.pkl'"
            save_path = os.path.abspath(save)
            if os.path.exists(save_path):
                self.logger.warning(f"File already exists and will be overwritten: '{save_path}'")
            else:
                self.logger.info(f"File successfully saved to: '{save_path}'")
            try:
                with open(save_path, "wb") as f:
                    pickle.dump(res_dict, f)
            except Exception as e:
                    self.logger.error(f"Failed to save results:\n{e}")
        return res_dict

    def cal_metrics(self, adata: AnnData, res_dict: dict, embed_key: str = "Embed", cluster_key: str = "Cluster",
                    batch_key: str = "batch", label_key: str = "cell_type", workers: int = 1, 
                    min_max_scale: bool = False, methods: str|List = "all", verbose: bool = False, 
                    save: str = None, rep: int = None, **kwargs
                    ):
        
        assert batch_key in adata.obs, f"'{batch_key}' does not exist in anndata object"
        assert label_key in adata.obs, f"'{label_key}' does not exist in anndata object"
        assert methods == "all" or (isinstance(methods, list) and len(methods) > 0)

        embed_dict = res_dict[embed_key]
        cluster_dict = res_dict[cluster_key]
        cal_embed_dict = {}
        cal_cluster_dict = {}
        if(isinstance(methods, list)):
            for m in methods:
                assert m in embed_dict and m in cluster_dict, f"'{m}' does not exist in the result dict"
                cal_embed_dict[m] = embed_dict[m]
                cal_cluster_dict[m] = cluster_dict[m]
        else:
            methods = sorted(list(embed_dict.keys()))
            cal_embed_dict = embed_dict
            cal_cluster_dict = cluster_dict

        preview = 4
        msg = _shorten_list(sorted(cal_embed_dict.keys()), preview)
        rep = rep or max(len(v) if isinstance(v, list) else 1 for v in cal_embed_dict.values())
        if rep > 1:
            msg = f"About to calculate metrics for {len(methods)} methods, each repeated {rep} times: {msg}"
        else:
            msg = f"About to calculate metrics for {len(methods)} methods (no repeats): {msg}"
        self.logger.info(msg)

        if(verbose):
            self.logger.info(f"Star calculating metrics using 'scib_metrics'")

        scib_metrics = cal_scib_metrics(adata=adata, batch_key=batch_key, label_key=label_key, workers=workers,
                                        embed_dict=cal_embed_dict, verbose=False, min_max_scale=min_max_scale,
                                        n_rep=rep, **kwargs)
        if(verbose):
            self.logger.info(f"Star calculating 'ARI' and 'NMI'")

        ari_nmi = cal_ari_nmi(adata=adata, label_key=label_key, cluster_dict=cal_cluster_dict, 
                              min_max_scale=min_max_scale, n_rep=rep)
        
        if(verbose):
            self.logger.info(f"Star calculating 'CHAOS' and 'PAS'")

        chaos_pas = cal_chaos_pas(adata=adata, cluster_dict=cluster_dict, batch_key=batch_key, n_rep=rep,
                                  min_max_scale=min_max_scale)
        
        for i in range(len(scib_metrics)):
            ari_nmi[i] = ari_nmi[i].reindex(scib_metrics[i].index)
            chaos_pas[i] = chaos_pas[i].reindex(scib_metrics[i].index)

            scib_metrics[i]["KMeans ARI"] = ari_nmi[i]["ARI"].copy()
            scib_metrics[i]["KMeans NMI"] = ari_nmi[i]["NMI"].copy()

            scib_metrics[i] = scib_metrics[i].rename(columns={"KMeans ARI": "ARI", 
                                                              "KMeans NMI": "NMI"})

            scib_metrics[i] = pd.concat([scib_metrics[i], chaos_pas[i]], axis=1)
            
            per_class_score = scib_metrics[i].T.groupby("Metric Type").mean().transpose()
            # print(per_class_score)
            per_class_score["Total"] = (
                0.4 * per_class_score["Batch correction"] + 0.6 * per_class_score["Bio conservation"]
            )
            per_class_score.loc["Metric Type"] = ['Aggregate score'] * per_class_score.shape[1]
            per_class_score = per_class_score.reindex(scib_metrics[i].index)
            scib_metrics[i]["Bio conservation"] = per_class_score["Bio conservation"]
            scib_metrics[i]["Total"] = per_class_score["Total"]
            scib_metrics[i] = pd.concat([scib_metrics[i], per_class_score["Domain continuity"]], axis=1)
            scib_metrics[i] = scib_metrics[i].T.reindex(METRIC_ORDER).T

        if(save is not None):
            assert save[-4:]==".pkl", f"save file name must end with '.pkl'"
            save_path = os.path.abspath(save)
            if os.path.exists(save_path):
                self.logger.warning(f"File already exists and will be overwritten: '{save_path}'")
            else:
                self.logger.info(f"File successfully saved to: '{save_path}'")
            try:
                with open(save_path, "wb") as f:
                    pickle.dump(scib_metrics, f)
            except Exception as e:
                    self.logger.error(f"Failed to save results:\n{e}")
            
        return  scib_metrics

    def cal_marker_score(self, adata: AnnData, label_dict: dict, marker_genes: List,
                         batch_key: str = "batch", order: List = None, re_run: bool = False,
                         R_conda_env: str = None, verbose: bool = True, **kwargs):

        order = order or sorted(label_dict.keys())
        batches = list(set(adata.obs[batch_key]))
        assert batch_key in adata.obs, f"'{batch_key}' do not exist"

        conda_env = R_conda_env or self.R_conda_env

        gene_idx = {var: i for i, var in enumerate(adata.var_names)}
        temp_adata = adata[:, [gene_idx[i] for i in marker_genes]].copy()
        if(not os.path.exists("bm_temp")):
            os.mkdir("bm_temp")

        if(not re_run):
            for method in order:
                for batch_name in batches:
                    file_path = f"bm_temp/{method}_cluster_{batch_name}.csv"
                    if(not os.path.exists(file_path)):
                        re_run = True
            if(not re_run):
                self.logger.info(f"Marker score results exist, set `re_run=True` to re-calculate marker score")
        
        if(re_run):
            temp_adata.obs = pd.DataFrame(index=adata.obs_names)
            temp_adata.obs[batch_key] =  list(adata.obs[batch_key])
            for method in order:
                temp_adata.obs[f"{method}_cluster"] = [str(i) for i in list(label_dict[method].flatten())]
            self.h5ad2rds(in_file=temp_adata, out_file="bm_temp/bm_temp.rds", conda_env=conda_env, verbose=verbose)

            cwd = os.getcwd()
            if(verbose):
                self.logger.info(f"Calculating marker score using 'Seurat'")
            stderr = _seurat_marker_score(file_path=f"{cwd}/bm_temp/bm_temp.rds",
                                        save_path=f"{cwd}/bm_temp", batch_key=batch_key,
                                        conda_env=conda_env)
            if stderr=="":
                if(verbose):
                    self.logger.info(f"Marker score results saved in '{cwd}/bm_temp'")
            else:
                self.logger.error(f"An error occurred while running 'cal_marker_score'. The log message is as follows:")
                for line in stderr:
                    if(line!=""):
                        self.logger.error(line)
        
        gene_score_dict = defaultdict(list)
        for method in order:
            merged_dict = defaultdict(lambda: defaultdict(list))
            for batch_name in batches:
                file_path = f"bm_temp/{method}_cluster_{batch_name}.csv"
                df = pd.read_csv(file_path, index_col=0)
                for _, row in df.iterrows():
                    gene = row["gene"]
                    cluster = str(row["cluster"])
                    auc = row["myAUC"]
                    merged_dict[gene][cluster].append(auc)
            for gene in merged_dict.keys():
                sum_score = [sum(auc_list) for auc_list in merged_dict[gene].values()]
                gene_score_dict[gene].append(max(sum_score)/len(batches))

        res_df = pd.DataFrame(index=order)
        for gene in marker_genes:
            res_df[gene] = gene_score_dict[gene]
        return res_df

    def set_plot_params(self, params_dict: dict = None, font_file_path: str = None, set_to_default: bool = False):
        import matplotlib
        from matplotlib import font_manager

        d_params = {'figure.dpi': 100, 'font.size': 12, "savefig.dpi": 300}
        if(params_dict is None):
            params_dict = d_params
        else:
            d_params.update(params_dict)

        if(not self._plot_param_initialized or set_to_default):
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            self._plot_param_initialized = True

        matplotlib.rcParams.update(params_dict)
        if font_file_path is not None:
            font_manager.fontManager.addfont(font_file_path)
            font_prop = font_manager.FontProperties(fname=font_file_path)
            font_name = font_prop.get_name()
            matplotlib.rcParams['font.family'] = font_name
            self.logger.info(f"Custom font '{font_name}' has been set")
        
    def plot_heatmap(self, metric_df: pd.DataFrame, min_max_scale: bool = False, save: str = None,
                     hl_spatial_methods: bool = False, hl_color: str = "#ED7D31", **kwargs):
        
        order = [i for i in METRIC_ORDER if i in metric_df.columns ]
        metric_df = metric_df.T.reindex(order).T
        
        if(min_max_scale):
            metric_type = metric_df.loc["Metric Type", :]
            metric_df = metric_df.drop("Metric Type", axis=0)
            metric_df = _min_max_scale(metric_df)
            metric_df.loc["Metric Type", :] = metric_type
        # if(hl_spatial_methods):
        #     plot_results_table(res_df=metric_df, show=True, save_dir=save, highlight_methods=SPATIAL_METHODS, 
        #                        highlight_color=hl_color,**kwargs)
        # else:
        #     plot_results_table(res_df=metric_df, show=True, save_dir=save, **kwargs)
        plot_results_table(res_df=metric_df, show=True, save_dir=save, **kwargs)

    def plot_umap(self, embed_dict: dict, batch_dict: dict = None, annot_list: List = None, label_dict: dict = None,
                  order: List = None, figsize: Tuple = (20,5), ncol: int = 5, save: str = None, 
                  inner_gs_col: int = 1, inner_gs_row: int = 1, palettes: List = [None], merge: bool = False, 
                  xlabel: str|List = None, ylabel: str|List = None, merge_margin_size: float = 0.2, **kwargs):

        plot_embed = []
        order = order or sorted(embed_dict.keys())
        for m in order:
            assert m in embed_dict, f"'{m}' does not exist in `embed_dict`"
            if(not merge):
                plot_embed.append(embed_dict[m])
            else:
                plot_embed.append(embed_dict[m])
                plot_embed.append(embed_dict[m])
        temp_adata = AnnData(X=np.zeros((list(embed_dict.values())[0].shape[0], 1)))
            
        if(xlabel=="name"):
            xlabel = repeat_to_length(order, len(plot_embed))
        elif(xlabel is not None):
            xlabel = repeat_to_length(xlabel, len(plot_embed))
        if(ylabel=="name"):
            ylabel = repeat_to_length(order, len(plot_embed))
        elif(ylabel is not None):
            ylabel = repeat_to_length(ylabel, len(plot_embed))

        if(not merge):
            if(annot_list is not None):
                plot_label = []
                for m in order:
                    # assert m in label_dict, f"'{m}' does not exist in `annot_list`"
                    plot_label.append([str(i) for i in annot_list])
                _embed_plot(adata=temp_adata, embed_list=plot_embed, label_list=plot_label, figsize=figsize,
                            ncol=ncol, save=save, inner_gs_col=inner_gs_col, inner_gs_row=inner_gs_row,
                            palettes=palettes, xlabel=xlabel, ylabel=ylabel, **kwargs)
            
            if(batch_dict is not None):
                plot_batch = []
                for m in order:
                    assert m in batch_dict, f"'{m}' does not exist in `batch_dict`"
                    plot_batch.append([str(i) for i in batch_dict[m]])

                _embed_plot(adata=temp_adata, embed_list=plot_embed, label_list=plot_batch, figsize=figsize,
                            ncol=ncol, save=save, inner_gs_col=inner_gs_col, inner_gs_row=inner_gs_row,
                            palettes=palettes, xlabel=xlabel, ylabel=ylabel, **kwargs)
        else:
            assert annot_list is not None and batch_dict is not None, \
            f"When setting `merge=True`, `batch_dict` and `annot_list` must be provided"
            plot_label = []
            for m in order:
                # assert m in label_dict, f"'{m}' does not exist in `label_dict`"
                assert m in batch_dict, f"'{m}' does not exist in `batch_dict`"
                plot_label.append([str(i) for i in batch_dict[m]])
                plot_label.append([str(i) for i in annot_list])

            _embed_plot(adata=temp_adata, embed_list=plot_embed, label_list=plot_label, figsize=figsize,
                        ncol=ncol, save=save, inner_gs_col=inner_gs_col, inner_gs_row=inner_gs_row,
                        palettes=palettes, xlabel=xlabel, ylabel=ylabel,**kwargs)

    def plot_spatial(self, spatial: List[np.ndarray], label_dict: dict, order: List = None, figsize: Tuple = (20,5),
                     ncol: int = 5, save: str = None, inner_gs_col: int = 1, inner_gs_row: int = 1,
                     palette: matplotlib.cm = None, xlabel: str|List = None, ylabel: str|List = None, 
                     inner_common_camp: bool = True, sizes: List = None, **kwargs):
        
        plot_label = []
        plot_embed = []
        order = order or sorted(label_dict.keys())
        for m in order:
            assert m in label_dict, f"'{m}' does not exist in `label_dict`"
            assert np.concatenate(spatial).shape[0]==label_dict[m].shape[0], f"shape do not match"
            plot_embed = plot_embed + spatial
            start = 0
            all_label = map_to_integers(label_dict[m].flatten())
            for i in spatial:
                plot_label.append(all_label[start:start+i.shape[0]])
                start = start+i.shape[0]
        
        if(xlabel=="name"):
            xlabel = repeat_to_length(order, len(plot_embed))
        elif(xlabel is not None):
            xlabel = repeat_to_length(xlabel, len(plot_embed))

        if(ylabel=="name"):
            ylabel = repeat_to_length(order, len(plot_embed))
        elif(ylabel is not None):
            ylabel = repeat_to_length(ylabel, len(plot_embed))
        
        if(sizes is not None):
            sizes = repeat_to_length(sizes, len(plot_embed))


        temp_adata = AnnData(X=np.zeros((list(label_dict.values())[0].shape[0], 1)))
        _embed_plot(adata=temp_adata, embed_list=plot_embed, label_list=plot_label, figsize=figsize,
                    ncol=ncol, save=save, inner_gs_col=inner_gs_col, inner_gs_row=inner_gs_row,
                    palettes=[palette], xlabel=xlabel, ylabel=ylabel, 
                    sizes=sizes, **kwargs)

    def plot_gene_exp(self, adata: AnnData, genes: List,  embed: List[np.ndarray],
                      figsize: Tuple = (20,5), use_layer: str = None,
                      ncol: int = 5, save: str = None, inner_gs_col: int = 1, inner_gs_row: int = 1,
                      cmap: matplotlib.cm = None, xlabel: str|List = None, ylabel: str|List = None, 
                      sizes: List = None, **kwargs):
        
        used_x = adata.X if use_layer is None else adata.layers[use_layer]

        gene_idx = {var: i for i, var in enumerate(adata.var_names)}

        plot_label = []
        plot_embed = []

        import scipy.sparse as sp

        for gene in genes:
            assert gene in gene_idx, f"'{gene}' does not exist"
            plot_embed = plot_embed + embed
            start = 0
            expr_lst = used_x[:,gene_idx[gene]]
            expr_lst = expr_lst.toarray().flatten().tolist() if sp.issparse(expr_lst) else expr_lst.tolist()
            for i in embed:
                plot_label.append(expr_lst[start:start+i.shape[0]])
                start = start+i.shape[0]
        
        if(xlabel=="name"):
            xlabel = genes
        elif(xlabel is not None):
            xlabel = repeat_to_length(xlabel, len(plot_embed))

        if(ylabel=="name"):
            ylabel = genes
        elif(ylabel is not None):
            ylabel = repeat_to_length(ylabel, len(plot_embed))
        
        if(sizes is not None):
            sizes = repeat_to_length(sizes, len(plot_embed))

        temp_adata = AnnData(X=np.zeros((adata.shape[0], 1)))

        _embed_plot(adata=temp_adata, embed_list=plot_embed, label_list=plot_label, figsize=figsize,
                    ncol=ncol, save=save, inner_gs_col=inner_gs_col, inner_gs_row=inner_gs_row,
                    cmaps=[cmap], xlabel=xlabel, ylabel=ylabel,
                    sizes=sizes, **kwargs)

    def plot_marker_score(self, metric_df: pd.DataFrame, sort_by: List|Literal["mean", "median"]="median",
                          figsize: Tuple = None, vert: bool = False, xlabel: str = None, ylabel: str =None,
                          label_rotation: int = None, save: str = None, show_data_points: bool = True, **kwargs):
        
        if(isinstance(sort_by, list)):
            sorted_cols = sort_by
        elif(sort_by=="median"):
            sort_val = metric_df.median(axis=1).sort_values(ascending=True)
            sorted_cols = sort_val.index.tolist()
        elif(sort_by=="mean"):
            sort_val = metric_df.mean(axis=1).sort_values(ascending=True)
            sorted_cols = sort_val.index.tolist()
        else:
            raise ValueError(f"Unsupported `sort_by`: '{sort_by}'")
        
        if(figsize is None):
            if(vert):
                figsize = (0.4 * metric_df.shape[0], 4)
            else:
                figsize = (4, 0.4 * metric_df.shape[0])
        sorted_df = metric_df.loc[sorted_cols,:]
        assert not sorted_df.isna().any().any()
        
        _box_plot(df=sorted_df, figsize=figsize, label_rotation=label_rotation, ylabel=ylabel,
                  xlabel=xlabel, vert=vert, save=save, show_data_points=show_data_points, **kwargs)

    def plot_legend(self, category_lst: int = None, color_map: dict = None, ncol: int = 3, marker: str = "s",
                    figsize: Tuple[int] = (6,1), order: List = None,
                    markersize: int = 6, save: str = None,
                    **kwargs):
        assert category_lst is not None or color_map is not None
        if(color_map is None):
            self.logger.info(f"`color_map` is None, using default color_map based on `category_lst`")
            color_map = get_scatter_cmap(list(category_lst))
        _legend_plot(color_dict=color_map, marker=marker, ncol=ncol, figsize=figsize,
                     markersize=markersize, save=save, order=order,**kwargs)
    
    def plot_heatmap_legend(self, max_rank: int, cmap1: matplotlib.cm = None, cmap2: matplotlib.cm = None, 
                            save: str =  None):
        d_camp1, d_camp2 = get_default_cmap()
        cmap1 = cmap1 or d_camp1
        cmap2 = cmap2 or d_camp2

        _heatmap_legend_plot(cmap1=cmap1, cmap2=cmap2, n_ranks=max_rank, save=save)