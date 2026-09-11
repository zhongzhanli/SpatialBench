import subprocess
import os
import warnings
from typing import Literal
warnings.filterwarnings("ignore")

ENV = os.environ.copy()
for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    ENV[var] = "1"


def call_r_method(method: str,
                  RNA_file_path: str,
                  ATAC_file_path: str,
                  ADT_file_path: str,
                  save_path: str,
                  n_cluster: int,
                  batch_key: str = "batch",
                  hvg_num: int = 3000,
                  save_key: str = None,
                  script_path: str = None,
                  conda_env: str = None,
                  spec_params: dict = {},
                  mode: Literal["mo", "mb"] = "mo",
):
    
    if RNA_file_path is not None and not os.path.exists(RNA_file_path):
        raise ValueError(f"input file '{RNA_file_path}' does not exist")
    if ATAC_file_path is not None and not os.path.exists(ATAC_file_path):
        raise ValueError(f"input file '{ATAC_file_path}' does not exist")
    if ADT_file_path is not None and not os.path.exists(ADT_file_path):
        raise ValueError(f"input file '{ADT_file_path}' does not exist")
    
    cwd = os.path.dirname(__file__)

    data_type = "multiomics" if mode == "mo" else "multibatch"
    script_path = script_path or f"{cwd}/_script/{data_type}/_{method.lower()}.r"

    if not os.path.exists(script_path):
        raise ValueError(f"'{method}' script does not exist")
    save_key = save_key or method.lower()

    if RNA_file_path is None:
        RNA_file_path =  "NULL"
    if ATAC_file_path is None:
        ATAC_file_path = "NULL"
    if ADT_file_path is None:
        ADT_file_path = "NULL"

    cmd = f"Rscript {script_path} {n_cluster} {RNA_file_path} {ATAC_file_path} {ADT_file_path} {save_path} {save_key}  {hvg_num} {batch_key}"

    if(conda_env is not None):
        cmd = f"conda run -n {conda_env} " + cmd

    if(method.lower() in spec_params):
        if not isinstance(spec_params[method.lower()], list):
            raise ValueError(f"`spec_params` for {method} must be list")
        params = " ".join([str(i) for i in spec_params[method.lower()]])
        cmd = cmd + f" {params}"
    try:
        p = subprocess.run(cmd, shell=True,check=True, capture_output=True, env=ENV)
        _ = p.stdout 
        return ""
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode().split("\n")
        stderr = [f"An error occurred while running '{method}'. The log message is as follows:"] + stderr
        return stderr

def call_py_method(method: str,
                   RNA_file_path: str,
                   ATAC_file_path: str,
                   ADT_file_path: str,
                   save_path: str,
                   n_cluster: int,
                   batch_key: str = "batch",
                   hvg_num: int = 3000,
                   save_key: str = None,
                   script_path: str = None,
                   conda_env: str = None,
                   spec_params: dict = {},
                   mode: Literal["mo", "mb"] = "mo",
):
    
    if RNA_file_path is not None and not os.path.exists(RNA_file_path):
        raise ValueError(f"input file '{RNA_file_path}' does not exist")
    if ATAC_file_path is not None and not os.path.exists(ATAC_file_path):
        raise ValueError(f"input file '{ATAC_file_path}' does not exist")
    if ADT_file_path is not None and not os.path.exists(ADT_file_path):
        raise ValueError(f"input file '{ADT_file_path}' does not exist")

    cwd = os.path.dirname(__file__)

    data_type = "multiomics" if mode == "mo" else "multibatch"
    script_path = script_path or f"{cwd}/_script/{data_type}/_{method.lower()}.py"

    if not os.path.exists(script_path):
        raise ValueError(f"'{method}' script does not exist")
    save_key = save_key or method.lower()

    if RNA_file_path is None:
        RNA_file_path =  "NULL"
    if ATAC_file_path is None:
        ATAC_file_path = "NULL"
    if ADT_file_path is None:
        ADT_file_path = "NULL"

    cmd = f"python {script_path} {n_cluster} {RNA_file_path} {ATAC_file_path} {ADT_file_path} {save_path} {save_key} {hvg_num} {batch_key}"
    if(conda_env is not None):
        cmd = f"conda run -n {conda_env} " + cmd

    if(method.lower() in spec_params):
        if not isinstance(spec_params[method.lower()], list):
            raise ValueError(f"`spec_params` for {method} must be list")
        params = " ".join([str(i) for i in spec_params[method.lower()]])
        cmd = cmd + f" {params}"

    try:
        p = subprocess.run(cmd, shell=True,check=True, capture_output=True, env=ENV)
        _ = p.stdout 
        return ""
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode().split("\n")
        stderr = [f"An error occurred while running '{method}'. The log message is as follows:"] + stderr
        return stderr