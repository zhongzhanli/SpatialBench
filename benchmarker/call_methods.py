import subprocess
import os
import warnings
warnings.filterwarnings("ignore")

ENV = os.environ.copy()
for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    ENV[var] = "1"


def call_r_method(method: str, file_path: str, save_path: str, n_cluster: int, batch_key: str = "batch",
                  hvg_num: int = 3000, save_key: str = None, cluster_method: str = "mclust",
                  script_path: str = None, conda_env: str = None, spec_params: dict = {}):
    
    if not os.path.exists(file_path):
        raise ValueError(f"input file '{file_path}' does not exist")
    if not cluster_method in ["mclust","leiden"]:
        raise ValueError("Only support 'mclust' or 'leiden'")

    cwd = os.path.dirname(__file__)

    script_path = script_path or f"{cwd}/_script/_{method.lower()}.r"
    if not os.path.exists(script_path):
        raise ValueError(f"'{method}' script does not exist")
    save_key = save_key or method.lower()

    cmd = f"Rscript {script_path} {n_cluster} {file_path} {save_path} {save_key} {batch_key} {hvg_num} {cluster_method}"

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

def call_py_method(method: str,file_path: str, save_path: str, n_cluster: int, batch_key: str = "batch",
                   hvg_num: int = 3000, save_key: str = None, cluster_method: str = "mclust",
                   script_path: str = None, conda_env: str = None, spec_params: dict = {}):
    
    if not os.path.exists(file_path):
        raise ValueError(f"input file '{file_path}' does not exist")
    if not cluster_method in ["mclust","leiden"]:
        raise ValueError("Only support 'mclust' or 'leiden'")

    cwd = os.path.dirname(__file__)

    script_path = script_path or f"{cwd}/_script/_{method.lower()}.py"
    if not os.path.exists(script_path):
        raise ValueError(f"'{method}' script does not exist")
    save_key = save_key or method.lower()

    cmd = f"python {script_path} {file_path} {save_path} {n_cluster} {batch_key} {hvg_num} {save_key} {cluster_method}"
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