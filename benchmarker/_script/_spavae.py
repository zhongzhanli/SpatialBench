import sys
sys.path.append("../../Benchmark/external/spaVAE/spaVAE_Batch")
from time import time
import torch
from spaVAE_Batch import SPAVAE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scanpy as sc
import pandas as pd
from preprocess import normalize
import scipy.sparse as sp
from utils import search_resolution, mclust, list_to_onehot

args = sys.argv

file_path = args[1]
save_path = args[2]
n_cluster = int(args[3])
batch_key =  args[4]
hvg_num = int(args[5])
save_key = args[6]
cluster_method = args[7]

if __name__ == "__main__":


    class Args(object):
        def __init__(self):
            self.data_file = ''
            self.select_genes = 0
            self.batch_size = "auto"
            self.maxiter = 5000
            self.train_size = 0.95
            self.patience = 200
            self.lr = 1e-3
            self.weight_decay = 1e-6
            self.noise = 0
            self.dropoutE = 0
            self.dropoutD = 0
            self.encoder_layers = [128, 64]
            self.GP_dim = 2
            self.Normal_dim = 8
            self.decoder_layers = [128]
            self.init_beta = 10
            self.min_beta = 4
            self.max_beta = 25
            self.KL_loss = 0.025
            self.num_samples = 1
            self.shared_dispersion = False
            self.fix_inducing_points = True
            self.grid_inducing_points = True
            self.inducing_point_steps = 8
            self.inducing_point_nums = None
            self.fixed_gp_params = False
            self.loc_range = 20.
            self.kernel_scale = 20.
            self.allow_batch_kernel_scale = True
            self.model_file = "model.pt"
            self.final_latent_file = "final_latent.txt"
            self.denoised_counts_file = "denoised_counts.txt"
            self.num_denoise_samples = 10000
            self.device = "cuda:1"
            self.dynamicVAE = True

    args = Args()

    adata = sc.read_h5ad(file_path)

    if(not sp.issparse(adata.X)):
        adata.X = sp.csr_matrix(adata.X)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_num, subset=True, batch_key=batch_key)

    x = adata.X.toarray().astype('float64')
    obs_names = list(adata.obs_names)
    loc = np.array(adata.obsm["spatial"]).astype('float64')
    batch_list = list(adata.obs[batch_key])
    batch, _ = list_to_onehot(batch_list)
    batch = batch.astype('float64')

    if args.batch_size == "auto":
        if x.shape[0] <= 1024:
            args.batch_size = 128
        elif x.shape[0] <= 2048:
            args.batch_size = 256
        else:
            args.batch_size = 512
    else:
        args.batch_size = int(args.batch_size)
        
    print(args)

    n_batch = batch.shape[1]

    # scale locations per batch
    loc_scaled = np.zeros(loc.shape, dtype=np.float64)
    for i in range(n_batch):
        scaler = MinMaxScaler()
        b_loc = loc[batch[:,i]==1, :]
        b_loc = scaler.fit_transform(b_loc) * args.loc_range
        loc_scaled[batch[:,i]==1, :] = b_loc
    loc = loc_scaled

    loc = np.concatenate((loc, batch), axis=1)

    # build inducing point matrix with batch index
    eps = 1e-5
    initial_inducing_points_0_ = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
    initial_inducing_points_0 = np.tile(initial_inducing_points_0_, (n_batch, 1))
    initial_inducing_points_1 = []
    for i in range(n_batch):
        initial_inducing_points_1_ = np.zeros((initial_inducing_points_0_.shape[0], n_batch))
        initial_inducing_points_1_[:, i] = 1
        initial_inducing_points_1.append(initial_inducing_points_1_)
    initial_inducing_points_1 = np.concatenate(initial_inducing_points_1, axis=0)
    initial_inducing_points = np.concatenate((initial_inducing_points_0, initial_inducing_points_1), axis=1)
    print(initial_inducing_points.shape)

    adata = sc.AnnData(x, dtype="float64")

    adata = normalize(adata,
                    size_factors=True,
                    normalize_input=True,
                    logtrans_input=True,
                    filter_min_counts=False)
    # raise ValueError(f"{adata.X.shape}\n{loc.shape}\n{batch.shape}\n{adata.raw.X.shape}\n{adata.obs.size_factors.shape}")
    

    model = SPAVAE(input_dim=adata.n_vars, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim,
                   n_batch=n_batch, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
                   noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
                   shared_dispersion=args.shared_dispersion,
                   fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
                   fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale,
                   allow_batch_kernel_scale=args.allow_batch_kernel_scale,
                   N_train=adata.n_obs, KL_loss=args.KL_loss, init_beta=args.init_beta, min_beta=args.min_beta,
                   max_beta=args.max_beta, dtype=torch.float64, device=args.device, dynamicVAE=args.dynamicVAE)
    print(str(model))


    t0 = time()

    model.train_model(pos=loc, ncounts=adata.X, raw_counts=adata.raw.X, size_factors=adata.obs.size_factors, batch=batch,
                      lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                      train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True,
                      model_weights=args.model_file)
    print('Training time: %d seconds.' % int(time() - t0))

    final_latent = model.batching_latent_samples(X=loc, Y=adata.X, B=batch, batch_size=args.batch_size)

    # denoised_counts = model.batching_denoise_counts(X=loc, Y=adata.X, B=batch, batch_size=args.batch_size, n_samples=25)
    
    adata_latent = sc.AnnData(final_latent)
    adata_latent.obs_names = obs_names
    adata_latent.obs[batch_key] = batch_list
    sc.pp.neighbors(adata_latent, n_neighbors=20, use_rep="X")
    sc.tl.umap(adata_latent)
    if(cluster_method=="mclust"):
        adata_latent.obs["cluster"] = mclust(adata_latent.X, n_cluster=n_cluster)
    else:
        res = search_resolution(adata_latent, fixed_clus_count=n_cluster)
        sc.tl.leiden(adata_latent, resolution=res, key_added="cluster")
    
    embed = pd.DataFrame(adata_latent.X)
    embed.index = list(adata_latent.obs_names)
    embed.to_csv(save_path+f"/embed_{save_key}.csv")
    umap = pd.DataFrame(adata_latent.obsm["X_umap"])
    umap.index = list(adata_latent.obs_names)
    umap.to_csv(save_path+"/umap_"+save_key+".csv")
    batch = pd.DataFrame(adata_latent.obs[batch_key])
    batch.index = list(adata_latent.obs_names)
    batch.to_csv(save_path+"/batch_"+save_key+".csv")
    cluster = pd.DataFrame(adata_latent.obs["cluster"])
    cluster.index = list(adata_latent.obs_names)
    cluster.to_csv(save_path+"/cluster_"+save_key+".csv")
