
import os
import h5py
import time
import torch
import random
import numpy as np
import scanpy as sc
import time
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
import pickle, os, numbers

import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import scipy

def read_dataset(adata, transpose=False, test_split=False, copy=False):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata

def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

# from munkres import Munkres
import os
import sys
import math
import scanpy as sc
from scipy import stats, spatial, sparse
from scipy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
from sklearn.neighbors import kneighbors_graph

def geneSelection(data, threshold=0, atleast=10, 
                  yoffset=.02, xoffset=5, decay=1.5, n=None, 
                  plot=True, markers=None, genes=None, figsize=(6,3.5),
                  markeroffsets=None, labelsize=10, alpha=1, verbose=1):
    
    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data>threshold).mean(axis=0)))
        A = data.multiply(data>threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:,detected].mean(axis=0))) / (1-zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data>threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:,detected]>threshold
        logs = np.zeros_like(data[:,detected]) * np.nan
        logs[mask] = np.log2(data[:,detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    lowDetection = np.array(np.sum(data>threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan
            
    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low)/2
            else:
                low = xoffset
                xoffset = (xoffset + up)/2
        if verbose>0:
            print('Chosen offset: {:.2f}'.format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
                
    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold>0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1]+.1,.1)
        y = np.exp(-decay*(x - xoffset)) + yoffset
        if decay==1:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-x+{:.2f})+{:.2f}'.format(np.sum(selected),xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)
        else:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}'.format(np.sum(selected),decay,xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)

        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        xy = np.concatenate((np.concatenate((x[:,None],y[:,None]),axis=1), np.array([[plt.xlim()[1], 1]])))
        t = plt.matplotlib.patches.Polygon(xy, color=sns.color_palette()[1], alpha=.4)
        plt.gca().add_patch(t)
        
        plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
        if threshold==0:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of zero expression')
        else:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of near-zero expression')
        plt.tight_layout()
        
        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num,g in enumerate(markers):
                i = np.where(genes==g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color='k')
                dx, dy = markeroffsets[num]
                plt.text(meanExpr[i]+dx+.1, zeroRate[i]+dy, g, color='k', fontsize=labelsize)
    
    return selected

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from _scmdc_batch import scMultiClusterBatch
from utils import search_resolution, h5_to_h5ad

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('n_cluster', type=int)
    parser.add_argument('rna_file_path', type=str)
    parser.add_argument('atac_file_path', type=str)
    parser.add_argument('adt_file_path', type=str)
    parser.add_argument('save_path', type=str)
    parser.add_argument('save_key', type=str)
    parser.add_argument('hvg_num', type=int)
    parser.add_argument('batch_key', type=str)
    
    parser.add_argument('--n_clusters', default=27, type=int)
    parser.add_argument('--cutoff', default=0.5, type=float, help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='Normalized_filtered_BMNC_GSE128639_Seurat.h5')
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--gamma', default=.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--tau', default=1., type=float,
                        help='weight of clustering loss')
    parser.add_argument('--phi1', default=0.001, type=float,
                        help='coefficient of KL loss in pretraining stage')
    parser.add_argument('--phi2', default=0.001, type=float,
                        help='coefficient of KL loss in clustering stage')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--lr', default=1., type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/')
    parser.add_argument('--ae_weight_file', default='AE_weights_1.pth.tar')
    parser.add_argument('--resolution', default=0.2, type=float)
    parser.add_argument('--n_neighbors', default=30, type=int)
    parser.add_argument('--embedding_file', action='store_true', default=False)
    parser.add_argument('--prediction_file', action='store_true', default=False)
    parser.add_argument('-el','--encodeLayer', nargs='+', default=[256,64,32,16])
    parser.add_argument('-dl1','--decodeLayer1', nargs='+', default=[16,64,256])
    parser.add_argument('-dl2','--decodeLayer2', nargs='+', default=[16,20])
    parser.add_argument('--sigma1', default=2.5, type=float)
    parser.add_argument('--sigma2', default=1.5, type=float)
    parser.add_argument('--f1', default=1000, type=float, help='Number of mRNA after feature selection')
    parser.add_argument('--f2', default=2000, type=float, help='Number of ADT/ATAC after feature selection')
    parser.add_argument('--filter1', action='store_true', default=False, help='Do mRNA selection')
    parser.add_argument('--filter2', action='store_true', default=False, help='Do ADT/ATAC selection')
    parser.add_argument('--nbatch', default=1, type=int)
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--path1', metavar='DIR', default=[], nargs='+',  help='path to RNA')
    parser.add_argument('--path2', metavar='DIR', default=[], nargs='+',  help='path to ADT/ATAC')
    parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
    args = parser.parse_args()
    print(args)
    n_cluster = int(args.n_cluster)
    rna_file_path = args.rna_file_path
    atac_file_path = args.atac_file_path
    adt_file_path = args.adt_file_path
    save_path = args.save_path
    save_key = args.save_key
    hvg_num = int(args.hvg_num)
    batch_key = args.batch_key

    begin_time = time.time()

    def read_h5_file(file_paths):
        rna_path=file_paths['modality1_path']
        adt_path=file_paths['modality2_path']
        
        # read rna
        adata1 = h5_to_h5ad(rna_path, batch_key=batch_key)
        adata2 = h5_to_h5ad(adt_path, batch_key=batch_key)
        adata1.X = adata1.X.toarray().astype(float)
        adata2.X = adata2.X.toarray().astype(float)

        if batch_key in adata1.obs:
            all_batch = []
            batch = []
            for i in adata1.obs[batch_key]:
                if i not in all_batch:
                    all_batch.append(i)
            cnt = 0
            for i in all_batch:
                batch.append(np.ones((adata1[adata1.obs[batch_key]==i].shape[0], 1))  + cnt)
                cnt = cnt + 1
            batch = np.concatenate(batch,0)
            args.nbatch = len(set(adata1.obs[batch_key]))
        else:
            batch = np.ones((adata1.shape[0],1))
        
        return adata1, adata2, batch
                    

    file_paths = {
        "modality1_path": rna_file_path,
        "modality2_path": atac_file_path if atac_file_path!="NULL" else adt_file_path
    }
    
    adata1, adata2, b = read_h5_file(file_paths)
    enc = OneHotEncoder()
    enc.fit(b.reshape(-1, 1))
    B = enc.transform(b.reshape(-1, 1)).toarray()

    # #Gene filter
    # if args.filter1:
    #     importantGenes = geneSelection(x1, n=args.f1, plot=False)
    #     x1 = x1[:, importantGenes]
    # if args.filter2:
    #     importantGenes = geneSelection(x2, n=args.f2, plot=False)
    #     x2 = x2[:, importantGenes]

    # preprocessing scRNA-seq read counts matrix

    #adata1.obs['Group'] = y

    adata1 = read_dataset(adata1,
                     transpose=False,
                     test_split=False,
                     copy=True)
    adata1.layers["counts"] = adata1.X.copy()
    sc.pp.highly_variable_genes(adata1, n_top_genes=hvg_num, flavor="seurat_v3", subset=True, layer="counts")


    adata1 = normalize(adata1,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    

    #adata2.obs['Group'] = y
    adata2 = read_dataset(adata2,
                     transpose=False,
                     test_split=False,
                     copy=True)
    adata2.layers["counts"] = adata2.X.copy()

    if atac_file_path!="NULL":
        sc.pp.highly_variable_genes(adata2, n_top_genes=hvg_num*10, flavor="seurat_v3", subset=True, layer="counts")
        adata2 = normalize(adata2,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    else:
        adata2 = normalize(adata2,
                      size_factors=False,
                      normalize_input=False,
                      logtrans_input=False)

    
    input_size1 = adata1.n_vars
    input_size2 = adata2.n_vars
    
    print(args)
    
    encodeLayer = list(map(int, args.encodeLayer))
    decodeLayer1 = list(map(int, args.decodeLayer1))
    decodeLayer2 = list(map(int, args.decodeLayer2))
    
    model = scMultiClusterBatch(input_dim1=input_size1, input_dim2=input_size2, n_batch = args.nbatch, tau=args.tau,
                        encodeLayer=encodeLayer, decodeLayer1=decodeLayer1, decodeLayer2=decodeLayer2,
                        activation='elu', sigma1=args.sigma1, sigma2=args.sigma2, gamma=args.gamma,
                        cutoff = args.cutoff, phi1=args.phi1, phi2=args.phi2, device=args.device).to(args.device)
    
    print(str(model))
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # raise ValueError(
    #     f"X1: {adata1.X.shape}, X_raw1: {adata1.raw.X.shape}, sf1: {adata1.obs.size_factors.shape}, X2: {adata2.X.shape}, X_raw2: {adata2.raw.X.shape}, sf2: {adata2.obs.size_factors.shape}, B: {B.shape}"
    # )
    if args.ae_weights is None:
        model.pretrain_autoencoder(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors, 
                X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, B = B, batch_size=args.batch_size, 
                epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    
    latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device), torch.tensor(B).to(args.device), batch_size=args.batch_size)
    latent = latent.cpu().numpy()
    if args.n_clusters == -1:
       n_clusters = GetCluster(latent, res=args.resolution, n=args.n_neighbors)
    else:
       print("n_cluster is defined as " + str(args.n_clusters))
       n_clusters = args.n_clusters

    final_latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device), torch.tensor(B).to(args.device), batch_size=args.batch_size)
    final_latent = final_latent.cpu().numpy()
    end_time = time.time()
    all_time = end_time - begin_time

    adata1.obsm["latent"] = final_latent
    latent = pd.DataFrame(final_latent, index=adata1.obs_names)
    latent.to_csv(os.path.join(save_path, save_key + "_latent.csv"))

    sc.pp.neighbors(adata1, use_rep="latent")
    sc.tl.umap(adata1)
    res = search_resolution(adata1, fixed_clus_count=n_cluster)
    sc.tl.leiden(adata1, resolution=res, key_added="cluster")

    ## save UMAP
    umap = pd.DataFrame(adata1.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata1.obs_names)
    umap.insert(2, "cluster", adata1.obs['cluster'].values)
    umap.to_csv(os.path.join(save_path, save_key + ".csv"))