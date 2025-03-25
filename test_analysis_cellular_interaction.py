import os
import numpy as np
import pandas as pd
import pickle, argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc


from scipy.sparse import issparse
from scipy import stats

import torch
import pickle
import anndata as ad
from scipy.sparse import csr_matrix
import spatialdm as sdm
import joblib

from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy.stats import spearmanr, pearsonr

import multiprocessing


def Moran_R_std(spatial_W, by_trace=False):
    """Calculate standard deviation of Moran's R under the null distribution.
    """
    N = spatial_W.shape[0]

    if by_trace:
        W = spatial_W.copy()
        H = np.identity(N) - np.ones((N, N)) / N
        HWH = H.dot(W.dot(H))
        var = np.trace(HWH.dot(HWH)) * N**2 / (np.sum(W) * (N-1))**2
    else:
        if issparse(spatial_W):
            nm = N ** 2 * spatial_W.multiply(spatial_W.T).sum() \
                - 2 * N * (spatial_W.sum(0) @ spatial_W.sum(1)).sum() \
                + spatial_W.sum() ** 2
        else:
            nm = N ** 2 * (spatial_W * spatial_W.T).sum() \
                - 2 * N * (spatial_W.sum(1) * spatial_W.sum(0)).sum() \
                + spatial_W.sum() ** 2
        dm = N ** 2 * (N - 1) ** 2
        var = nm / dm

    return np.sqrt(var)


def Moran_R(X, Y, spatial_W, standardise=True, nproc=1):
    """Computing Moran's R for pairs of variables
    
    :param X: Variable 1, (n_sample, n_variables) or (n_sample, )
    :param Y: Variable 2, (n_sample, n_variables) or (n_sample, )
    :param spatial_W: spatial weight matrix, sparse or dense, (n_sample, n_sample)
    :param nproc: default to 1. Numpy may use more without much speedup.
    
    :return: (Moran's R, z score and p values)
    """
    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    if len(Y.shape) < 2:
        Y = Y.reshape(-1, 1)

    if standardise:
        X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
        Y = (Y - np.mean(Y, axis=0, keepdims=True)) / np.std(Y, axis=0, keepdims=True)

    # Consider to dense array for speedup (numpy's codes is optimised)
    if X.shape[0] <= 5000 and issparse(spatial_W):
        # Note, numpy may use unnessary too many threads
        # You may use threadpool.threadpool_limits() outside
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=nproc, user_api='blas'):
            R_val = (spatial_W.A @ X * Y).sum(axis=0) / np.sum(spatial_W)
    else:
        # we assume it's sparse spatial_W when sample size > 5000
        R_val = (spatial_W @ X * Y).sum(axis=0) / np.sum(spatial_W)

    _R_std = Moran_R_std(spatial_W)
    R_z_score = R_val / _R_std
    R_p_val = stats.norm.sf(R_z_score)

    return R_val, R_z_score, R_p_val


def process_pair(pair, adata):
    X = adata[:, pair[0]].X.A
    Y = adata[:, pair[1]].X.A
    R_val, R_z_score, R_p_val = Moran_R(X, Y, adata.obsp['weight'])
    return [pair[0], pair[1], R_val[0], R_z_score[0], R_p_val[0]]


def parallel_process(combinations, adata, num_cpus=48):
    df = pd.DataFrame(columns=['A', 'B', 'R_val', 'R_z_score', 'R_p_val'])
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = list(tqdm(pool.starmap(process_pair, [(pair, adata) for pair in combinations]), total=len(combinations)))
    for result in results:
        df.loc[len(df)] = result
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and worker arguments.")
    parser.add_argument("--dataset_name", "-d", type=str, required=True, help="Name of the dataset to process.")
    parser.add_argument("--num_workers", "-n", type=int, default=64, help="Number of workers for multiprocessing.")
    parser.add_argument("--save_to_path", "-p", type=str, required=False, default=None, help="Path for saving.")
    args = parser.parse_args()

    savepath = "tutorial_analysis/csv_results" if args.save_to_path is None else args.save_to_path
    dataset_name = args.dataset_name
    num_workers = args.num_workers
    # dataset_name = "humanlung_cell2location"
    # dataset_name = "her2st"

    cell_types = joblib.load(f"example_data/{dataset_name}/cell_types.pkl")
    combinations = []
    for i in range(len(cell_types)):
        for j in range(i+1, len(cell_types)):
            combinations.append((cell_types[i], cell_types[j]))

    root_path = "results/"
    file_name = "all_slides_test_spot_predictions.pkl"

    mapping_exp={
    'CUCA': "CUCA_virchow2_BN_RMSE_a0.3b0.6_ep100_bs128_lr0.002_OneCycleLR",
    'Hist2Cell': "hist2cell_resnet18_BN_MSE_a0.6b0.3_ep10_bs16_lr0.0001_CosineAnnealingLR",
    'LinearProbing': "LinearProbing_virchow2_RMSE_a0.b0._ep100_bs128_lr0.002_1",
    'HisToGene': "HisToGene_resnet18_MSE_a0.6b0.3_ep100_bs1_lr0.00001_scheNoAdjust",
    'THItoGene': "THItoGene_resnet18_MSE_a0.6b0.3_ep200_bs1_lr0.00001_scheNoAdjust",
    'ST-Net': "ST-Net_densenet121_RMSE_a0.b0._ep50_bs32_sgdlr0.01_NoAdjust",
    }
    mapping_color = {
        'CUCA': "#C7C826", 'LinearProbing': "#799f80", "Hist2Cell": "#F4CEDD",
                    "HisToGene":  "#B6C9D8", "THItoGene": "#CD79C0", "PPEG": "#dd91b8", "ST-Net": "#C5B5E6"}

    mapping_exp['CUCA'] = "CUCA_virchow2_BN_RMSE_a0.2b0.7_ep100_bs128_lr0.002_ext0" if dataset_name == "stnet" else "CUCA_virchow2_BN_RMSE_a0.3b0.6_ep100_bs128_lr0.002_OneCycleLR"


    all_cosine_sim = {}
    all_pearson_correlation = {}
    all_spearman_correlation = {}
    for method in mapping_exp.keys():
        with open(os.path.join(root_path, dataset_name, mapping_exp[method], file_name), 'rb') as f:
            Predictions = pickle.load(f)

        one_method_all_slide_cosine_sim  = {}
        one_method_all_slide_pearson_correlation = {}
        one_method_all_slide_spearman_correlation = {}
        for slide in Predictions.keys():
            print(f"Method: {method}, Slide: {slide}")
            X = csr_matrix(Predictions[slide]['cell_abundance_labels'])
            adata = ad.AnnData(X, obsm={"spatial": Predictions[slide]['coords']})
            adata.var_names = cell_types

            sdm.weight_matrix(adata, l=500, cutoff=0.2, single_cell=False, n_neighbors=160) 

            df_label_raw = parallel_process(combinations, adata, num_cpus=num_workers)

            X = csr_matrix(Predictions[slide]['cell_abundance_predictions'])
            adata = ad.AnnData(X, obsm={"spatial": Predictions[slide]['coords']})
            adata.var_names = cell_types

            sdm.weight_matrix(adata, l=500, cutoff=0.2, single_cell=False, n_neighbors=160) 

            df_pred_raw = parallel_process(combinations, adata, num_cpus=num_workers)

            # 计算余弦相似度
            cosine_similarity = ((1 - cosine(df_label_raw.R_val, df_pred_raw.R_val)) + 1)/ 2 # 余弦相似度，归一化到0-1之间
            print(f"Cosine Similarity: {cosine_similarity}")

            one_method_all_slide_cosine_sim[slide] = cosine_similarity


            # Calculate Pearson correlation coefficient
            pcc, _ = pearsonr(df_label_raw['R_val'], df_pred_raw['R_val'])
            print(f"Pearson Correlation Coefficient: {pcc}")
            one_method_all_slide_pearson_correlation[slide] = pcc

            spearman, p_val = spearmanr(df_label_raw['R_val'], df_pred_raw['R_val'])
            print(f"Spearman Correlation Coefficient: {spearman}")
            one_method_all_slide_spearman_correlation[slide] = spearman

        all_cosine_sim[method] = one_method_all_slide_cosine_sim
        all_pearson_correlation[method] = one_method_all_slide_pearson_correlation
        all_spearman_correlation[method] = one_method_all_slide_spearman_correlation

    pd.DataFrame(all_cosine_sim).to_csv(f"{savepath}/revision_cellular_intera_{dataset_name}_cosine_sim.csv")
    pd.DataFrame(all_pearson_correlation).to_csv(f"{savepath}/revision_cellular_intera_{dataset_name}_pearson_correlation.csv")
    pd.DataFrame(all_spearman_correlation).to_csv(f"{savepath}/revision_cellular_intera_{dataset_name}_spearman_correlation.csv")    