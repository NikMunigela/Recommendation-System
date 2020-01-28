import os
import time

import numpy as np
import pandas as pd

from preprocessing.clean import CleanData
from svd.svd_algorithm import SVDAlgorithm
from error_measures.measures import *
from cur.cur_algorithm import *
from collaborative_filtering.collaborate import *
from LF.latent import *

def format_dataset():
    for file in os.listdir('preprocessing/'):
        if str(file).endswith('ml-100k'):
            print("Dataset exists.")
            cleaner = CleanData('preprocessing/ml-100k/ml-100k/u.data')
            cleaner.process()
            break
        elif os.listdir('preprocessing/').index(file) == len(os.listdir('preprocessing/')) - 1:
            print("Dataset doesn't exist. Run 'run.sh' again.")


def run_collaborative_filtering(M):
    start = time.time()
    m = M[300:350, 150:200].T
    cf = Collaborate(m.T)
    m_p = cf.fill()
    print("Collaborative Filtering Time: " +str(time.time() - start))
    print("RMSE Collaborative Filtering: " + str(rmse(m, m_p.T)))
    print("MAE Collaborative Filtering: " + str(mae(m, m_p.T)))

def run_collaborative_filtering_baseline(M):
    start = time.time()
    m = M[300:350, 150:200]
    cfb = Collaborate(m.T)
    m_p = cfb.fill(baseline=True)
    print("Collaborative Filtering with baseline Time: " +str(time.time() - start))
    print("MAE Collaborative Filtering with baseline: " + str(mae(m, m_p.T)))
    print("RMSE Collaborative Filtering with baseline: " + str(rmse(m, m_p.T)))

def run_svd(M):
    s = SVDAlgorithm()
    svd_start = time.time()
    U, sigma, V = s.svd(M, dimension_reduction=1.0)
    M_p = np.dot(np.dot(U, sigma), V)
    print("SVD Time: " +str(time.time() - svd_start))
    print("MAE SVD: " + str(mae(M, M_p)))
    print("RMSE SVD: " + str(rmse(M, M_p)))

def run_svd_reduce(M):
    s = SVDAlgorithm()
    svd_reduce_start = time.time()
    U, sigma, V = s.svd(M, dimension_reduction=0.9)
    M_p = np.dot(np.dot(U, sigma), V)
    print("SVD Reduction Time: " +str(time.time() - svd_reduce_start))
    print("RMSE Reduction SVD: " + str(rmse(M, M_p)))
    print("MAE Reduction  SVD: " + str(mae(M, M_p)))

def run_cur(M):
    cur_start = time.time()
    M_p = cur(M, 600, 600, repeat=False)
    print("CUR Time: " +str(time.time() - cur_start))
    print("RMSE CUR: " + str(rmse(M, M_p)))
    print("MAE  CUR: " + str(mae(M, M_p)))

def run_cur_reduce(M):
    cur_reduce_start = time.time()
    M_p = cur(M, 600, 600, dim_red=0.9, repeat=True)
    print("CUR Reduction Time: " +str(time.time() - cur_reduce_start))
    print("RMSE Reduction CUR: " + str(rmse(M, M_p)))
    print("MAE Reduction CUR: " + str(mae(M, M_p)))

def latent(M):
    latent_start = time.time()
    u,v=matrix_factorization(M, dimensions=10)
    M_p = np.dot(u,v)
    print("LATENT Reduction Time: " +str(time.time() - latent_start))
    print("RMSE Reduction LATENT: " + str(rmse(M, M_p)))
    print("MAE Reduction LATENT: " + str(mae(M, M_p)))


if __name__=="__main__":
    formated_dataset = False
    for files in os.listdir('.'):
        if str(files).endswith('.npy') or str(files).endswith('.csv'):
            print("Formatted dataset already exists.")
            formated_dataset = True
            break
    if formated_dataset is False:
        format_dataset()
    M = np.load('data.npy')
    run_collaborative_filtering(M)
    run_collaborative_filtering_baseline(M)
    run_svd(M)
    run_svd_reduce(M)
    run_cur(M)
    run_cur_reduce(M)
