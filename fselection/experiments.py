import os

import pandas as pd
from sklearn.model_selection import train_test_split
from methods.xgb import XGBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

from workflows import wf_lag_selection
from common.frequency import find_frequency
from utils import save_data, load_data

DATA_DIR = 'data/m4/'
RESULTS_DIR = 'data/results_rf'
datasets = os.listdir(DATA_DIR)

for file in datasets:
    # file = 'ds_1326.csv'
    print(file)
    if file in os.listdir(RESULTS_DIR):
        continue
    else:
        print(f'Running experiments for dataset: {file}')
    #
    data_results = pd.DataFrame()
    # data_results.to_csv(RESULTS_DIR + '/' + file, index=False)
    save_data(data_results, filepath=RESULTS_DIR + '/' + file)
    # df = load_data(RESULTS_DIR + '/' + file)
    #
    series = pd.read_csv(DATA_DIR + '/' + file).V1
    freq = find_frequency(series)
    #
    series = series.diff()[1:].reset_index(drop=True)
    #
    tr, ts = train_test_split(series, test_size=0.2, shuffle=False)
    try:
        k_hat, results_by_method, k_time = \
            wf_lag_selection(y_train=tr,
                             y_test=ts,
                             algorithm=KNeighborsRegressor,
                             frequency=freq)
        #
        ds_results = dict(k_hat=k_hat, results_by_method=results_by_method, k_time=k_time)
        save_data(ds_results, filepath=RESULTS_DIR + '/' + file)
    except ValueError:
        continue

