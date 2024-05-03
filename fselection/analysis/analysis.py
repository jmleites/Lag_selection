import os

import pandas as pd

from utils import load_data

# DATA_DIR = '/Users/vcerqueira/Desktop/results_xgbt/'
DATA_DIR = '/Users/vcerqueira/Desktop/results_rf/'
# DATA_DIR = '/Users/vcerqueira/Desktop/results_lasso/'

files = os.listdir(DATA_DIR)

# h = [1, 7, 14][2]
h = 1
algorithm = 'LASSO'

error_list = []
exec_time_list = []
k_hat_list = []
for file_name in files:
    print(file_name)
    # file_name = 'ds_2945.csv'
    file_data = load_data(f'{DATA_DIR}{file_name}')

    try:
        k_hat, results, k_time = file_data.values()
    except TypeError:
        continue

    mae_err = pd.Series(results)

    error_list.append(mae_err)
    exec_time_list.append(k_time)
    k_hat_list.append(k_hat)

error_df = pd.concat(error_list, axis=1).T
error_df.to_csv(f'data/FSELECTION_ERROR_{algorithm}_{h}.csv', index=False)
# error_df.rank(axis=1).mean()
exec_df = pd.DataFrame(exec_time_list)
exec_df.to_csv(f'data/FSELECTION_EXECT_{algorithm}_{h}.csv', index=False)
k_df = pd.DataFrame(k_hat_list)
k_df.to_csv(f'data/FSELECTION_KHAT_{algorithm}_{h}.csv', index=False)

error_df.rank(axis=1).mean().sort_values()
df=error_df.loc[error_df.rank(axis=1)['bl_2']> 2,:]
df=error_df.loc[error_df.rank(axis=1)['freq']< 3,:]
df=error_df.loc[k_df['freq'] != 2,:]
df.rank(axis=1).mean().sort_values()


error_df2=error_df.copy()

df = pd.concat([error_df,error_df2],axis=0)