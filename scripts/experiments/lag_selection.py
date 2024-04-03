import pandas as pd

from codebase.load_data.config import DATASETS
from codebase.lags.false_nearest import false_nearest_neighbors
from codebase.lags.heuristics import bandara_heuristic
from codebase.lags.pacf import pacf_estimation
from codebase.lags.cv import LagSelectionFromCV
from codebase.load_data.base import LoadDataset
from codebase.load_data.results import InnerCVReader

inner_cv = InnerCVReader.read_results()

group = 'Monthly'
data_name = 'M3'
data_loader = DATASETS[data_name]

df = data_loader.load_data(group)
horizon = data_loader.horizons_map.get(group)
freq_str = data_loader.frequency_pd.get(group)
freq_int = data_loader.frequency_map.get(group)
inner_cv = inner_cv.query(f'dataset=="{data_name}"').reset_index(drop=True)

# series_example = df.query('unique_id=="M1"')['y']


selection_cv = LagSelectionFromCV(inner_cv)

avg_err, avg_rank = selection_cv.select_using_uid_err()
aic_avg_min, aic_max_min = selection_cv.select_using_info('aic')
bic_avg_min, bic_max_min = selection_cv.select_using_info('bic')


class LagSelectionFromData:

    def __init__(self, df: pd.DataFrame,
                 horizon: int,
                 frequency: int):
        self.df = df
        self.horizon = horizon
        self.frequency = frequency

    def select_by_uid(self):
        train_df, _ = LoadDataset.train_test_split(self.df, self.horizon)
        train_by_uid = train_df.groupby('unique_id')

        results_by_uid = {}
        for uid, df in train_by_uid:
            # print(uid)
            s = df['y'].reset_index(drop=True)

            bandara = bandara_heuristic(horizon, freq_int)

            fnn0 = false_nearest_neighbors(s, tol=0)
            fnn1 = false_nearest_neighbors(s, tol=0.01)
            fnn2 = false_nearest_neighbors(s, tol=0.001)

            pacf0 = pacf_estimation(s, tol=0)
            pacf1 = pacf_estimation(s, tol=0.01)
            pacf2 = pacf_estimation(s, tol=0.001)

            results_by_uid[uid] = {
                'fnn0': fnn0,
                'fnn1': fnn1,
                'fnn2': fnn2,
                'pacf0': pacf0,
                'pacf1': pacf1,
                'pacf2': pacf2,
                # 'bandara': bandara,
                # 'horizon_1x': horizon,
                # 'horizon_2x': horizon * 2,
                # 'freq_1x': freq_int,
                # 'freq_2x': freq_int * 2,
                # 'freq_half': freq_int / 2,
                # 'bl_1': 1,
                # 'cv_err': avg_err,
                # 'cv_rank': avg_rank,
                # 'aic_avg': aic_avg_min,
                # 'aic_max': aic_max_min,
                # 'bic_avg': bic_avg_min,
                # 'bic_max': bic_max_min,
            }

        return results_by_uid
