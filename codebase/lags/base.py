import pandas as pd
import numpy as np
from neuralforecast.losses.numpy import smape

from codebase.lags.information_criteria import aic, bic
from codebase.lags.false_nearest import false_nearest_neighbors
from codebase.lags.heuristics import bandara_heuristic
from codebase.lags.pacf import pacf_estimation
from codebase.load_data.base import LoadDataset


class LagSelectionFromCV:

    def __init__(self, cv):
        self.cv = cv
        self.col_names = self.get_col_names()
        self.uid_scores = None

    def set_error_by_uid(self):
        cv_ = self.cv.groupby('unique_id')

        scores = []
        for g, df in cv_:
            sc = {int(k.split('_')[1]): smape(y=df['y'], y_hat=df[k])
                  for k in self.col_names}

            scores.append(pd.Series(sc))

        scores_df = pd.DataFrame(scores)
        scores_df.index = [*cv_.groups]

        self.uid_scores = scores_df

    def select_using_uid_err(self):
        if self.uid_scores is None:
            self.set_error_by_uid()

        avg_err = self.uid_scores.mean().sort_values().index[0]
        avg_rank = self.uid_scores.rank(axis=1).mean().sort_values().index[0]

        return avg_err, avg_rank

    def select_using_info(self, method='aic'):
        cv_ = self.cv.groupby('unique_id')

        scores = []
        for g, df in cv_:

            sc = {}
            for k in self.col_names:
                k_ = int(k.split('_')[1])

                if method == 'aic':
                    ic = aic(y=df['y'], y_hat=df[k], k=k_)
                else:
                    ic = bic(y=df['y'], y_hat=df[k], k=k_)

                sc[k_] = ic

            scores.append(pd.Series(sc))

        scores_df = pd.DataFrame(scores)
        scores_df.index = [*cv_.groups]

        idxmin_ = scores_df.idxmin(axis=1)

        avg_min, max_min = np.ceil(idxmin_.mean()), idxmin_.max()
        avg_min = int(avg_min)

        return avg_min, max_min

    def predict_lags(self):

        avg_err, avg_rank = self.select_using_uid_err()
        aic_avg_min, aic_max_min = self.select_using_info('aic')
        bic_avg_min, bic_max_min = self.select_using_info('bic')

        vals = {
            'CV': avg_err,
            'Avg. Rank': avg_rank,
            'AIC': aic_avg_min,
            # 'aic_max': aic_max_min,
            'BIC': bic_avg_min,
            # 'bic_max': bic_max_min,
        }

        return vals

    def get_col_names(self):
        cols_bool = self.cv.columns.str.contains('NHITS')
        col_names = self.cv.columns[cols_bool]

        return col_names


class LagSelectionFromData:

    def __init__(self,
                 df: pd.DataFrame,
                 horizon: int,
                 frequency: int):
        self.df = df
        self.horizon = horizon
        self.frequency = frequency

    def select_by_uid(self):
        train_df, _ = LoadDataset.train_test_split(self.df, self.horizon)
        train_by_uid = train_df.groupby('unique_id')

        results_by_uid = {}
        for uid, df_ in train_by_uid:
            # df_ = df.query('unique_id=="M1"')
            s = df_['y'].reset_index(drop=True)

            fnn0 = false_nearest_neighbors(s, tol=0)
            fnn1 = false_nearest_neighbors(s, tol=0.01)

            pacf0 = pacf_estimation(s, tol=0)
            pacf1 = pacf_estimation(s, tol=0.01)

            results_by_uid[uid] = {
                'FNN': fnn0,
                'FNN@0.01': fnn1,
                'PACF': pacf0,
                'PACF@0.01': pacf1,
            }

        results_df = pd.DataFrame(results_by_uid).T

        avg_val = np.ceil(results_df.mean()).astype(int).to_dict()

        return avg_val

    def select_from_params(self):
        bandara = bandara_heuristic(self.horizon, self.frequency)

        vals = {
            'Bandara': bandara,
            'Horizon': self.horizon,
            # 'horizon_2x': self.horizon * 2,
            'Frequency': self.frequency,
            '2*Frequency': self.frequency * 2,
            # 'freq_half': self.frequency / 2,
            'Previous': 1,
        }

        return vals
