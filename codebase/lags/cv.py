import pandas as pd
import numpy as np
from neuralforecast.losses.numpy import smape

from codebase.lags.information_criteria import aic, bic


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

    def get_col_names(self):
        cols_bool = self.cv.columns.str.contains('NHITS')
        col_names = self.cv.columns[cols_bool]

        return col_names
