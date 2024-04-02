import pandas as pd
from neuralforecast.losses.numpy import smape
from codebase.lags.information_criteria import aic, bic

inner_cv = pd.read_csv('/Users/vcerq/Dropbox/Research/feature_engineering/lag_size/assets/results/M3_Monthly_inner.csv')

nhits_cols = inner_cv.columns.str.contains('NHITS')

col_names = inner_cv.columns[nhits_cols]

inner_cv_g = inner_cv.groupby('unique_id')

scores = []
for g, df in inner_cv_g:
    print(g)
    sc = {int(k.split('_')[1]): smape(y=df['y'], y_hat=df[k])
          for k in col_names}

    scores.append(pd.Series(sc))

scores_df = pd.DataFrame(scores)
scores_df.index = [*inner_cv_g.groups]
scores_df.rank(axis=1).mean()

scores_df.mean().idxmin()

scores = []
for g, df in inner_cv_g:
    print(g)

    sc = {}
    for k in col_names:
        k_ = int(k.split('_')[1])

        aic_sc = aic(y=df['y'], y_hat=df[k], k=k_)
        aic_sc = bic(y=df['y'], y_hat=df[k], k=k_)

        sc[k_] = aic_sc

    scores.append(pd.Series(sc))

scores_df = pd.DataFrame(scores)
scores_df.index = [*inner_cv_g.groups]

scores_df.idxmin(axis=1)
scores_df.idxmin(axis=1).mean()
scores_df.rank(axis=1).mean().sort_values()


class LagSelectionFromCV:

    def __init__(self, cv):
        self.cv = cv

    def calc_error_by_uid(self):
        pass

    def select_using_info(self, method='aic'):
        pass