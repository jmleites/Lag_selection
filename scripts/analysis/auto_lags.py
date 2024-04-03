import pandas as pd
from neuralforecast.losses.numpy import smape, rmae

results = pd.read_csv('assets/results/Tourism_Monthly_lags.csv')

cols_bool = results.columns.str.contains('NHITS')
col_names = results.columns[cols_bool]

results_g = results.groupby('unique_id')

scores = []
for g, df in results_g:
    sc = {}
    for k in col_names:
        sc[k] = smape(y=df['y'], y_hat=df[k])
        # sc[k] = rmae(y=df['y'], y_hat1=df[k], y_hat2=df['SeasonalNaive'])

    scores.append(pd.Series(sc))

scores_df = pd.DataFrame(scores)
scores_df.index = [*results_g.groups]

scores_df.mean()
scores_df.apply(lambda x: x[x > x.quantile(0.9)].mean())
