import pandas as pd
from neuralforecast.losses.numpy import smape, rmae

from codebase.workflows.config import LAGS

ds = 'M3'
results = pd.read_csv(f'assets/results/{ds}_Monthly_lags.csv')

lags = LAGS[ds]

predictions = {m: results[f'NHITS_{int(lag)}'] for m, lag in lags.items()}
predictions = pd.DataFrame(predictions)

methods = predictions.columns.tolist() + ['SeasonalNaive']

results_all = pd.concat([results, predictions], axis=1)

cols_bool = results_all.columns.str.contains('NHITS')
results_all = results_all.loc[:, ~cols_bool]

# overall error

overall_sc = {}
for k in methods:
    overall_sc[k] = smape(y=results_all['y'], y_hat=results_all[k])
    # sc[k] = rmae(y=df['y'], y_hat1=df[k], y_hat2=df['SeasonalNaive'])

overall_sc = pd.Series(overall_sc)
overall_sc.sort_values()


# by series

results_g = results_all.groupby('unique_id')

scores = []
for g, df in results_g:
    sc = {}
    for k in methods:
        sc[k] = smape(y=df['y'], y_hat=df[k])
        # sc[k] = rmae(y=df['y'], y_hat1=df[k], y_hat2=df['SeasonalNaive'])

    scores.append(pd.Series(sc))

scores_df = pd.DataFrame(scores)
scores_df.index = [*results_g.groups]

scores_df.rank(axis=1).mean().sort_values()

scores_df.boxplot()
scores_df.mean().sort_values()
scores_df.median().sort_values()
scores_df.apply(lambda x: x[x > x.quantile(0.9)].mean()).sort_values()

