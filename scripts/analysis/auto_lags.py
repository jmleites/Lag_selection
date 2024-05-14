import pandas as pd
import plotnine as p9
from neuralforecast.losses.numpy import smape

from codebase.load_data.config import DATASETS_NAMES
from codebase.workflows.config import LAGS

print(DATASETS_NAMES)

BASE_THEME = p9.theme_538(base_family='Palatino', base_size=12) + \
             p9.theme(plot_margin=.001,
                      panel_background=p9.element_rect(fill='white'),
                      plot_background=p9.element_rect(fill='white'),
                      strip_background=p9.element_rect(fill='white'),
                      legend_background=p9.element_rect(fill='white'),
                      legend_title=p9.element_blank(),
                      legend_position='none')

# ds = 'Tourism'

ds_perf, median_perf, es_perf = {}, {}, {}
for ds in DATASETS_NAMES:
    print(ds)
    results = pd.read_csv(f'assets/results/{ds}_outer.csv')

    lags = LAGS[ds]

    predictions = {m: results[f'NHITS_{int(lag)}'] for m, lag in lags.items()}
    predictions = pd.DataFrame(predictions)
    methods = predictions.columns.tolist()

    predictions['unique_id'] = results['unique_id']
    predictions['y'] = results['y']

    overall_perf = {}
    for m in methods:
        overall_perf[m] = smape(y=results['y'], y_hat=predictions[m])

    perf_s = pd.Series(overall_perf)

    results_g = predictions.groupby('unique_id')

    scores = []
    for g, df in results_g:
        sc = {}
        for m in methods:
            sc[m] = smape(y=df['y'], y_hat=df[m])

        scores.append(pd.Series(sc))

    scores_df = pd.DataFrame(scores)
    scores_df.index = [*results_g.groups]

    ds_perf[ds] = perf_s
    median_perf[ds] = scores_df.median()
    es_perf[ds] = scores_df.apply(lambda x: x[x > x.quantile(0.95)].mean())

df = pd.DataFrame(ds_perf).drop(['AIC', 'BIC']).round(4)

df['Average'] = df.mean(axis=1)

df_str = df.round(4).sort_values('Average').astype(str)

print(df_str.to_latex(caption='cap', label='tab:results'))
