import pandas as pd
import plotnine as p9
from neuralforecast.losses.numpy import smape

from codebase.load_data.config import DATASETS_NAMES

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

ds_perf = {}
for ds in DATASETS_NAMES:
    print(ds)
    results = pd.read_csv(f'assets/results/{ds}_outer.csv')

    methods = [f'NHITS_{int(lag)}' for lag in range(1, 121)] + ['SeasonalNaive']
    naming_map = {f'NHITS_{i}': f'{i}' for i in range(1, 121)}

    overall_perf = {}
    for m in methods:
        overall_perf[m] = smape(y=results['y'], y_hat=results[m])

    perf_s = pd.Series(overall_perf)
    perf_s = perf_s.rename(naming_map)

    ds_perf[ds] = perf_s

df = pd.DataFrame(ds_perf).reset_index().melt('index')

df_nhits = df.query('index!="SeasonalNaive"')
df_sn = df.query('index=="SeasonalNaive"')
df_nhits['index'] = pd.Categorical(df_nhits['index'], categories=df_nhits['index'].unique()).astype(int)

plot1 = p9.ggplot(df_nhits,
                  p9.aes(x='index',
                         y='value',
                         fill='variable',
                         group='variable')) + \
        p9.facet_wrap('~variable', nrow=3, scales='free') + \
        p9.theme(plot_margin=0.025,
                 axis_text=p9.element_text(size=12),
                 # axis_text_x=p9.element_blank(),
                 legend_title=p9.element_blank(),
                 legend_position='right') + \
        p9.geom_line(color='black', size=1) + \
        BASE_THEME + \
        p9.labs(x='Number of lags', y='SMAPE') + \
        p9.geom_hline(data=df_sn,
                      mapping=p9.aes(group='variable', yintercept='value'),
                      linetype='dashed',
                      color='red',
                      size=1.1)

# by series
ds = 'Tourism'
results = pd.read_csv(f'assets/results/{ds}_outer.csv')

results_g = results.groupby('unique_id')

scores = []
for g, df in results_g:
    sc = {}
    for m in methods:
        sc[m] = smape(y=df['y'], y_hat=df[m])

    scores.append(pd.Series(sc))

scores_df = pd.DataFrame(scores)
scores_df.index = [*results_g.groups]
scores_df = scores_df.rename(columns=naming_map)

df_ranks = scores_df.rank(axis=1).iloc[:, :60].melt()
df_ranks['variable'] = pd.Categorical(df_ranks['variable'], categories=df_ranks['variable'].unique())

plot2 = p9.ggplot(df_ranks,
                  p9.aes(x='variable',
                         y='value')) + \
        p9.theme(plot_margin=0.025,
                 axis_text=p9.element_text(size=12),
                 # axis_text_x=p9.element_blank(),
                 legend_title=p9.element_blank(),
                 legend_position='right') + \
        p9.geom_boxplot() + \
        BASE_THEME + \
        p9.labs(x='Number of lags', y='SMAPE rank')

plot1.save('assets/plots/plot1.pdf', height=8, width=13)
plot2.save('assets/plots/plot2.pdf', height=6, width=13)
