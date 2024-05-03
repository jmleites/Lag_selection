import pandas as pd


def time_delay_embedding(x: pd.Series, n_lags: int, return_X_y: bool):
    series_as_df = []
    for i in range(n_lags, 0, -1):
        shifted_series = x.shift(i)
        shifted_series.name = f't-{i - 1}' if i != 1 else 't'
        series_as_df.append(shifted_series)

    series_as_df = pd.concat(series_as_df, axis=1).dropna().reset_index(drop=True)
    series_as_df.index = x.tail(series_as_df.shape[0]).index

    target = x.tail(series_as_df.shape[0])
    if return_X_y:
        return series_as_df, target
    else:
        series_as_df['t+1'] = target
        return series_as_df
