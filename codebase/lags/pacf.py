import numpy as np
import pandas as pd
from statsmodels.tsa import stattools
from pmdarima.arima import ndiffs


def pacf_estimation(series: pd.Series, tol=0.01, max_lags: int = 100):
    d_required = ndiffs(series)

    if d_required > 0:
        series = series.diff(periods=d_required)[1:]

    if max_lags > len(series)/2:
        max_lags = int(len(series)/2 - 1)

    pacf_score = stattools.pacf(series, nlags=max_lags)
    pacf_abs_score = np.abs(pacf_score)

    try:
        n_lags = np.where(pacf_abs_score <= tol)[0][0]
    except IndexError:
        n_lags = max_lags

    return n_lags
