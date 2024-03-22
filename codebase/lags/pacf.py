from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa import stattools
from pmdarima.arima import ndiffs




def pacf(self, series: pd.Series, tol=0.01):
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    d_required = ndiffs(series)

    if d_required > 0:
        series = series.diff(periods=d_required)[1:]

    pacf_score = stattools.pacf(series, nlags=self.MAX_K)

    pacf_abs_score = np.abs(pacf_score)

    try:
        k_hat = np.where(pacf_abs_score <= tol)[0][0]
    except IndexError:
        k_hat = self.MAX_K

    predictor_names = self.LAG_NAMES[:k_hat]

    return k_hat, predictor_names