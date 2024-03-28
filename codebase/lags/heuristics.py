import numpy as np


def bandara_heuristic(horizon, frequency):
    n = max(horizon, frequency)
    n_lags = int(np.ceil(1.25 * n))

    return n_lags
