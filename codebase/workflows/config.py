import numpy as np

INPUT_RANGE = np.arange(1, 121).tolist()
INPUT_SET = [1, 2, 3, 6, 12, 18, 24, 36, 48, 60, 90, 120]

LAGS = {
    'M3': {'aic_avg': 3,
           'aic_max': 22,
           'bandara': 23,
           'bic_avg': 2,
           'bic_max': 14,
           'bl_1': 1,
           'cv_err': 31,
           'cv_rank': 63,
           'fnn0': 5,
           'fnn1': 5,
           'freq_1x': 12,
           'freq_2x': 24,
           'freq_half': 6.0,
           'horizon_1x': 18,
           'horizon_2x': 36,
           'pacf0': 48,
           'pacf1': 24},
    'Tourism': {'aic_avg': 7,
                'aic_max': 24,
                'bandara': 23,
                'bic_avg': 6,
                'bic_max': 17,
                'bl_1': 1,
                'cv_err': 43,
                'cv_rank': 43,
                'fnn0': 7,
                'fnn1': 6,
                'freq_1x': 12,
                'freq_2x': 24,
                'freq_half': 6.0,
                'horizon_1x': 18,
                'horizon_2x': 36,
                'pacf0': 98,
                'pacf1': 29}

}
