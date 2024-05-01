import numpy as np

INPUT_RANGE = np.arange(1, 121).tolist()
INPUT_SET = [1, 2, 3, 6, 12, 18, 24, 36, 48, 60, 90, 120]

LAGS = {
    'M3': {'2*Frequency': 24,
           'AIC': 5,
           'Avg. Rank': 63,
           'BIC': 3,
           'Bandara': 23,
           'CV': 17,
           'FNN': 5,
           'FNN@0.01': 5,
           'Frequency': 12,
           'Horizon': 18,
           'PACF': 48,
           'PACF@0.01': 24,
           'Previous': 1},
    'Tourism': {'2*Frequency': 24,
                'AIC': 10,
                'Avg. Rank': 43,
                'BIC': 8,
                'Bandara': 23,
                'CV': 43,
                'FNN': 7,
                'FNN@0.01': 6,
                'Frequency': 12,
                'Horizon': 18,
                'PACF': 98,
                'PACF@0.01': 29,
                'Previous': 1},
    'Gluonts': {'2*Frequency': 24,
                'AIC': 4,
                'Avg. Rank': 58,
                'BIC': 3,
                'Bandara': 15,
                'CV': 66,
                'FNN': 4,
                'FNN@0.01': 4,
                'Frequency': 12,
                'Horizon': 8,
                'PACF': 31,
                'PACF@0.01': 19,
                'Previous': 1}

}
