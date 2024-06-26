import numpy as np

INPUT_RANGE = np.arange(1, 121).tolist()
INPUT_SET = [1, 2, 3, 6, 12, 18, 24, 36, 48, 60, 90, 120]

LAGS = {
    'M3': {'Frequency*2': 24,
           'AIC': 5,
           'Avg. Rank': 63,
           'BIC': 3,
           'Bandara': 23,
           'CV': 17,
           'FNN@0.001': 5,
           'FNN@0.01': 5,
           'Frequency': 12,
           'Horizon': 18,
           'PACF@0.001': 44,
           'PACF@0.01': 24,
           'Previous': 1},
    'Tourism': {'Frequency*2': 24,
                'AIC': 10,
                'Avg. Rank': 43,
                'BIC': 8,
                'Bandara': 23,
                'CV': 43,
                'FNN@0.001': 7,
                'FNN@0.01': 6,
                'Frequency': 12,
                'Horizon': 18,
                'PACF@0.001': 77,
                'PACF@0.01': 29,
                'Previous': 1},
    'Gluonts': {'Frequency*2': 24,
                'AIC': 4,
                'Avg. Rank': 58,
                'BIC': 3,
                'Bandara': 15,
                'CV': 66,
                'FNN@0.001': 4,
                'FNN@0.01': 4,
                'Frequency': 12,
                'Horizon': 8,
                'PACF@0.001': 29,
                'PACF@0.01': 19,
                'Previous': 1}

}
