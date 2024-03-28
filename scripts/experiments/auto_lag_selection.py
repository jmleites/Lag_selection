import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from codebase.load_data.config import DATASETS

from codebase.lags.false_nearest import false_nearest_neighbors
from codebase.lags.pacf import pacf_estimation

group = 'Monthly'
data_loader = DATASETS['M3']

df = data_loader.load_data(group)
horizon = data_loader.horizons_map.get(group)
n_lags = data_loader.context_length.get(group)
freq_str = data_loader.frequency_pd.get(group)
freq_int = data_loader.frequency_map.get(group)

series_example = df.query('unique_id=="M1"')['y']

false_nearest_neighbors(series_example, tol=0)
false_nearest_neighbors(series_example, tol=0.1)

pacf_estimation(series_example, tol=0)
pacf_estimation(series_example, tol=0.1)

horizon = 12
models = [NHITS(h=horizon,
                input_size=2 * horizon,
                max_steps=100,
                start_padding_enabled=True,
                accelerator='cpu')]
nf = NeuralForecast(models=models, freq='M')
nf.fit(df=df, val_size=horizon)

nf.predict(df=df)

test_index = df['ds'].sort_values().unique()[-12:]
train_df = df.loc[~df['ds'].isin(test_index), :]

df_by_unq = df.groupby('unique_id')
df_list = []
for g, df_ in df_by_unq:
    train_df_g = df_.sort_values('ds').head(-horizon)
    if train_df_g.shape[0] > horizon * 4:
        df_list.append(train_df_g)

train_df = pd.concat(df_list).reset_index(drop=True)


class ModelBasedLagSelection:

    def __init__(self,
                 train: pd.DataFrame,
                 frequency: str,
                 horizon: int):
        self.train = train
        self.frequency = frequency
        self.models = [NHITS(h=horizon,
                             input_size=2 * horizon,
                             start_padding_enabled=True,
                             max_steps=1000,
                             accelerator='cpu')]

        self.cv_preds_single = None

    def get_cv_single_preds(self):
        nf = NeuralForecast(models=self.models, freq='M')
        # nf = NeuralForecast(models=models, freq='M')
        # preds = nf.cross_validation(train_df, test_size=horizon, n_windows=None)
        self.cv_preds_single = nf.cross_validation(self.train, test_size=horizon, n_windows=None)
        # nf.fit(df=self.train)
        # nf.fit(df=train_df)
        # self.insample_predictions = nf.predict_insample()
        # self.insample_predictions = nf.predict()

    @staticmethod
    def aic(self):
        if self.cv_preds_single is None:
            self.get_cv_single_preds()

        pass

    def aic(self):
        if self.cv_preds_single is None:
            self.get_cv_single_preds()

        pass

    def information_criterion(self, criteria: str = 'bic'):
        if self.cv_preds_single is None:
            self.get_cv_single_preds()

        cv_by_ts = cv_preds_single.groupby('unique_id')
        results = {}
        for g, cv_ in cv_by_ts:


        pass

    def cross_validation(self):
        pass


class DataBasedLagSelection:

    @staticmethod
    def bandara_heuristic(horizon, frequency):
        n = max(horizon, frequency)
        n_lags = int(np.ceil(1.25 * n))

        return n_lags
