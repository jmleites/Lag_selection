import os

from dotenv import load_dotenv
import pandas as pd

load_dotenv('.env', verbose=True)

DATASET_PATH_ = os.environ.get('DATASET_PATH')


class LoadDataset:
    DATASET_PATH = DATASET_PATH_
    DATASET_NAME = ''

    horizons = []
    frequency = []
    horizons_map = {}
    frequency_map = {}
    context_length = {}
    frequency_pd = {}
    data_group = [*horizons_map]

    @classmethod
    def load_data(cls, group):
        pass

    @staticmethod
    def train_test_split(df: pd.DataFrame):
        df_by_unq = df.groupby('unique_id')

        train_l, test_l = [], []
        for g, df_ in df_by_unq:
            df_ = df_.sort_values('ds')

            train_df_g = df_.head(-horizon)
            test_df_g = df_.tail(horizon)

            train_l.append(train_df_g)
            test_l.append(test_df_g)

        train_df = pd.concat(train_l).reset_index(drop=True)
        test_df = pd.concat(test_l).reset_index(drop=True)

        return train_df, test_df
