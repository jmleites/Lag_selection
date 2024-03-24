import os

import pandas as pd
import numpy as np

from codebase.load_data.base import LoadDataset


class TourismDataset(LoadDataset):
    DATASET_PATH = os.path.abspath('../../assets/datasets/tourism/')
    DATASET_NAME = 'T'

    horizons = [4, 8, 24]
    frequency = [1, 4, 12]
    horizons_map = {
        'Yearly': 4,
        'Quarterly': 8,
        'Monthly': 24
    }

    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12
    }

    context_length = {
        'Yearly': 1,
        'Quarterly': 8,
        'Monthly': 60
    }

    frequency_pd = {
        'Yearly': 'Y',
        'Quarterly': 'QS',
        'Monthly': 'MS'
    }

    data_group = [*horizons_map]

    @classmethod
    def load_data(cls, group):

        assert group in cls.data_group

        ds = {}
        train = pd.read_csv(os.path.join(cls.DATASET_PATH, f'{group.lower()}_in.csv'),
                            header=0, delimiter=",")
        test = pd.read_csv(os.path.join(cls.DATASET_PATH, f'{group.lower()}_oos.csv'),
                           header=0, delimiter=",")

        if group == 'Yearly':
            train_meta = train[:2]
            meta_length = train_meta.iloc[0].astype(int)
            test = test[2:].reset_index(drop=True).T
            train = train[2:].reset_index(drop=True).T
        else:
            train_meta = train[:3]
            meta_length = train_meta.iloc[0].astype(int)
            test = test[3:].reset_index(drop=True).T
            train = train[3:].reset_index(drop=True).T

        train_set = [ts[:ts_length] for ts, ts_length in zip(train.values, meta_length)]
        test_set = [ts[:ts_length] for ts, ts_length in zip(test.values, meta_length)]

        for i, idx in enumerate(train.index):
            ds[idx] = np.concatenate([train_set[i], test_set[i]])

        max_len = np.max([len(x) for k, x in ds.items()])
        idx = pd.date_range(end=pd.Timestamp('2023-11-01'),
                            periods=max_len,
                            freq=cls.frequency_pd[group])

        ds = {k: pd.Series(series, index=idx[-len(series):]) for k, series in ds.items()}
        df = pd.concat(ds, axis=1)
        df = df.reset_index().melt('index').dropna().reset_index(drop=True)
        df.columns = ['ds', 'unique_id', 'y']

        return df
