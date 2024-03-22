from datasetsforecast.m3 import M3

from codebase.load_data.base import LoadDataset


class M3Dataset(LoadDataset):
    # DATASET_PATH = './assets/datasets/'
    DATASET_NAME = 'M3'

    horizons = [6, 8, 18]
    frequency = [1, 4, 12]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18
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
        'Quarterly': 'Q',
        'Monthly': 'M'
    }

    data_group = [*horizons_map]

    @classmethod
    def load_data(cls, group):
        ds, *_ = M3.load(cls.DATASET_PATH, group=group)
        return ds
