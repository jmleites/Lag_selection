from datasetsforecast.m3 import M3

from codebase.load_data.base import LoadDataset


class M3Dataset(LoadDataset):
    DATASET_NAME = 'M3'

    horizons_map = {
        'Monthly': 18
    }

    frequency_map = {
        'Monthly': 12
    }

    context_length = {
        'Monthly': 24
    }

    frequency_pd = {
        'Monthly': 'M'
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls, group):
        ds, *_ = M3.load(cls.DATASET_PATH, group=group)
        return ds
