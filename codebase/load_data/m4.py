from datasetsforecast.m4 import M4

from codebase.load_data.base import LoadDataset


class M4Dataset(LoadDataset):
    DATASET_NAME = 'M4'

    horizons = [6, 8, 18, 13, 14, 48]
    frequency = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48,
    }

    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24,
    }

    context_length = {
        'Yearly': 1,
        'Quarterly': 8,
        'Monthly': 60,
        'Weekly': 26,
        'Daily': 31,
        'Hourly': 72,
    }

    frequency_pd = {
        'Yearly': 'Y',
        'Quarterly': 'Q',
        'Monthly': 'M',
        'Weekly': 'W',
        'Daily': 'D',
        'Hourly': 'H',
    }

    data_group = [*horizons_map]

    @classmethod
    def load_data(cls, group):
        ds, *_ = M4.load(cls.DATASET_PATH, group=group)
        ds['ds'] = ds['ds'].astype(int)
        return ds
