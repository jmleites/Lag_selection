import os

from dotenv import load_dotenv

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
