from codebase.load_data.m4 import M4Dataset
from codebase.load_data.m3 import M3Dataset
from codebase.load_data.tourism import TourismDataset
from codebase.load_data.gluonts import GluontsDataset

DATASETS = {
    'M4': M4Dataset,
    'M3': M3Dataset,
    'Tourism': TourismDataset,
    'Gluonts': GluontsDataset,
}
