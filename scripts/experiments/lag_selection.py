from pprint import pprint

from codebase.load_data.config import DATASETS
from codebase.lags.base import LagSelectionFromCV, LagSelectionFromData

from codebase.load_data.results import InnerCVReader
from codebase.workflows.config import GROUP

inner_cv = InnerCVReader.read_results()

data_name = 'Tourism'
data_loader = DATASETS[data_name]

df = data_loader.load_data(GROUP)
horizon = data_loader.horizons_map.get(GROUP)
freq_int = data_loader.frequency_map.get(GROUP)
inner_cv = inner_cv.query(f'dataset=="{data_name}"').reset_index(drop=True)

selection_cv = LagSelectionFromCV(inner_cv)
selection_data = LagSelectionFromData(df=df, horizon=horizon, frequency=freq_int)

r1 = selection_cv.predict_lags()
r2 = selection_data.select_by_uid()
r3 = selection_data.select_from_params()

selected_lags = {**r1, **r2, **r3}

pprint(selected_lags)
