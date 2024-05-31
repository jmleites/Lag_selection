from pprint import pprint

from codebase.load_data.config import DATASETS
from codebase.lags.base import LagSelectionFromCV, LagSelectionFromData

from codebase.load_data.results import InnerCVReader

inner_cv = InnerCVReader.read_results()

data_name = 'Tourism'
group = 'Monthly' if data_name != 'Gluonts' else 'm1_monthly'
data_loader = DATASETS[data_name]

df = data_loader.load_data(group)
horizon = data_loader.horizons_map.get(group)
freq_int = data_loader.frequency_map.get(group)
inner_cv = inner_cv.query(f'dataset=="{data_name}"').reset_index(drop=True)

selection_cv = LagSelectionFromCV(inner_cv)
selection_data = LagSelectionFromData(df=df, horizon=horizon, frequency=freq_int)

cv_based = selection_cv.predict_lags()
data_based, results_df = selection_data.select_by_uid()
params_based = selection_data.select_from_params()

selected_lags = {**cv_based, **data_based, **params_based}

pprint(selected_lags)
