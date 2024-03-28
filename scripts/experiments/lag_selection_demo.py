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
