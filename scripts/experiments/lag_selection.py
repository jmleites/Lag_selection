from codebase.load_data.config import DATASETS
from codebase.lags.false_nearest import false_nearest_neighbors
from codebase.lags.heuristics import bandara_heuristic
from codebase.lags.pacf import pacf_estimation
from codebase.load_data.base import LoadDataset

group = 'Monthly'
data_name = 'M3'
data_loader = DATASETS[data_name]

df = data_loader.load_data(group)
horizon = data_loader.horizons_map.get(group)
freq_str = data_loader.frequency_pd.get(group)
freq_int = data_loader.frequency_map.get(group)

# series_example = df.query('unique_id=="M1"')['y']


train_df, _ = LoadDataset.train_test_split(df, horizon)

train_by_uid = train_df.groupby('unique_id')

results_by_uid = {}
for uid, df in train_by_uid:
    print(uid)
    s = df['y'].reset_index(drop=True)

    bandara = bandara_heuristic(horizon, freq_int)

    fnn0 = false_nearest_neighbors(s, tol=0)
    fnn1 = false_nearest_neighbors(s, tol=0.01)

    pacf0 = pacf_estimation(s, tol=0)
    pacf1 = pacf_estimation(s, tol=0.01)

    results_by_uid[uid] = {
        'fnn0': fnn0,
        'fnn1': fnn1,
        'pacf0': pacf0,
        'pacf1': pacf1,
        'bandara': bandara,
        'horizon_1x': horizon,
        'horizon_2x': horizon*2,
        'freq_1x': freq_int,
        'freq_2x': freq_int*2,
        'freq_half': freq_int/2,
        'bl_1': 1,
    }
