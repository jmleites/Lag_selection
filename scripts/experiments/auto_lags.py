import pandas as pd

from codebase.load_data.config import DATASETS
from codebase.workflows.cross_validation import cross_validation
from codebase.workflows.models import initialize_models
from codebase.workflows.config import (GROUP,
                                       TEST_SIZE,
                                       VALIDATION_SIZE,
                                       LAGS)

EXPERIMENT = 'lags'

while True:
    dataset_choice = input("Choose dataset (M3/M4/Tourism) or 'exit' to quit: ").strip().lower()
    if dataset_choice == 'exit':
        break
    if dataset_choice not in ['m3', 'm4', 'tourism']:
        print("Invalid choice. Please choose a valid dataset.")
        continue

    models = []
    data_name = dataset_choice.capitalize()

    selected_lags = LAGS[data_name]

    lag_values = pd.Series(selected_lags).astype(int).sort_values().unique()

    print("For dataset:", data_name)
    data_loader = DATASETS[data_name]
    df = data_loader.load_data(GROUP)
    horizon = data_loader.horizons_map.get(GROUP)
    n_lags = data_loader.context_length.get(GROUP)
    freq_str = data_loader.frequency_pd.get(GROUP)
    freq_int = data_loader.frequency_map.get(GROUP)

    df, sf, nf = initialize_models(data_name, freq_int, freq_str, lag_values, models, horizon, df)
    cv_df = cross_validation(df, horizon, TEST_SIZE, VALIDATION_SIZE, sf, nf, data_name, GROUP, results_name=EXPERIMENT)
