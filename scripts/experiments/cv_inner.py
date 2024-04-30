from codebase.load_data.config import DATASETS
from codebase.load_data.base import LoadDataset
from codebase.workflows.cross_validation import cross_validation
from codebase.workflows.models import initialize_models
from codebase.workflows.config import INPUT_RANGE

EXPERIMENT = 'inner'

while True:
    dataset_choice = input("Choose dataset (M3/M4/Tourism) or 'exit' to quit: ").strip().lower()
    if dataset_choice == 'exit':
        break
    if dataset_choice not in ['m3', 'm4', 'tourism', 'gluonts']:
        print("Invalid choice. Please choose a valid dataset.")
        continue

    data_name = dataset_choice.capitalize()

    group = 'Monthly' if dataset_choice != 'gluonts' else 'm1_monthly'

    print("For dataset:", data_name)
    data_loader = DATASETS[data_name]
    df = data_loader.load_data(group)
    horizon = data_loader.horizons_map.get(group)
    n_lags = data_loader.context_length.get(group)
    freq_str = data_loader.frequency_pd.get(group)
    freq_int = data_loader.frequency_map.get(group)

    df, _ = LoadDataset.train_test_split(df, horizon)

    df, sf, nf = initialize_models(data_name, freq_int, freq_str, INPUT_RANGE, horizon, df)
    cv_df = cross_validation(df, horizon, sf, nf, data_name, results_name=EXPERIMENT)
