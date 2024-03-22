from neuralforecast.models import NHITS
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import SMAPE
from neuralforecast.losses.numpy import smape
from lightning.pytorch.loggers import CSVLogger
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse
import pandas as pd
import os
from statsforecast.models import SeasonalNaive, WindowAverage
from statsforecast import StatsForecast
from codebase.load_data.config import DATASETS

def initialize_models():
    stats_models = [
        SeasonalNaive(season_length=freq_int),
        WindowAverage(window_size=freq_int),
    ]

    if data_name == 'M3':
        sf = StatsForecast(
            models=stats_models,
            freq=freq_str,
            n_jobs=1,
        )
    else:
        sf = StatsForecast(
            models=stats_models,
            freq=1,
            n_jobs=1,
        )

    nhits_config = {
        'max_steps': 1000,
        'val_check_steps': 30,
        'enable_checkpointing': True,
        'early_stop_patience_steps': 25,
        'start_padding_enabled': True,
        'valid_loss': SMAPE(),
        'accelerator': 'cpu'
    }

    for i in input_list:
        models.append(NHITS(h=horizon,
                            input_size=i,
                            **nhits_config))

    nf = NeuralForecast(models=models, freq=freq_str)

    return df, sf, nf

def cross_validation():
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'results'))
    cached_cv_results = os.path.join(parent_directory, f'{data_name}_{group}_results.csv')
    if os.path.exists(cached_cv_results):
        cv_df = pd.read_csv(cached_cv_results)
    else:
        cv_sf = sf.cross_validation(df=df, h=horizon, test_size=horizon * test_size, n_windows=None)
        cv_nf = nf.cross_validation(df=df, val_size=horizon * val_size, test_size=horizon * test_size, n_windows=None, use_init_models=True)
        cv_df = cv_nf.merge(cv_sf.drop(columns=['y']), how='left', on=['unique_id', 'ds', 'cutoff'])
        cv_df['dataset'] = data_name
        cv_df.to_csv(cached_cv_results, index=False)
    return cv_df

def calculate_smape(cv_df, model):
    smape_model = smape(y=cv_df['y'], y_hat=cv_df[model])
    return smape_model

def evaluation():
    done = False
    best_input = []
    for model in range(len(evaluation_df)):
        done = False
        for i in range(len(evaluation_df.loc[model, 'best_model'])):
            letter = evaluation_df.loc[model, 'best_model'][i]
            if letter.isnumeric():
                if i < len(evaluation_df.loc[model, 'best_model']) - 1:
                    next_letter = evaluation_df.loc[model, 'best_model'][i + 1]
                    if next_letter.isnumeric():
                        numeric_str = letter + next_letter
                        best_input.append(int(numeric_str))
                        done = True
                        break
                    else:
                        best_input.append(int(letter))
                        done = True
                        break
                else:
                    best_input.append(int(letter))
                    done = True
        if not done:
            best_input.append(0)

    corresponding_values = []
    for i in best_input:
        corresponding_values.append(input_list[i])

    evaluation_df.insert(evaluation_df.columns.get_loc('best_model') + 1, 'best_input', corresponding_values)
    return evaluation_df

input_list = [1,2,3, 6, 12, 18, 24, 36, 48, 60, 90, 120]

while True:
    dataset_choice = input("Choose dataset (M3 or M4) or 'exit' to quit: ").strip().lower()
    if dataset_choice == 'exit':
        break
    if dataset_choice not in ['m3', 'm4']:
        print("Invalid choice. Please choose 'M3' or 'M4'.")
        continue

    models = []
    data_name = dataset_choice.upper()
    group = 'Monthly'
    print("For dataset:", data_name)
    data_loader = DATASETS[data_name]
    df = data_loader.load_data(group)
    horizon = data_loader.horizons_map.get(group)
    n_lags = data_loader.context_length.get(group)
    freq_str = data_loader.frequency_pd.get(group)
    freq_int = data_loader.frequency_map.get(group)
    test_size = 2
    val_size = 1
    df, sf, nf = initialize_models()
    cv_df = cross_validation()
    list_models = []
    for model in sf.models:
        list_models.append(model.__class__.__name__)
    for model in nf.models:
        list_models.append(model.__class__.__name__)
    for i, model in enumerate(list_models):
        if i == 0 or i == 1:
            smape_value = calculate_smape(cv_df, model)
            print("SMAPE for", model, ":", smape_value)
        else:
            if i == 2:
                smape_value = calculate_smape(cv_df, model)
                print("SMAPE for NHITS with input size", input_list[i-2], ":", smape_value)
            else :
                smape_value = calculate_smape(cv_df, model + str(i-2))
                print("SMAPE for NHITS with input size", input_list[i-2], ":", smape_value)

    evaluation_df = evaluate(cv_df.drop(columns=['cutoff', 'ds', 'dataset']), metrics=[rmse])
    evaluation_df['best_model'] = evaluation_df.drop(columns=['metric', 'unique_id']).idxmin(axis=1)
    evaluation_df = evaluation()
    print(evaluation_df)
