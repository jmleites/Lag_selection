from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse

from codebase.load_data.config import DATASETS
from codebase.workflows.cross_validation import cross_validation
from codebase.workflows.models import initialize_models
from codebase.workflows.evaluation import calculate_smape, evaluation
from codebase.workflows.config import (GROUP,
                                       TEST_SIZE,
                                       VALIDATION_SIZE,
                                       INPUT_RANGE)

EXPERIMENT = 'main'

while True:
    dataset_choice = input("Choose dataset (M3/M4/Tourism) or 'exit' to quit: ").strip().lower()
    if dataset_choice == 'exit':
        break
    if dataset_choice not in ['m3', 'm4', 'tourism']:
        print("Invalid choice. Please choose a valid dataset.")
        continue

    models = []
    data_name = dataset_choice.capitalize()

    print("For dataset:", data_name)
    data_loader = DATASETS[data_name]
    df = data_loader.load_data(GROUP)
    horizon = data_loader.horizons_map.get(GROUP)
    n_lags = data_loader.context_length.get(GROUP)
    freq_str = data_loader.frequency_pd.get(GROUP)
    freq_int = data_loader.frequency_map.get(GROUP)

    df, sf, nf = initialize_models(data_name, freq_int, freq_str, INPUT_RANGE, models, horizon, df)
    cv_df = cross_validation(df, horizon, TEST_SIZE, VALIDATION_SIZE, sf, nf, data_name, GROUP, results_name=EXPERIMENT)

    list_models = []
    for model in sf.models:
        list_models.append(model.__class__.__name__)
    for model in nf.models:
        list_models.append(repr(model))

    model_counts = {}
    list_models_num = list_models
    for i, model in enumerate(list_models):
        if model in model_counts:
            model_counts[model] += 1
            list_models_num[i] = f"{model}{model_counts[model]}"
        else:
            model_counts[model] = 0
          
    for i, model in enumerate(list_models):
        smape_value = calculate_smape(cv_df, model)
        print("SMAPE for", model, ":", smape_value)

    smape_graph(smape_list, data_name, GROUP, results_name=EXPERIMENT)

    evaluation_df = evaluate(cv_df.drop(columns=['cutoff', 'ds', 'dataset']), metrics=[smape])
    evaluation_df['best_model'] = evaluation_df.drop(columns=['metric', 'unique_id']).idxmin(axis=1)
    evaluation_df = evaluation(evaluation_df, INPUT_RANGE)
    print(evaluation_df)

    smape_boxplot(evaluation_df, list_models_num, data_name, GROUP, results_name=EXPERIMENT)
