from models import initialize_models
from cross_validation import cross_validation
from evaluation import calculate_smape, evaluation
from lightning.pytorch.loggers import CSVLogger
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse
from codebase.load_data.config import DATASETS

input_list = [1,2,3, 6, 12, 18, 24, 36, 48, 60, 90, 120]
test_size = 2
val_size = 1

while True:
    dataset_choice = input("Choose dataset (M3/M4/Tourism) or 'exit' to quit: ").strip().lower()
    if dataset_choice == 'exit':
        break
    if dataset_choice not in ['m3', 'm4', 'tourism']:
        print("Invalid choice. Please choose a valid dataset.")
        continue

    models = []
    data_name = dataset_choice.capitalize()
    group = 'Monthly'
    
    print("For dataset:", data_name)
    data_loader = DATASETS[data_name]
    
    df = data_loader.load_data(group)
    horizon = data_loader.horizons_map.get(group)
    n_lags = data_loader.context_length.get(group)
    freq_str = data_loader.frequency_pd.get(group)
    freq_int = data_loader.frequency_map.get(group)
    
    df, sf, nf = initialize_models(data_name, freq_int, freq_str, input_list, models, horizon, df)
    cv_df = cross_validation(df, horizon, test_size, val_size, sf, nf, data_name, group)
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
    evaluation_df = evaluation(evaluation_df, input_list)
    print(evaluation_df)
