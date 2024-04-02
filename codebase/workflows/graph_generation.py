import matplotlib.pyplot as plt

def smape_graph(smape_list):
    models = [item[0] for item in smape_list]
    values = [item[1] for item in smape_list]

    plt.figure(figsize=(10, 6))
    plt.bar(models, values, color='skyblue')
    plt.title('SMAPE Values for Forecasting Models')
    plt.xlabel('Model')
    plt.ylabel('SMAPE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def rmse_graph(df, list_models):
    df_sorted = df.sort_values(by='unique_id')

    unique_ids = df_sorted['unique_id']
    model_values = {model: df_sorted[model] for model in list_models}

    plt.figure(figsize=(10, 6))
    for model, values in model_values.items():
        plt.plot(unique_ids, values, label=model)

    plt.xlabel('unique_id')
    plt.ylabel('RMSE')
    plt.title('Trajectory of models values across unique_id')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def best_model_graph(best_model_counts, list_models_num):
    import matplotlib.pyplot as plt

    models = [model for model in list_models_num if model in best_model_counts]
    counts = [best_model_counts[model] for model in models]

    # Plotting the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(models, counts, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Count')
    plt.title('Frequency of Best Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
