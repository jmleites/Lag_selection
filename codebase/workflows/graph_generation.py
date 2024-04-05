import os
import matplotlib.pyplot as plt
def smape_graph(smape_list, data_name, group, results_name: str = 'all'):
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logs'))
    cached_logs = os.path.join(parent_directory, f'{data_name}_{group}_bars_{results_name}.pdf')

    models = [item[0] for item in smape_list]
    values = [item[1] for item in smape_list]

    plt.figure(figsize=(10, 6))
    plt.bar(models, values, color='skyblue')
    plt.title('SMAPE Values for Forecasting Models')
    plt.xlabel('Model')
    plt.ylabel('SMAPE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if not os.path.exists(cached_logs):
        plt.savefig(cached_logs)
    plt.show()

def smape_boxplot(df, list_models, data_name, group, results_name: str = 'all'):
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logs'))
    cached_logs = os.path.join(parent_directory, f'{data_name}_{group}_boxplot_{results_name}.pdf')

    df_sorted = df.sort_values(by='unique_id')
    model_values = {model: df_sorted[model] for model in list_models}

    plt.figure(figsize=(10, 12))
    plt.boxplot(model_values.values(), labels=model_values.keys())
    plt.xlabel('Model')
    plt.ylabel('SMAPE')
    plt.title('Boxplot of SMAPE values for each model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if not os.path.exists(cached_logs):
        plt.savefig(cached_logs)
    plt.show()
