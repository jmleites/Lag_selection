import os
import pandas as pd


def cross_validation(df, horizon, sf, nf, data_name, results_name: str = 'all'):
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'results'))
    cached_cv_results = os.path.join(parent_directory, f'{data_name}_{results_name}.csv')
    if os.path.exists(cached_cv_results):
        cv_df = pd.read_csv(cached_cv_results)
    else:
        cv_sf = sf.cross_validation(df=df, h=horizon, test_size=horizon, n_windows=None)
        cv_nf = nf.cross_validation(df=df, val_size=horizon, test_size=horizon, n_windows=None,
                                    use_init_models=True)
        cv_df = cv_nf.merge(cv_sf.drop(columns=['y']), how='left', on=['unique_id', 'ds', 'cutoff'])
        cv_df['dataset'] = data_name
        cv_df.to_csv(cached_cv_results, index=False)

    return cv_df
