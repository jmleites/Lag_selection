import pandas as pd
from sklearn_relief import RReliefF


def rrelieff_wrapper(X, y):
    feature_importance_model = RReliefF()
    feature_importance_model.n_jobs = 1

    if isinstance(X, pd.DataFrame):
        X = X.values

    feature_importance_model.fit(X, y)
    importance_rrelieff = -feature_importance_model.w_

    return importance_rrelieff
