import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from utils import expand_grid_from_dict, parse_config

PARAM_GRID = {
    'booster': ['gbtree'],
    'n_estimators': [300],
    'max_depth': [1, 3, 5],
    'learning_rate': [0.1, 0.2],
    'lambda': [0, 1],
    'alpha': [0, 1, 10]
}

# PARAM_GRID = {
#     'booster': ['gblinear'],
#     'n_estimators': [150],
#     'max_depth': [1, 2, 3, 5],
#     'learning_rate': [0.1, 0.2, 0.05],
#     'lambda': [0],
#     'alpha': [0]
# }


class XGBoostRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, early_stopping_rounds: int = 20):

        self.model = XGBRegressor(n_jobs=1, verbosity = 0)
        self.params = self.model.get_params()
        self.early_stopping_rounds = early_stopping_rounds
        self.config = None

    def fit(self, X, y):

        self.optimize_model(X, y)

        self.model.fit(X, y)

    def predict(self, X):
        if self.model.get_params()['booster'] == 'gbtree':
            y_hat = self.model.predict(X)
        else:
            y_hat = self.model.predict(X, ntree_limit=0)

        return y_hat

    def optimize_model(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y,
                             test_size=0.1,
                             shuffle=False)

        param_grid = expand_grid_from_dict(PARAM_GRID)

        errors = []
        for i in range(param_grid.shape[0]):
            config = parse_config(param_grid.iloc[i, :])

            model = XGBRegressor(**config, verbosity = 0)

            model.fit(X_train, y_train,
                      early_stopping_rounds=self.early_stopping_rounds,
                      eval_set=[(X_test, y_test)],
                      verbose=False)

            if config['booster'] == 'gbtree':
                predictions = model.predict(X_test, ntree_limit=model.best_iteration)
            else:
                predictions = model.predict(X_test, ntree_limit=0)

            score = mae(y_test, predictions)
            errors.append(score)

        best_config_id = np.argmin(errors)
        best_config = parse_config(param_grid.iloc[best_config_id, :])

        self.config = best_config
        self.model.set_params(**best_config)
        self.params = self.model.get_params()
