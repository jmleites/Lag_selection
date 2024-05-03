from typing import Optional

import numpy as np
import pandas as pd
from methods.tde import time_delay_embedding
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa import stattools
from pmdarima.arima import ndiffs

from common.frequency import find_frequency
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection \
    import (SelectKBest,
            f_regression,
            mutual_info_regression)
from utils import sort_dict
import copy
import rpy2.robjects as r_objects
from rpy2.robjects import pandas2ri

from feature_selection.relieff import rrelieff_wrapper
from feature_selection.information_criteria import aic, bic
from feature_selection.permutation import PermutationImportance
from common.workflows import holdout_estimation
from common.estimators import Holdout, MonteCarloCV


class LagSelector:
    MAX_K = 50
    PCA_MIN_EXPLAINED_VAR = 0.95
    LAG_NAMES = ['t'] + ['t-' + str(i) for i in range(1, MAX_K)]

    def __init__(self, frequency: Optional[int]):

        if frequency is None:
            self.frequency = 1
        else:
            self.frequency = frequency

        self.model = None

    def pca(self, series):

        X, _ = time_delay_embedding(series, n_lags=self.MAX_K, return_X_y=True)

        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA()
        pca.fit(X)

        explained_variance = pca.explained_variance_ratio_
        cs_exp_variance = np.cumsum(explained_variance)

        n_components = \
            np.where(cs_exp_variance >= self.PCA_MIN_EXPLAINED_VAR)[0][0] + 1

        if n_components < 2:
            n_components = 2

        pca = PCA(n_components=n_components)
        pca.fit(X)

        self.model = pca

        return n_components

    def fnn_rpy(self, series, tol):
        pandas2ri.activate()

        series_fit = copy.deepcopy(series)

        data_set = pandas2ri.py2rpy_pandasseries(series_fit)

        r_objects.r('''
                        estimate_k <-
                            function(x, max_k=20,tol=.15) {
                                require(tseriesChaos)

                                fn.out <- false.nearest(x, max_k, d=1, t=1)
                                fn.out <- round(fn.out,4)
                                fn.out[is.na(fn.out)] <- 0

                                fnp.tol <- fn.out["fraction",] > tol
                                fnp.tol.sum <- sum(fnp.tol)

                                m <- ifelse(fnp.tol.sum < max_k,fnp.tol.sum + 1, max_k)

                                return(m)
                            }
                        ''')

        estimate_k_fnn = r_objects.globalenv['estimate_k']
        k_hat = int(estimate_k_fnn(data_set, 20, tol)[0])
        pandas2ri.deactivate()

        predictor_names = self.LAG_NAMES[:k_hat]

        return k_hat, predictor_names

    def pacf(self, series: pd.Series, tol=0.01):

        if isinstance(series, np.ndarray):
            series = pd.Series(series)

        d_required = ndiffs(series)

        if d_required > 0:
            series = series.diff(periods=d_required)[1:]

        pacf_score = stattools.pacf(series, nlags=self.MAX_K)

        pacf_abs_score = np.abs(pacf_score)

        try:
            k_hat = np.where(pacf_abs_score <= tol)[0][0]
        except IndexError:
            k_hat = self.MAX_K

        predictor_names = self.LAG_NAMES[:k_hat]

        return k_hat, predictor_names

    def frequency_selection(self, series):

        freq = find_frequency(series)

        if freq == 1:
            k_hat = 7
        else:
            k_hat = freq

        predictor_names = self.LAG_NAMES[:k_hat]

        return k_hat, predictor_names

    def grid_search(self, series, model):
        iter_values = list(range(2, self.MAX_K))

        results_by_k = []
        for k in iter_values:
            y_hat_iter, y_iter = \
                holdout_estimation(series=series,
                                   k=k,
                                   model=model)

            score = mse(y_iter, y_hat_iter)

            results_by_k.append(score)

        k_hat = iter_values[np.argmin(results_by_k)]

        predictor_names = self.LAG_NAMES[:k_hat]

        return k_hat, predictor_names

    def bic_selection(self, series, model):
        iter_values = list(range(2, self.MAX_K))

        results_by_k = []
        for k in iter_values:
            y_hat_iter, y_iter = \
                holdout_estimation(series=series,
                                   k=k,
                                   model=model)

            bic_score = bic(y_hat_iter, y_iter, k)

            results_by_k.append(bic_score)

        k_hat = iter_values[np.argmin(results_by_k)]

        predictor_names = self.LAG_NAMES[:k_hat]

        return k_hat, predictor_names

    def aic_selection(self, series, model):
        iter_values = list(range(2, self.MAX_K))

        results_by_k = []
        for k in iter_values:
            y_hat_iter, y_iter = \
                holdout_estimation(series=series,
                                   k=k,
                                   model=model)

            aic_score = aic(y_hat_iter, y_iter, k)

            results_by_k.append(aic_score)

        k_hat = iter_values[np.argmin(results_by_k)]

        predictor_names = self.LAG_NAMES[:k_hat]

        return k_hat, predictor_names

    def ts_feature_selection(self, series, fs_method, permutation_model, predictive_model):

        assert fs_method in ['f', 'rrelieff', 'mi', 'permutation']

        # tde = TimeDelayEmbedding(embedding_dimension=self.MAX_K, horizon=1)
        X, y = time_delay_embedding(series, n_lags=self.MAX_K, return_X_y=True)

        if fs_method == 'f':
            select_fun = f_regression
        elif fs_method == 'mi':
            select_fun = mutual_info_regression
        elif fs_method == 'permutation':
            cv = MonteCarloCV(n_splits=1, train_size=0.7, test_size=0.2)
            perm = PermutationImportance(model=permutation_model,
                                         cv=cv,
                                         permutation_repeats=10,
                                         return_dict=False)
            select_fun = perm.permutation
        else:
            select_fun = rrelieff_wrapper

        pipeline = Pipeline([
            ('select', SelectKBest(score_func=select_fun)),
            ('model', predictive_model)])

        top_k = np.linspace(start=2, stop=self.MAX_K, num=int(self.MAX_K / 2))
        top_k = top_k.astype(int)

        param_grid = {'select__k': top_k}

        holdout = Holdout(test_size=0.2, n=X.shape[0])

        search = GridSearchCV(pipeline,
                              param_grid,
                              scoring='neg_mean_absolute_error',
                              verbose=2,
                              cv=holdout)

        search.fit(X.values, y)

        importance_scores = select_fun(X.values, y)
        if isinstance(importance_scores, tuple):
            importance_scores = importance_scores[0]

        importance_scores = dict(zip(X.columns, importance_scores))

        top_k_value = search.best_params_['select__k']

        importance_scores = sort_dict(importance_scores, decreasing=True)
        top_features = list(importance_scores.keys())[:top_k_value]

        return top_k_value, top_features
