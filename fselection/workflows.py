import time
import copy

import pandas as pd
from sklearn.metrics import mean_absolute_error

from methods.selector import LagSelector
from methods.tde import time_delay_embedding


def wf_lag_selection(y_train, y_test, algorithm, frequency):
    print('Init class')
    selector = LagSelector(frequency=frequency)

    print('Estimating Lags')
    start = time.time()
    k_pacf1, predictors_pacf1 = selector.pacf(y_train, tol=0.01)
    pacf1_time = time.time() - start
    start = time.time()
    k_pacf2, predictors_pacf2 = selector.pacf(y_train, tol=0.005)
    pacf2_time = time.time() - start
    start = time.time()
    k_pacf3, predictors_pacf3 = selector.pacf(y_train, tol=0.001)
    pacf3_time = time.time() - start
    start = time.time()
    k_pca = selector.pca(y_train)
    pca_time = time.time() - start
    start = time.time()
    k_fnn0, predictors_fnn0 = selector.fnn_rpy(y_train, tol=0)
    fnn0_time = time.time() - start
    start = time.time()
    k_fnn1, predictors_fnn1 = selector.fnn_rpy(y_train, tol=0.1)
    fnn1_time = time.time() - start
    start = time.time()
    k_fnn2, predictors_fnn2 = selector.fnn_rpy(y_train, tol=0.01)
    fnn2_time = time.time() - start
    start = time.time()
    k_gs, predictors_gs = selector.grid_search(y_train, copy.deepcopy(algorithm)())
    gs_time = time.time() - start
    start = time.time()
    k_aic, predictors_aic = selector.aic_selection(y_train, copy.deepcopy(algorithm)())
    aic_time = time.time() - start
    start = time.time()
    k_bic, predictors_bic = selector.bic_selection(y_train, copy.deepcopy(algorithm)())
    bic_time = time.time() - start
    start = time.time()
    k_freq, predictors_freq = selector.frequency_selection(y_train)
    freq_time = time.time() - start
    start = time.time()
    k_perm, predictors_perm = \
        selector.ts_feature_selection(y_train,
                                      fs_method='permutation',
                                      permutation_model=copy.deepcopy(algorithm)(),
                                      predictive_model=copy.deepcopy(algorithm)())
    perm_time = time.time() - start
    start = time.time()
    k_f, predictors_f = \
        selector.ts_feature_selection(y_train,
                                      fs_method='f',
                                      permutation_model=copy.deepcopy(algorithm)(),
                                      predictive_model=copy.deepcopy(algorithm)())
    f_time = time.time() - start
    start = time.time()
    k_mi, predictors_mi = \
        selector.ts_feature_selection(y_train,
                                      fs_method='mi',
                                      permutation_model=copy.deepcopy(algorithm)(),
                                      predictive_model=copy.deepcopy(algorithm)())
    mi_time = time.time() - start
    # start = time.time()
    # k_rrelieff, predictors_rr = \
    #     selector.ts_feature_selection(y_train,
    #                                   fs_method='rrelieff',
    #                                   permutation_model=algorithm(),
    #                                   predictive_model=algorithm())
    # rrelieff_time = time.time() - start

    print('Saving values')
    k_time = \
        dict(pacf1=pacf1_time,
             pacf2=pacf2_time,
             pacf3=pacf3_time,
             pca=pca_time,
             fnn0=fnn0_time,
             fnn1=fnn1_time,
             fnn2=fnn2_time,
             gs=gs_time,
             aic=aic_time,
             bic=bic_time,
             freq=freq_time,
             permutation=perm_time,
             f_regress=f_time,
             mi=mi_time)

    k_hat = dict(bl_10=10,
                 bl_2=2,
                 bl_50=50,
                 pacf1=k_pacf1,
                 pacf2=k_pacf2,
                 pacf3=k_pacf3,
                 pca=k_pca,
                 fnn0=k_fnn0,
                 fnn1=k_fnn1,
                 fnn2=k_fnn2,
                 gs=k_gs,
                 aic=k_aic,
                 bic=k_bic,
                 freq=k_freq,
                 f=k_f,
                 mi=k_mi,
                 # rrelieff=k_rrelieff,
                 perm=k_perm)

    predictor_names = dict(f=predictors_f,
                           mi=predictors_mi,
                           # rrelieff=predictors_rr,
                           permutation=predictors_perm)

    results_by_k = dict()
    results_by_method = dict()
    print('Training')
    for method in k_hat:
        print(method)
        k = k_hat[method]
        print(k_hat[method])

        if k < 2:
            k = 2

        if method in ['pca', 'f', 'mi', 'rrelieff', 'permutation']:
            k = selector.MAX_K

        if k in results_by_k and method not in ['pca', 'f', 'mi', 'rrelieff', 'permutation']:
            results_by_method[method] = results_by_k[k]
            continue

        X_tr, y_tr = time_delay_embedding(y_train, n_lags=k, return_X_y=True)
        X_ts, y_ts = time_delay_embedding(y_test, n_lags=k, return_X_y=True)

        if method == 'pca':
            X_tr = selector.model.transform(X_tr)
            X_ts = selector.model.transform(X_ts)

        if method in ['f', 'mi', 'rrelieff', 'permutation']:
            X_tr = X_tr[predictor_names[method]]
            X_ts = X_ts[predictor_names[method]]

        if isinstance(X_tr, pd.DataFrame):
            X_tr = X_tr.values

        if isinstance(X_ts, pd.DataFrame):
            X_ts = X_ts.values

        model = copy.deepcopy(algorithm)()
        model.fit(X_tr, y_tr)

        y_hat = model.predict(X_ts)

        err = mean_absolute_error(y_ts, y_hat)

        if method not in ['pca', 'f', 'mi', 'rrelieff', 'permutation']:
            results_by_k[k] = err

        results_by_method[method] = err

    return k_hat, results_by_method, k_time
