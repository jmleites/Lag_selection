from sklearn.model_selection import train_test_split

from methods.tde import time_delay_embedding


def holdout_estimation(series, k, model, train_size: float = 0.8):
    X, y = time_delay_embedding(series, n_lags=k, return_X_y=True)

    X_tr, X_ts, y_tr, y_ts = \
        train_test_split(X, y, train_size=train_size)

    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_ts)

    return y_hat, y_ts
