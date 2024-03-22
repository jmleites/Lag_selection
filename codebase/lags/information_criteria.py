import numpy as np


def aic(y_hat, y, k):
    n = len(y)
    residual = y_hat - y
    rss = np.sum(np.power(residual, 2))
    aic_score = n * np.log(rss / n) + 2 * k

    return aic_score


def bic(y_hat, y, k):
    n = len(y)
    residual = y_hat - y
    rss = np.sum(np.power(residual, 2))
    bic_score = n * np.log(rss / n) + k * np.log(n)

    return bic_score
