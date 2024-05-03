from typing import Dict
import itertools
import pickle

import numpy as np
import pandas as pd


def sort_dict(x, decreasing: bool = False):
    """ Sort a dictionary by value

    :param x: A dictionary in which x.values() are numeric need to be sorted
    :param decreasing: Boolean; whether the sorting should be decreasing
    :return: A sorted dictionary
    todo doesnt work if theres nan
    """
    x_sorted = {k: v
                for k, v in sorted(x.items(),
                                   reverse=decreasing,
                                   key=lambda item: item[1])}
    return x_sorted


def na_by_row(x: pd.DataFrame):
    """ Any NA in pd.DF row

    :param x: A pd.DataFrame
    :return:
    """

    x = x.copy()
    x = pd.DataFrame(x)

    x = x.isna()
    x = x.any(axis=1)

    return x.values


def dict_argmin(x):
    x_values = list(x.values())
    x_best_key = list(x.keys())[int(np.argmin(x_values))]

    return x_best_key


def expand_grid(*iters):
    product = list(itertools.product(*iters))
    return {i: [x[i] for x in product]
            for i in range(len(iters))}


def expand_grid_from_dict(x: Dict) -> pd.DataFrame:
    param_grid = expand_grid(*x.values())
    param_grid = pd.DataFrame(param_grid)
    param_grid.columns = x.keys()

    return param_grid


def parse_config(x: pd.Series) -> Dict:
    """

    :param x:
    :return:
    """
    config = dict(x)

    for key in config:
        try:
            if np.isnan(config[key]):
                config[key] = None
        except TypeError:
            continue

    return config


def load_data(filepath):
    with open(filepath, 'rb') as fp:
        data = pickle.load(fp)

    return data


def save_data(data, filepath):
    with open(filepath, 'wb') as fp:
        pickle.dump(data, fp)
