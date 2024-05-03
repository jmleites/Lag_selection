import copy

import numpy as np
import rpy2.robjects as r_objects
from rpy2.robjects import pandas2ri


def find_frequency(y: np.ndarray):
    pandas2ri.activate()

    y_fit = copy.deepcopy(y)

    y_r = pandas2ri.py2rpy_pandasseries(y_fit)

    r_objects.r('''
               f <- function(y_train) {
                        library(forecast)

                        freq <- findfrequency(y_train)

                        return(freq)
                }
                ''')

    find_frequency_r_ = r_objects.globalenv['f']
    frequency = find_frequency_r_(y_r)[0]
    pandas2ri.deactivate()

    return frequency
