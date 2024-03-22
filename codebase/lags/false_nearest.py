import copy
import rpy2.robjects as r_objects
from rpy2.robjects import pandas2ri

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