import pandas as pd
from sklearn.inspection import permutation_importance


class PermutationImportance:

    def __init__(self,
                 model,
                 cv,
                 permutation_repeats: int,
                 return_dict: bool = False):

        self.model = model
        self.cv = cv
        self.permutation_repeats = permutation_repeats
        self.return_dict = return_dict

    def permutation(self,
                    X: pd.DataFrame,
                    y: pd.Series):
        # assert isinstance(y, pd.Series)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        scores = []
        for tr_ind, ts_ind in self.cv.split(X):
            X_tr, X_ts = X.values[tr_ind], X.values[ts_ind]
            y_tr, y_ts = y.values[tr_ind], y.values[ts_ind]

            self.model.fit(X_tr, y_tr)

            perm_importance = \
                permutation_importance(estimator=self.model,
                                       X=X_ts,
                                       y=y_ts,
                                       scoring='neg_mean_absolute_error',
                                       n_repeats=self.permutation_repeats)

            iter_score = perm_importance['importances_mean']
            scores.append(iter_score)

        scores = pd.DataFrame(scores)

        importance_scores = scores.mean().values

        if self.return_dict:
            importance_scores = dict(zip(X.columns, importance_scores))

        return importance_scores

