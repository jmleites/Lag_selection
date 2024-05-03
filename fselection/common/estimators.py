from typing import List, Generator

import numpy as np

from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import indexable, _num_samples


class MonteCarloCV(_BaseKFold):
    def __init__(self,
                 n_splits: int,
                 train_size: float,
                 test_size: float,
                 gap: int = 0):
        """
        Monte Carlo Cross-Validation
        Holdout applied in multiple testing periods

        Testing origin (time-step where testing
        begins) is randomly chosen according to a monte carlo simulation

        :param n_splits: (int) Number of monte carlo repetitions in the procedure
        :param train_size: (float) Train size, in terms of ratio of the total length of the series
        :param test_size: (float) Test size, in terms of ratio of the total length of the series
        :param gap: (int) Number of samples to exclude from the end of each train set before the test set.
        """

        self.n_splits = n_splits
        self.n_samples = -1
        self.gap = gap
        self.train_size = train_size
        self.test_size = test_size
        self.train_n_samples = 0
        self.test_n_samples = 0

        self.mc_origins = []

    def split(self, X, y=None, groups=None) -> Generator:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """

        X, y, groups = indexable(X, y, groups)
        self.n_samples = _num_samples(X)

        self.train_n_samples = int(self.n_samples * self.train_size) - 1
        self.test_n_samples = int(self.n_samples * self.test_size) - 1

        # Make sure we have enough samples for the given split parameters
        if self.n_splits > self.n_samples:
            raise ValueError(
                f'Cannot have number of folds={self.n_splits} greater'
                f' than the number of samples={self.n_samples}.'
            )
        if self.train_n_samples - self.gap <= 0:
            raise ValueError(
                f'The gap={self.gap} is too big for number of training samples'
                f'={self.train_n_samples} with testing samples={self.test_n_samples} and gap={self.gap}.'
            )

        indices = np.arange(self.n_samples)

        # print(self.train_n_samples)
        # print(self.n_samples)
        # print(self.test_n_samples)
        selection_range = np.arange(self.train_n_samples, self.n_samples - self.test_n_samples)
        # selection_range = np.arange(67, 96 - 28)

        self.mc_origins = \
            np.random.choice(a=selection_range,
                             size=self.n_splits,
                             replace=True)

        # print(selection_range)
        # print(self.mc_origins)

        for origin in self.mc_origins:
            if self.gap > 0:
                train_end = origin - self.gap + 1
            else:
                train_end = origin - self.gap
            train_start = origin - self.train_n_samples - 1

            test_end = origin + self.test_n_samples

            yield (
                indices[train_start:train_end],
                indices[origin:test_end],
            )

    def get_origins(self) -> List[int]:
        return self.mc_origins


class Holdout:
    """
    Hold-out cross-validator generator. In the hold-out, the
    data is split only once into a train set and a test set.
    Unlike in other cross-validation schemes, the hold-out
    consists of only one iteration.

    Parameters
    ----------
    n : total number of samples
    test_size : 0 < float < 1
        Fraction of samples to use as test set. Must be a
        number between 0 and 1.
    random_state : int
        Seed for the random number generator.
    """

    def __init__(self, n, test_size=0.1):
        self.n = n
        self.test_size = test_size

    def __iter__(self):
        n_test = int(np.ceil(self.test_size * self.n))
        n_train = self.n - n_test
        permutation = np.arange(self.n)
        ind_test = permutation[:n_train]
        ind_train = permutation[n_train:n_train + n_test]
        yield ind_train, ind_test
