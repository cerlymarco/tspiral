import numbers
import numpy as np
import pandas as pd

from sklearn.utils import indexable
from sklearn.model_selection import BaseCrossValidator


class TemporalSplit(BaseCrossValidator):
    """Time Series cross-validator

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    Instead of classical TimeSeriesSplit from sklearn, there is the possibility
    to create overlapped train/test indexes between consecutive folds.
    A gap equal to k means that there are k observations to separate
    consecutive test folds.
    It's possible to operate grouped train/test split by specifying the groups
    parameter passing an iterable when calling the split method.
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 1.

    test_size : int, default=None
        Size of the test set.
        If None, it's automatically set to 25% of received data.
        When using groups, a test size equal to test_size is produced
        for each group.

    max_train_size : int, default=None
        Maximum size for the training set.
        If None, all the past data are used for training.
        When using groups, a train size equal to max_train_size is produced
        for each group.

    equal_test_size : bool, default=True
        Effective when using groups.
        When set to True, it produces folds of equal size discharging
        data from groups with not enough train/test samples.
        When set to False, it produces folds of not consistent size creating
        train/test partitions with all the data at disposal of each group.

    gap : int, default=1
        Number of samples to skip between consecutive test set.

    Examples
    --------
    >>> import numpy as np
    >>> from tspiral.model_selection import TemporalSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> tscv = TemporalSplit(test_size=1)
    >>> for train_index, test_index in tscv.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]
    TRAIN: [0 1 2 3] TEST: [4]
    TRAIN: [0 1 2 3 4] TEST: [5]
    >>> # Fix test_size to 2 with 12 samples
    >>> X = np.random.randn(12, 2)
    >>> y = np.random.randint(0, 2, 12)
    >>> tscv = TemporalSplit(n_splits=3, test_size=2)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]
    TRAIN: [0 1 2 3 4 5 6 7 8] TEST: [ 9 10]
    TRAIN: [0 1 2 3 4 5 6 7 8 9] TEST: [10 11]
    >>> # Add a 2 period gap
    >>> tscv = TemporalSplit(n_splits=3, test_size=2, gap=2)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 2 3 4 5] TEST: [6 7]
    TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]
    TRAIN: [0 1 2 3 4 5 6 7 8 9] TEST: [10 11]
    >>> # Add group partition
    >>> groups = np.array([0]*6+[1]*6)
    >>> tscv = TemporalSplit(n_splits=3, test_size=2)
    >>> for train_index, test_index in tscv.split(X, None, groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 6 7] TEST: [2 3 8 9]
    TRAIN: [0 1 2 6 7 8] TEST: [ 3  4  9 10]
    TRAIN: [0 1 2 3 6 7 8 9] TEST: [ 4  5 10 11]
    """

    def __init__(
            self,
            n_splits=5,
            *,
            test_size,
            max_train_size=None,
            equal_test_size=True,
            gap=1
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.max_train_size = max_train_size
        self.equal_test_size = equal_test_size
        self.gap = gap

    def _split(self, idx, equal_test_size):
        """Private method to generate split indexes."""

        n_samples = idx.shape[0]
        test_starts = n_samples - self.test_size
        test_starts = test_starts - np.arange((self.n_splits - 1), -1, -1) * self.gap

        if test_starts[0] <= 0 and equal_test_size:
            yield [], []

        else:
            if not equal_test_size: test_starts = test_starts[test_starts > 0]
            for test_start in test_starts:
                if self.max_train_size is not None \
                        and self.max_train_size < test_start:
                    yield (
                        idx[test_start - self.max_train_size: test_start].tolist(),
                        idx[test_start: test_start + self.test_size].tolist(),
                    )
                else:
                    yield (
                        idx[:test_start].tolist(),
                        idx[test_start: test_start + self.test_size].tolist(),
                    )

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data in ascending temporal order, where `n_samples` is
            the number of samples and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,) or
            (n_samples, n_groups), default=None
            Group labels for the samples used while splitting the dataset
            into train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """

        if not isinstance(self.n_splits, numbers.Integral) or \
                self.n_splits < 1:
            raise ValueError(
                "n_splits must be an integer > 0. Got {} ({})." \
                    .format(self.n_splits, type(self.n_splits))
            )

        if not isinstance(self.test_size, numbers.Integral) or \
                self.test_size < 1:
            raise ValueError(
                "test_size must be an integer > 0. Got {} ({})." \
                    .format(self.test_size, type(self.test_size))
            )

        if not isinstance(self.gap, numbers.Integral) or \
                self.gap < 1:
            raise ValueError(
                "gap must be an integer > 0. Got {} ({})." \
                    .format(self.gap, type(self.gap))
            )

        X, y, groups = indexable(X, y, groups)
        folds_id_train = {f: [] for f in range(self.n_splits)}
        folds_id_test = {f: [] for f in range(self.n_splits)}

        if groups is not None:
            groups = np.asarray(groups)
            grouper = pd.DataFrame(groups).groupby(
                list(range(groups.shape[1] if len(groups.shape) > 1 else 1))
            ).indices

            for g_indexes in grouper.values():
                splitter = self._split(g_indexes, self.equal_test_size)
                for i, (id_train, id_test) in enumerate(splitter):
                    folds_id_train[i].extend(id_train)
                    folds_id_test[i].extend(id_test)

        else:
            indexes = np.arange(len(X))
            splitter = self._split(indexes, equal_test_size=True)
            for i, (id_train, id_test) in enumerate(splitter):
                folds_id_train[i].extend(id_train)
                folds_id_test[i].extend(id_test)

        if min(map(len, folds_id_train.values())) < 1:
            raise ValueError(
                "No train samples available with test_size={}. "
                "Try reducing test_size, or providing more samples.".format(
                    self.test_size
                )
            )

        for id_train, id_test in zip(
                folds_id_train.values(), folds_id_test.values()
        ):
            id_train = np.asarray(id_train)
            id_test = np.asarray(id_test)
            yield id_train, id_test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits