import numpy as np
from sklearn.utils import indexable
from sklearn.model_selection import BaseCrossValidator


class TemporalSplit(BaseCrossValidator):
    """Time Series cross-validator

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    Instead of classical TimeSeriesSplit from sklearn, there is overlapping
    between train/test indexes of each fold.
    A gap equal to k means that there are k observations to separate
    consecutive test folds.
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 1.

    test_size : int, default=None
        Size of the test set.
        If None, it's automatically set to 25% of received data.

    max_train_size : int, default=None
        Maximum size for the training set.
        If None, all the past data are used for training.

    gap : int, default=1
        Number of samples to skip between consecutive test set.

    Examples
    --------
    >>> import numpy as np
    >>> from tspiral.model_selection import TemporalSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> tscv = TemporalSplit()
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
    >>> # Add in a 2 period gap
    >>> tscv = TemporalSplit(n_splits=3, test_size=2, gap=2)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 2 3 4 5] TEST: [6 7]
    TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]
    TRAIN: [0 1 2 3 4 5 6 7 8 9] TEST: [10 11]
    """

    def __init__(
            self,
            n_splits=5,
            *,
            test_size=None,
            max_train_size=None,
            gap=1
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.max_train_size = max_train_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
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

        if self.gap < 1:
            raise ValueError(
                "gap must be an integer > 0. Got {}.".format(self.gap)
            )

        if self.n_splits < 2:
            raise ValueError(
                "n_splits must be an integer > 1. Got {}.".format(self.n_splits)
            )

        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)
        test_size = self.test_size if self.test_size is not None \
            else int(n_samples * 0.25)

        test_start = (n_samples - test_size)

        if test_start - (self.n_splits - 1) * self.gap <= 0:
            raise ValueError(
                "Too many splits={} for number of samples={} "
                "with test_size={} and gap={}.".format(
                    self.n_splits, n_samples, test_size, self.gap
                )
            )

        indices = np.arange(n_samples)
        test_starts = test_start - np.arange((self.n_splits - 1), -1, -1) * self.gap

        for test_start in test_starts:
            if self.max_train_size is not None and self.max_train_size < test_start:
                yield (
                    indices[test_start - self.max_train_size: test_start],
                    indices[test_start: test_start + test_size],
                )
            else:
                yield (
                    indices[:test_start],
                    indices[test_start: test_start + test_size],
                )

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