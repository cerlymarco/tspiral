import numbers
import warnings
import numpy as np
import pandas as pd
from inspect import signature
from scipy.optimize import fmin_slsqp

from sklearn.base import clone
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import \
    _check_sample_weight, has_fit_parameter, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from joblib import Parallel, delayed

from .model_selection import TemporalSplit

_vstack = lambda x: np.concatenate(x, axis=0)
_hstack = lambda x: np.concatenate(x, axis=1)


class LinearCombination(BaseEstimator):
    """Estimator for optimal linear combitaion of predictions."""

    def loss(self, W, X, y, loss='mae'):
        if loss == 'mse':
            return np.mean((y - X.dot(W)) ** 2)
        elif loss == 'mae':
            return np.mean(np.abs(y - X.dot(W)))
        else:
            raise Exception

    def fit(self, X, y):
        n_cols = X.shape[1]
        W_start = np.ones(n_cols) / n_cols
        self.coef_ = fmin_slsqp(
            lambda w: self.loss(w, X=X, y=y),
            W_start,
            f_eqcons=lambda x: np.sum(x) - 1,
            bounds=[(0.0, 1.0)] * n_cols,
            iter=1_000,
            iprint=0,
            full_output=False,
        )
        return self

    def predict(self, X):
        return X.dot(self.coef_)

    def predict(self, X):
        return X.dot(self.coef_)


def _fit_recursive(fit_params, grouper, X, y,
                   sample_weight, y_mean, y_std):
    """Utility function to fit a single recursive forecaster."""

    model = ForecastingCascade(**fit_params)
    model.n_targets_ = y.shape[1] if len(y.shape) > 1 else 1
    model._validate_params()
    model._fit(grouper, X, y, sample_weight, y_mean, y_std)

    return model

def _fit_direct(fit_params, grouper, X, y,
                sample_weight, y_mean, y_std):
    """Utility function to fit a single direct forecaster."""

    model = ForecastingChain(**fit_params)
    model.n_targets_ = y.shape[1] if len(y.shape) > 1 else 1
    model._validate_params()
    model._fit(grouper, X, y, sample_weight, y_mean, y_std)

    return model


class BaseForecaster(BaseEstimator, RegressorMixin):
    """Base class for Forecasting meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self):
        pass

    def _get_feature_names(self, X, y):
        """Private method to extract feature names."""

        feature_names = []
        n_features = np.shape(X)[1] - len(self.groups_)
        if hasattr(X, "columns"):
            exog_names = X.columns
            exog_names = list(exog_names.difference(exog_names[self.groups_]))
        else:
            exog_names = \
                ['exog{}'.format(c) for c in range(n_features)]
        feature_names.extend([exog_names])

        if self.exog_lags_ is not None:
            if isinstance(self.exog_lags_, dict):
                feature_names.extend(
                    [['{}_lag{}'.format(exog_names[c], l) for l in lags]
                     for c, lags in self.exog_lags_.items()]
                )
            else:
                feature_names.extend(
                    [['{}_lag{}'.format(exog_names[c], l)
                      for c in range(n_features)]
                     for l in self.exog_lags_]
                )

        if hasattr(y, "columns"):
            feature_names.extend(
                [['{}_lag{}'.format(c, l)
                  for c in y.columns] for l in self.lags_]
            )
        elif hasattr(y, "name"):
            feature_names.extend(
                [['{}_lag{}'.format(y.name, l) for l in self.lags_]]
            )
        else:
            feature_names.extend(
                [['target{}_lag{}'.format(c, l)
                  for c in range(self.n_targets_)] for l in self.lags_]
            )

        self.feature_names_ = sum(feature_names, [])

    def _create_last_windows(self, grouper, X, y):
        """Private method to extract latest data features."""

        last_y, last_X = {}, {}
        y_mean, y_std = {}, {}
        for g_id, g_indexes in grouper.items():
            if self.exog_lags_ is None:
                last_X[g_id] = None
            else:
                _last_X = X[g_indexes][-self.min_exog_samples_:]
                if not self.accept_nan:
                    last_X[g_id] = _last_X
                else:
                    last_X[g_id] = np.pad(_last_X, pad_width=(
                        (max(self.min_exog_samples_ - _last_X.shape[0], 0), 0), (0, 0)
                    ), constant_values=np.nan)
            _last_y = y[g_indexes][-self.min_samples_:]
            if not self.accept_nan:
                last_y[g_id] = _last_y
            else:
                last_y[g_id] = np.pad(_last_y, pad_width=(
                    (max(self.min_samples_ - _last_y.shape[0], 0), 0), (0, 0)
                ), constant_values=np.nan)
            if self.target_standardize:
                if self.target_diff:
                    y_mean[g_id] = np.nanmean(
                        np.diff(y[g_indexes], axis=0, prepend=np.nan),
                        axis=0, keepdims=True
                    )
                    y_std[g_id] = np.nanstd(
                        np.diff(y[g_indexes], axis=0, prepend=np.nan),
                        axis=0, keepdims=True
                    )
                else:
                    y_mean[g_id] = np.nanmean(
                        y[g_indexes], axis=0, keepdims=True
                    )
                    y_std[g_id] = np.nanstd(
                        y[g_indexes], axis=0, keepdims=True
                    )
            else:
                y_mean[g_id] = None
                y_std[g_id] = None

        return last_y, last_X, y_mean, y_std

    def _validate_data_fit(self, _X, _y, sample_weight=None):
        """Private method to validate data for fitting."""

        X, y = self._validate_data(
            _X, _y,
            reset=True,
            validate_separately=(
                {
                    'accept_sparse': False,
                    'dtype': 'float32',
                    'force_all_finite': 'allow-nan' if self.accept_nan \
                        else True,
                    'ensure_2d': True,
                    'allow_nd': False,
                    'ensure_min_samples': 1,
                    'ensure_min_features': 1
                },
                {
                    'accept_sparse': False,
                    'dtype': 'float32',
                    'force_all_finite': 'allow-nan' if self.accept_nan \
                        else True,
                    'ensure_2d': False,
                    'allow_nd': False,
                    'ensure_min_samples': 1,
                    'ensure_min_features': 1
                }
            ),
        )
        check_consistent_length([X, y])
        self.n_targets_ = y.shape[1] if len(y.shape) > 1 else 1
        if min(self.groups_) < 0 or max(self.groups_) >= X.shape[1]:
            raise ValueError(
                "Invalid groups. groups must be "
                "an iterable of integers in [0, n_features_in_)."
            )
        if isinstance(self.exog_lags_, dict):
            if min(self.exog_lags_.keys()) < 0 or \
                    max(self.exog_lags_.keys()) >= (X.shape[1] - len(self.groups_)):
                raise ValueError(
                    "Invalid exog_lags. Keys must be "
                    "integers in [0, n_features_in_)."
                )
        self._get_feature_names(_X, _y)
        del _X, _y
        y = y.reshape(-1, self.n_targets_)

        grouper = pd.DataFrame(X[:, self.groups_]).groupby(
            list(range(len(self.groups_)))
        ).indices
        if max(map(len, grouper.values())) < self.min_samples_ + 1:
            raise ValueError(
                "At least a group with {} samples is required." \
                    .format(self.min_samples_ + 1)
            )
        X = X[:, np.setdiff1d(range(X.shape[1]), self.groups_)]

        if sample_weight is not None and \
                hasattr(self, 'estimator') and \
                has_fit_parameter(self.estimator, 'sample_weight'):
            sample_weight = _check_sample_weight(sample_weight, y)
        else:
            sample_weight = None

        last_y, last_X, y_mean, y_std = self._create_last_windows(grouper, X, y)

        return X, y, sample_weight, last_y, last_X, y_mean, y_std, grouper

    def _validate_data_predict(self, X, last_y=None, last_X=None):
        """Private method to validate data for prediction."""

        X = self._validate_data(
            X,
            reset=False,
            accept_sparse=False,
            dtype='float32',
            force_all_finite='allow-nan' if self.accept_nan \
                else True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_features=self.n_features_in_,
        )
        grouper = pd.DataFrame(X[:, self.groups_]).groupby(
            list(range(len(self.groups_)))
        ).indices
        X = X[:, np.setdiff1d(range(X.shape[1]), self.groups_)]

        if (last_y is None and last_X is not None) or \
                (last_y is not None and last_X is None):
            raise ValueError("Both last_X and last_y must be not None.")

        elif last_y is not None and last_X is not None:
            last_X, last_y = self._validate_data(
                last_X, last_y,
                reset=False,
                validate_separately=(
                    {
                        'accept_sparse': False,
                        'dtype': 'float32',
                        'force_all_finite': 'allow-nan' if self.accept_nan \
                            else True,
                        'ensure_2d': True,
                        'allow_nd': False,
                        'ensure_min_samples': 1,
                        'ensure_min_features': self.n_features_in_
                    },
                    {
                        'accept_sparse': False,
                        'dtype': 'float32',
                        'force_all_finite': 'allow-nan' if self.accept_nan \
                            else True,
                        'ensure_2d': self.n_targets_ > 1,
                        'allow_nd': False,
                        'ensure_min_samples': 1,
                        'ensure_min_features': self.n_targets_
                    }
                ),
            )
            check_consistent_length([last_X, last_y])
            last_y = last_y.reshape(-1, self.n_targets_)

            last_grouper = pd.DataFrame(last_X[:, self.groups_]).groupby(
                list(range(len(self.groups_)))
            ).indices
            last_X = last_X[:, np.setdiff1d(range(last_X.shape[1]), self.groups_)]

            if min(map(len, last_grouper.values())) < self.min_samples_:
                raise ValueError(
                    "When passing last_y and last_X, "
                    "a minimum of {} samples is required in each group." \
                        .format(self.min_samples_)
                )
            last_y, last_X, y_mean, y_std = self._create_last_windows(
                last_grouper, last_X, last_y
            )

        else:
            last_X = self._last_X.copy()
            last_y = self._last_y.copy()
            y_mean = self._y_mean.copy()
            y_std = self._y_std.copy()

        return X, last_y, last_X, y_mean, y_std, grouper

    def _validate_params(self):
        """Private method to validate params."""

        def _check_1d_iterable(param_name, iterable, min_val=1):
            """Validate 1D iterable of integeres."""

            msg = "Invalid {}. {} must be an iterable of integers >={}."
            msg = msg.format(param_name, param_name, min_val)
            if not np.iterable(iterable) or isinstance(iterable, str):
                raise ValueError(
                    msg + " Got {} ({}).".format(iterable, type(iterable))
                )

            iterable = np.unique(iterable).tolist()
            if len(iterable) < 0: raise ValueError(msg)
            for i in iterable:
                if not isinstance(i, numbers.Integral): raise ValueError(msg)
                if i < min_val: raise ValueError(msg)

            return iterable

        msg = "{} must be bool. Got {} ({})."
        if not isinstance(self.target_diff, bool):
            raise ValueError(
                msg.format('target_diff', self.target_diff, type(self.target_diff))
            )
        if not isinstance(self.accept_nan, bool):
            raise ValueError(
                msg.format('accept_nan', self.accept_nan, type(self.accept_nan))
            )
        self.groups_ = _check_1d_iterable('groups', self.groups, min_val=0)

        if self.__class__.__name__ in ['ForecastingStacked', 'ForecastingRectified']:
            if not isinstance(self.estimators, (list, tuple, set)):
                self.estimators = [self.estimators]

            for est in self.estimators:
                if not hasattr(est, 'fit') or not hasattr(est, 'predict'):
                    raise ValueError(
                        "estimators must implement fit and predict methods. "
                        "{} doesn't.".format(est)
                    )

            if self.test_size is not None and \
                    not isinstance(self.test_size, (numbers.Integral, numbers.Real)):
                raise ValueError(
                    "Expected test_size as integer, float, or None. Got {} ({})." \
                        .format(self.test_size, type(self.test_size))
                )

        self.lags_ = _check_1d_iterable("lags", self.lags)
        self.min_ar_samples_ = max(self.lags_)
        if self.exog_lags is not None:
            msg = ("Invalid exog_lags. exog_lags must be a dictionary "
                   "in the format {int: iterable} or an iterable of integers.")
            if isinstance(self.exog_lags, dict):
                self.exog_lags_ = {}
                for c, lags in self.exog_lags.items():
                    if not isinstance(c, numbers.Integral): raise ValueError(msg)
                    self.exog_lags_[c] = _check_1d_iterable("exog_lags", lags)
                _id = np.searchsorted(self.groups_, list(self.exog_lags_.keys()))
                for i, (c, lags) in zip(_id.tolist(), self.exog_lags_.items()):
                    if c in self.groups_:
                        raise ValueError("Can't use exog_lags on groups.")
                    del self.exog_lags_[c]
                    self.exog_lags_[c - i] = lags
                self.min_exog_samples_ = max(sum(self.exog_lags_.values(), []))
            elif np.iterable(self.exog_lags) and not isinstance(self.exog_lags, str):
                self.exog_lags_ = _check_1d_iterable("exog_lags", self.exog_lags)
                self.min_exog_samples_ = max(self.exog_lags_)
            else:
                raise ValueError(msg)
        else:
            self.exog_lags_ = None
            self.min_exog_samples_ = 0

        self.chunk_size_ = min(self.lags_)
        self.min_samples_ = max(self.min_ar_samples_, self.min_exog_samples_)
        self.min_samples_ += int(self.target_diff)

        if self.__class__.__name__ in ['ForecastingChain', 'ForecastingRectified']:
            msg = "Invalid n_estimators. n_estimators must be an integer >0 " \
                  "or an iterable of integers."
            if isinstance(self.n_estimators, numbers.Integral):
                if self.n_estimators < 1: raise ValueError(msg)
                self.estimator_range_ = list(range(self.n_estimators))
            else:
                self.estimator_range_ = _check_1d_iterable('n_estimators', self.n_estimators)
                self.estimator_range_ = [0] + self.estimator_range_[:]
            self.min_samples_ += self.chunk_size_ * max(self.estimator_range_)

    def fit(self, X, y, sample_weight=None):
        """Fit a scikit-learn timeseries forecasting estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Exogenous feature series in ascending temporal order to concatenate
            with lagged target features.
            It must contain the group columns at the positions specified in the
            groups parameter.
            An array-like with at least a group with n_samples equal to min_samples_
            is required.

        y : array-like of shape (n_samples,) or also (n_samples, n_targets) if
            multiple outputs
            Series to forecast in ascending temporal order.
            An array-like with at least a group with n_samples equal to min_samples_
            is required.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Effective only if estimator accepts sample_weight.

        Returns
        -------
        self : object
        """

        self._validate_params()
        X, y, sample_weight, last_y, last_X, y_mean, y_std, grouper = \
            self._validate_data_fit(X, y, sample_weight)
        self._fit(grouper, X, y, sample_weight, y_mean, y_std)
        self._last_y, self._last_X = last_y, last_X
        self._y_mean, self._y_std = y_mean, y_std

        return self

    def _groupby_predict(self, grouper, X, last_y, last_X,
                         y_mean, y_std, **predict_params):
        """Private method to generate prediction from group labels."""

        preds = np.full((X.shape[0], self.n_targets_), np.nan) \
            if self.n_targets_ > 1 else np.full((X.shape[0],), np.nan)

        def _predict(
                g_id, X,
                last_y, last_X,
                y_mean, y_std,
                predict_params
        ):
            if g_id not in last_y: return None
            return self._predict(
                X=X,
                last_y=last_y[g_id], last_X=last_X[g_id],
                y_mean=y_mean[g_id], y_std=y_std[g_id],
                **predict_params
            )

        g_preds = [
            _predict(
                g_id,
                X[g_indexes],
                last_y, last_X,
                y_mean, y_std,
                predict_params
            )
            for g_id, g_indexes in grouper.items()
        ]

        for i, (g_id, g_indexes) in enumerate(grouper.items()):
            pred = g_preds[i]
            if pred is None:
                warn_msg = "Group '{}' not found in last_X and last_y," \
                           " returning NaN predictions.".format(g_id)
                warnings.warn(warn_msg)
                continue
            preds[g_indexes] = pred

        return preds

    def predict(self, X, last_y=None, last_X=None, **predict_params):
        """Forecast future values.

        If last_y and last_X are not received (None), the latest values from the
        train data are used to initialize lag features creation and generate forecasts
        for the groups seen in training.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Exogenous feature series in ascending temporal order to concatenate
            with lagged target features.
            It must contain the group columns at the positions specified in the
            groups parameter.
            An array-like with at least a group with n_samples equal to min_samples_
            is required.

        last_y : array-like of shape (n_samples,) or also (n_samples, n_targets) if
            multiple outputs, default=None
            Useful past target values in ascending temporal order to initialize feature
            lags creation.
            All the groups must have at leats n_samples equal to min_samples_.
            When not None with target_standarize=True, last_y is used to build mean and
            standard deviation useful for standardization.

        last_X : array-like of shape (n_samples, n_features), default=None
            Useful past exogenous feature in ascending temporal order to initialize
            feature lags creation (when exog_lags is not None).
            All the groups must have at leats n_samples equal to min_samples_.
            It must have the same schema of X. It must contain the group columns at the
            positions specified in the groups parameter.

        **predict_params : Additional prediction arguments used by estimator_

        Returns
        -------
        preds : ndarray of shape (n_samples,) or also (n_samples, n_targets) if
            multiple outputs
            Forecast values for each groups.
            It's possible to have flatten forecasts (the latest values seen in train repeted)
            for some groups. This happens when last_y and last_X are None and a group
            doesn't have enough samples in training.
        """

        check_is_fitted(self, 'n_targets_')
        X, last_y, last_X, y_mean, y_std, grouper = \
            self._validate_data_predict(X, last_y, last_X)
        preds = self._groupby_predict(grouper, X, last_y, last_X, y_mean, y_std)

        return preds

    def score(self, X, y, sample_weight=None,
              scoring='mse', uniform_average=True,
              **predict_params):
        """Return the selected score on the given test data and labels
        for the groups seen in training.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Exogenous feature series in ascending temporal order to concatenate
            with lagged target features.
            It must contain the group columns at the positions specified in the
            groups parameter.
            An array-like with at least a group with n_samples equal to min_samples_
            is required.

        y : array-like of shape (n_samples,) or also (n_samples, n_targets) if
            multiple outputs
            Series to forecast in ascending temporal order.
            An array-like with at least a group with n_samples equal to min_samples_
            is required.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        scoring : str, default='mse'
            Scoring function used.
            Supported scorings are:
            - 'mse' (mean squared error),
            - 'rmse' (root mean squared error),
            - 'mae' (mean absolute percentage error),
            - 'mape' (mean absolute percentage error),
            - 'r2' (coefficient of determination).

        uniform_average : bool, default=True
            Average the scores with uniform weights in case of multiple outputs.

        **predict_params : Additional prediction arguments

        Returns
        -------
        score : float or ndarray of floats
            A floating point value or an array of floating point values,
            in case of multiple outputs.
        """

        scores = ['mse', 'rmse', 'mae', 'mape', 'r2']
        multioutput = 'uniform_average' if uniform_average else 'raw_values'

        pred = self.predict(X, **predict_params)

        if scoring == 'mse':
            from sklearn.metrics import mean_squared_error as mse
            return mse(
                y, pred, sample_weight=sample_weight,
                multioutput=multioutput
            )
        elif scoring == 'rmse':
            from sklearn.metrics import mean_squared_error as mse
            return mse(
                y, pred, sample_weight=sample_weight,
                squared=False, multioutput=multioutput
            )
        elif scoring == 'mae':
            from sklearn.metrics import mean_absolute_error as mae
            return mae(
                y, pred, sample_weight=sample_weight,
                multioutput=multioutput
            )
        elif scoring == 'mape':
            from sklearn.metrics import mean_absolute_percentage_error as mape
            return mape(
                y, pred, sample_weight=sample_weight,
                multioutput=multioutput
            )
        elif scoring == 'r2':
            from sklearn.metrics import r2_score as r2
            return r2(
                y, pred, sample_weight=sample_weight,
                multioutput=multioutput
            )
        else:
            raise ValueError(
                "{} not recogized as scoring. "
                "Valid scoring are {}".format(scoring, scores)
            )


class ForecastingCascade(BaseForecaster):
    """Recursive Time Series Forecaster

    Scikit-learn estimator for global recursive time series forecasting.
    It automatically builds lag features on targets and optionally also on
    exogenous features. Target lag features are combined with exogenous
    regressors (if provided) and lagged exogenous features (if specified).
    Finally, a scikit-learn compatible regressor is fitted on the whole merged data.
    The fitted estimator is called iteratively to predict multiple steps ahead.
    Multivariate timeseries forecasting is natively supported. If the received
    scikit-learn model doesn't support multivariate targets, use it in
    MultiOutputRegressor.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator compatible with scikit-learn.

    lags : iterable
        1D iterable of integers representing the lag features to create from
        the target.

    groups : iterable
        1D iterable of integers representing the column indexes used to separate
        the received time series into groups.

    exog_lags : iterable or dict, default=None
        1D iterable of integers representing the lag features to create from
        the exogenous features. In that case, lagged features are created for
        all the received features simultaneously.
        if dict: keys represent the index of the desired features from which
        create lags, while values must be 1D iterable of integers representing the
        lag features to create.

    target_diff : bool, deafult=False
        Apply the first-order differentiation on the target.
        Lagged target features are created on differenced target values.

    target_standardize : bool, deafult=False
        Apply standardization on targets by removing the mean and scaling to
        unit variance.
        Lagged target features are created on standardized target values.
        When used with target_diff=True, the standardization is applied on the
        differenced target values.

    accept_nan : bool, default=False
        Use or ignore NaNs target observations during fitting.
        At least one not-nan observation is required, after lags creation,
        to make a successful fit.

    Attributes
    ----------
    n_targets_ : int
        Number of detected targets to forecast.

    lags_ : list of integers
        Lag features to create from the target.

    groups_ : list of integers
        Column indexes present in X and used to separate time series into groups.

    exog_lags_ : list of integers, dict, or None
        Lag features to create from the exogenous features.

    chunk_size_ : int
        How many samples can be processed and forecasted simultaneously.
        Automatically detected from lags_ and exog_lags_.

    min_samples_ : int
        Minimum number of samples required to create desired lag features and
        operate a fit or a forecast.
        Automatically detected from lags_ and exog_lags_.

    min_ar_samples_ : int
        Minimum number of samples required to create desired lag features from
        the target.
        Automatically detected from lags_.

    min_exog_samples_ : int or None
        Minimum number of samples required to create desired lag features from
        the exogenous features.
        Automatically detected from exog_lags_.

    estimator_ : estimator
        A fitted scikit-learn estimator instance on the lagged features.

    feature_names_ : list of strings
        A list containing the name of features (including the lagged ones)
        used to train the estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from tspiral.forecasting import ForecastingCascade
    >>> timesteps = 400
    >>> e = np.random.normal(0,1, (timesteps,))
    >>> y = 2*np.sin(np.arange(timesteps)*(2*np.pi/24))+e
    >>> X = np.zeros((timesteps,1))
    >>> model = ForecastingCascade(
    ...     Ridge(),
    ...     lags=range(1,24+1),
    ...     groups=[0],
    ...     accept_nan=False
    ... )
    >>> model.fit(X, y)
    >>> forecasts = model.predict(np.zeros((24*3,1)))
    """

    def __init__(
            self,
            estimator,
            *,
            lags,
            groups,
            exog_lags=None,
            target_diff=False,
            target_standardize=False,
            accept_nan=False,
    ):
        self.estimator = estimator
        self.lags = lags
        self.groups = groups
        self.exog_lags = exog_lags
        self.groups = groups
        self.target_diff = target_diff
        self.target_standardize = target_standardize
        self.accept_nan = accept_nan

    def _create_lags(self, X, lags, end, start=0):
        """Private method to create lag features."""

        return _hstack([X[(end - l):-(l - start) or None] for l in lags])

    def _fit(self, grouper, X, y, sample_weight, y_mean, y_std):
        """Private method for recursive forecasting train."""

        def _transform(g_id, X, y, sample_weight):

            if isinstance(self.exog_lags_, dict):
                X = _hstack(
                    [X[self.min_samples_:]] + \
                    [self._create_lags(X[:, [c]], lags, self.min_samples_)
                     for c, lags in self.exog_lags_.items()]
                )
            elif np.iterable(self.exog_lags_):
                X = _hstack(
                    [X[self.min_samples_:]] + \
                    [self._create_lags(X, self.exog_lags_, self.min_samples_)]
                )
            else:
                X = X[self.min_samples_:]

            if self.target_diff:
                y = np.diff(y, axis=0, prepend=np.nan)
            if self.target_standardize:
                y = (y - y_mean[g_id]) / y_std[g_id]
            Xar = self._create_lags(y, self.lags_, self.min_samples_)
            y = y[self.min_samples_:]
            X = _hstack([X, Xar])
            if sample_weight is not None:
                sample_weight = sample_weight[self.min_samples_:]

            return X, y, sample_weight

        group_trans = [
            _transform(
                g_id,
                X[g_indexes],
                y[g_indexes],
                (None if sample_weight is None
                 else sample_weight[g_indexes])
            )
            for g_id, g_indexes in grouper.items()
        ]

        X, y, sample_weight = [], [], []
        for g_id, g_trans in zip(grouper.keys(), group_trans):
            X.append(g_trans[0])
            y.append(g_trans[1])
            sample_weight.append(g_trans[2])
        X, y = _vstack(X), _vstack(y)
        sample_weight = None if sample_weight[0] is None \
            else _vstack(sample_weight)
        del group_trans, grouper

        if self.accept_nan:
            mask = ~(np.isnan(y).any(1))
            X, y = X[mask], y[mask]
            if sample_weight is not None: sample_weight = sample_weight[mask]
            if y.shape[0] < 1: raise Exception("y contains only NaNs.")

        y = y if self.n_targets_ > 1 else y.ravel()
        self.estimator_ = clone(self.estimator)
        if sample_weight is None:
            self.estimator_.fit(X, y)
        else:
            self.estimator_.fit(X, y, sample_weight=sample_weight)

        return self

    def _predict(self, X, last_y, last_X, y_mean, y_std, **predict_params):
        """Private method for recursive forecasting."""

        def _predict_chunk(X, last_y, **predict_params):

            y = last_y[-self.min_ar_samples_:]
            Xar = self._create_lags(
                y, self.lags_,
                end=self.min_ar_samples_, start=X.shape[0]
            )
            X = _hstack([X, Xar])

            pred = self.estimator_.predict(X, **predict_params)
            last_y = _vstack([y, pred.reshape(-1, self.n_targets_)])

            return pred, last_y

        latest_y = last_y[[-1]]
        if self.target_diff: last_y = np.diff(last_y, axis=0)
        if self.target_standardize: last_y = (last_y - y_mean) / y_std

        if X.size > 0 and last_X is not None:
            if isinstance(self.exog_lags_, dict):
                X = _vstack([last_X, X])
                X = _hstack(
                    [X[self.min_exog_samples_:]] + \
                    [self._create_lags(X[:, [c]], lags, self.min_exog_samples_)
                     for c, lags in self.exog_lags_.items()]
                )
            elif np.iterable(self.exog_lags_):
                X = _vstack([last_X, X])
                X = _hstack(
                    [X[self.min_exog_samples_:]] + \
                    [self._create_lags(X, self.exog_lags_, self.min_exog_samples_)]
                )

        n_preds = X.shape[0]
        if last_y.shape[0] < (self.min_samples_ - int(self.target_diff)):
            preds = np.full((n_preds, self.n_targets_), latest_y)
        else:
            idchunks = list(range(self.chunk_size_, n_preds, self.chunk_size_))
            n_chunks = len(idchunks) + 1
            idchunks = [0] + idchunks + [n_preds]

            preds = []
            for i in range(n_chunks):
                pred, last_y = _predict_chunk(
                    X[idchunks[i]:idchunks[i + 1]],
                    last_y,
                    **predict_params
                )
                preds.append(pred)
            preds = _vstack(preds).reshape(-1, self.n_targets_)
            if self.target_standardize:
                preds = (preds * y_std) + y_mean
            if self.target_diff:
                preds = _vstack([latest_y, preds])
                preds = np.cumsum(preds, axis=0)[1:]

        preds = preds if self.n_targets_ > 1 else preds.ravel()

        return preds


class ForecastingChain(BaseForecaster):
    """Direct Time Series Forecaster

    Scikit-learn estimator for global direct time series forecasting.
    It automatically builds lag features on target and optionally also on
    exogenous features. Target lag features are combined with exogenous
    regressors (if provided) and lagged exogenous features (if specified).
    Finally, a scikit-learn compatible regressor is fitted on the whole merged data
    for each forecast time step.
    Multivariate timeseries forecasting is natively supported. If the received
    scikit-learn model doesn't support multivariate targets, use it in
    MultiOutputRegressor.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator compatible with scikit-learn.

    n_estimators : int or iterable
        How many estimators fit to forecast future time steps.
        Each estimator is fitted independently from each other and correspond to
        a future time step to forecast.
        When received an integer, a model for each forecasting horizont in
        the range [1, n_estimators) is build to make predictions.
        When received a interable of integers, a model for each specified
        forecasting horizont is build. The time horizonts not declared are
        forecated in recursive manner using the previous available forecaster.
        Outside the n_estimators limit (or max(n_estimators) in case of an iterable),
        forecasting is made in a recursive manner using the last fitted estimator.

    lags : iterable
        1D iterable of integers representing the lag features to create from
        the target.

    groups : iterable
        1D iterable of integers representing the column indexes used to separate
        the received time series into groups.

    exog_lags : iterable or dict, default=None
        1D iterable of integers representing the lag features to create from
        the exogenous features. In that case, lagged features are created for
        all the received features simultaneously.
        if dict: keys represent the index of the desired features from which
        create lags, while values must be 1D iterable of integers representing the
        lag features to create.

    use_exog : bool, default=False
        Use or ignore the received features during fitting as X.
        If False, exog_lags are ignored.

    target_diff : bool, deafult=False
        Apply the first-order differentiation on the target.
        Lagged target features are created on differenced target values.

    target_standardize : bool, deafult=False
        Apply standardization on targets by removing the mean and scaling to
        unit variance.
        Lagged target features are created on standardized target values.
        When used with target_diff=True, the standardization is applied on the
        differenced target values.

    accept_nan : bool, default=False
        Use or ignore NaNs target observations during fitting.
        At least one not-nan observation is required, after lags creation,
        to make a successful fit.

    n_jobs : int, default=None
        The number of jobs to run in parallel for model fitting.
        ``None`` means 1 using one processor. ``-1`` means using all
        processors.

    verbose : int, default=0
        Verbosity mode.

    Attributes
    ----------
    n_targets_ : int
        Number of detected targets to forecast.

    lags_ : list of integers
        Lag features to create from the target.

    groups_ : list of integers
        Column indexes present in X and used to separate time series into groups.

    exog_lags_ : list of integers, dict, or None
        Lag features to create from the exogenous features.

    chunk_size_ : int
        How many samples can be processed and forecasted simultaneously.
        Automatically detected from lags_ and exog_lags_.

    min_samples_ : int
        Minimum number of samples required to create desired lag features and
        operate a fit or a forecast.
        Automatically detected from lags_ and exog_lags_.

    min_ar_samples_ : int
        Minimum number of samples required to create desired lag features from
        the target.
        Automatically detected from lags_.

    min_exog_samples_ : int or None
        Minimum number of samples required to create desired lag features from
        the exogenous features.
        Automatically detected from exog_lags_.

    estimators_ : list
        A list of fitted scikit-learn estimator instances on the lagged features.

    estimator_range_ : list
        A list of integers representing forecasting horizons on which a recursive
        estimator is fitted.

    feature_names_ : list of strings
        A list containing the name of features (including the lagged ones)
        used to train the estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from tspiral.forecasting import ForecastingChain
    >>> timesteps = 400
    >>> e = np.random.normal(0,1, (timesteps,))
    >>> y = 2*np.sin(np.arange(timesteps)*(2*np.pi/24))+e
    >>> X = np.zeros((timesteps,1))
    >>> model = ForecastingChain(
    ...     Ridge(),
    ...     n_estimators=24,
    ...     lags=range(1,24+1),
    ...     groups=[0],
    ...     accept_nan=False
    ... )
    >>> model.fit(X, y)
    >>> forecasts = model.predict(np.zeros((24*3,1)))
    """

    def __init__(
            self,
            estimator,
            *,
            n_estimators,
            lags,
            groups,
            exog_lags=None,
            target_diff=False,
            target_standardize=False,
            accept_nan=False,
            n_jobs=None,
            verbose=0
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.lags = lags
        self.groups = groups
        self.exog_lags = exog_lags
        self.target_diff = target_diff
        self.target_standardize = target_standardize
        self.accept_nan = accept_nan
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit(self, grouper, X, y, sample_weight, y_mean, y_std):
        """Private method for direct forecasting train."""

        fit_params = signature(ForecastingCascade.__init__).parameters.keys()
        fit_params = signature(self.__init__).parameters.keys() & fit_params
        fit_params = {p: self.__dict__[p] for p in fit_params}

        self.estimators_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )(
            delayed(_fit_recursive)(
                fit_params=(
                    lambda p: p.update({
                        'lags': list(map(
                            lambda l: l + self.chunk_size_ * e, self.lags_
                        ))
                    }) or p
                )(fit_params.copy()),
                X=X, y=y,
                sample_weight=sample_weight,
                y_mean=y_mean, y_std=y_std,
                grouper=grouper,
            )
            for e in self.estimator_range_
        )

        return self

    def _predict(self, X, last_y, last_X, y_mean, y_std, **predict_params):
        """Private method for direct forecasting."""

        n_preds = X.shape[0]
        n_estimators = np.unique(self.estimator_range_ + [n_preds])
        n_estimators = np.diff(n_estimators).tolist()

        if self.n_targets_ > 1:
            pred_vector = np.full((n_preds, self.n_targets_), np.nan)
        else:
            pred_vector = np.full((n_preds,), np.nan)

        pred_matrix = []
        istart, iend = 0, 0
        for i, e in enumerate(n_estimators):
            iend = istart + max(self.chunk_size_, e)
            pred = pred_vector.copy()
            _pred = self.estimators_[i]._predict(
                X=X[:iend],
                last_y=last_y,
                last_X=last_X,
                y_mean=y_mean,
                y_std=y_std,
                **predict_params
            )
            if _pred is None:
                return None
            else:
                pred[istart:iend] = _pred[istart:]
            pred_matrix.append([pred])
            if istart >= n_preds: break
            istart += e

        pred_matrix = np.nanmedian(_vstack(pred_matrix), axis=0)

        return pred_matrix


class BaseEnsembleForecaster(BaseForecaster):
    """Base class for ensemble Forecasting meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self):
        pass

    def _fit(self, grouper, X, y, sample_weight, y_mean, y_std):
        """Private method for stacked forecasting train."""

        if self.__class__.__name__ == 'ForecastingStacked':
            fit_params = signature(ForecastingCascade.__init__).parameters.keys()
            _fit_one = _fit_recursive
        else:
            fit_params = signature(ForecastingChain.__init__).parameters.keys()
            _fit_one = _fit_direct
        fit_params = signature(self.__init__).parameters.keys() & fit_params
        fit_params = {p: self.__dict__[p] for p in fit_params}

        if self.test_size is None:
            self.final_estimator_ = None
        else:
            groups = _vstack([np.full_like(g_id, i)
                              for i, g_id in enumerate(grouper.values())])
            splitter = TemporalSplit(1, test_size=self.test_size)
            id_train, id_test = next(splitter.split(X, None, groups))
            grouper_train = pd.DataFrame(groups[id_train]).groupby([0]).indices
            grouper_test = pd.DataFrame(groups[id_test]).groupby([0]).indices
            last_y, last_X, y_mean, y_std = self._create_last_windows(
                grouper_train, X[id_train], y[id_train]
            )

            preds = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )(
                delayed(
                    _fit_one(
                        fit_params=(
                            lambda p: p.update({
                                'estimator': est
                            }) or p
                        )(fit_params.copy()),
                        X=X[id_train],
                        y=y[id_train],
                        sample_weight=(sample_weight if sample_weight is None
                                       else sample_weight[id_train]),
                        y_mean=y_mean, y_std=y_std,
                        grouper=grouper_train,
                    )._groupby_predict
                )(
                    X=X[id_test],
                    last_y=last_y, last_X=last_X,
                    y_mean=y_mean, y_std=y_std,
                    grouper=grouper_test,
                )
                for est in self.estimators
            )

            del groups, grouper_train, grouper_test
            preds = _vstack([[p.reshape(-1, self.n_targets_)] for p in preds])
            preds = preds.transpose((2, 1, 0))  # (n_targets, n_samples, n_estimators)
            final_estimator = clone(self.final_estimator) \
                if self.final_estimator is not None else LinearCombination()
            self.final_estimator_ = [
                clone(final_estimator).fit(preds[t], y[id_test, t])
                for t in range(self.n_targets_)
            ]
            del preds

        last_y, last_X, y_mean, y_std = self._create_last_windows(grouper, X, y)
        self.estimators_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )(
            delayed(_fit_one)(
                fit_params=(
                    lambda d: d.update({
                        'estimator': est
                    }) or d
                )(fit_params.copy()),
                X=X, y=y,
                sample_weight=sample_weight,
                y_mean=y_mean, y_std=y_std,
                grouper=grouper,
            )
            for est in self.estimators
        )

        return self

    def _predict(self, X, last_y, last_X, y_mean, y_std, **predict_params):
        """Private method for stacked forecasting."""

        preds = _vstack([
            [
                est._predict(
                    X=X,
                    last_y=last_y, last_X=last_X,
                    y_mean=y_mean, y_std=y_std,
                ).reshape(-1, self.n_targets_)
            ]
            for est in self.estimators_
        ])
        preds = preds.transpose((2, 1, 0))  # (n_targets, n_samples, n_estimators)

        if self.final_estimator_ is not None:
            preds = _vstack([
                est.predict(preds[t], **predict_params)
                for t, est in enumerate(self.final_estimator_)
            ])
            preds = preds.reshape(len(preds), -1).T
        else:
            preds = np.median(preds, axis=-1).T

        preds = preds if self.n_targets_ > 1 else preds.ravel()

        return preds


class ForecastingStacked(BaseEnsembleForecaster):
    """Stacked Recursive Time Series Forecaster

    Scikit-learn estimator for stacked global time series forecasting.
    It fits multiple recursive time series forecasters (using ForecastingCascade)
    and combines them with a meta-learner.
    Multivariate timeseries forecasting is natively supported. If the received
    scikit-learn models doesn't support multivariate targets, use it in
    MultiOutputRegressor.

    Parameters
    ----------
    estimators : object or iterable of objects
        A single or list of supervised learning estimators compatible with
        scikit-learn.

    test_size : float, int, or None, default=None
        Final portion of data used to make forecasts and fit the final_estimator.
        If float, should be between 0.0 and 1.0 and represents the final
        proportion of the dataset used to fit final_estimator.
        If int, represents the absolute number of last test samples used to
        fit final_estimator.
        If None, no final_estimator is fitted and estimator predictions are
        combined with the median.

    lags : iterable
        1D iterable of integers representing the lag features to create from
        the target.

    groups : iterable
        1D iterable of integers representing the column indexes used to separate
        the received time series into groups.

    exog_lags : iterable or dict, default=None
        1D iterable of integers representing the lag features to create from
        the exogenous features. In that case, lagged features are created for
        all the received features simultaneously.
        if dict: keys represent the index of the desired features from which
        create lags, while values must be 1D iterable of integers representing the
        lag features to create.

    target_diff : bool, deafult=False
        Apply the first-order differentiation on the target.
        Lagged target features are created on differenced target values.

    target_standardize : bool, deafult=False
        Apply standardization on targets by removing the mean and scaling to
        unit variance.
        Lagged target features are created on standardized target values.
        When used with target_diff=True, the standardization is applied on the
        differenced target values.

    final_estimator : object, default=None
        A regressor which will be used to combine the recursive forecasts.
        When final_estimator is None predictions are combined using a
        LinearCombination estimator.
        Effetctive only if test_size is not None.

    n_jobs : int, default=None
        The number of jobs to run in parallel for model fitting.
        ``None`` means 1 using one processor. ``-1`` means using all
        processors.

    verbose : int, default=0
        Verbosity mode.

    Attributes
    ----------
    n_targets_ : int
        Number of detected targets to forecast.

    lags_ : list of integers
        Lag features to create from the target.

    groups_ : list of integers
        Column indexes present in X and used to separate time series into groups.

    exog_lags_ : list of integers, dict, or None
        Lag features to create from the exogenous features.

    chunk_size_ : int
        How many samples can be processed and forecasted simultaneously.
        Automatically detected from lags_ and exog_lags_.

    min_samples_ : int
        Minimum number of samples required to create desired lag features and
        operate a fit or a forecast.
        Automatically detected from lags_ and exog_lags_.

    min_ar_samples_ : int
        Minimum number of samples required to create desired lag features from
        the target.
        Automatically detected from lags_.

    min_exog_samples_ : int or None
        Minimum number of samples required to create desired lag features from
        the exogenous features.
        Automatically detected from exog_lags_.

    estimators_ : list
        A list of fitted scikit-learn estimator instances on the lagged features.

    final_estimator_ : list or None
        A list of final fitted estimators (one for each target) used to stack
        the estimators' forecasts.

    feature_names_ : list of strings
        A list containing the name of features (including the lagged ones)
        used to train the estimators.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from tspiral.forecasting import ForecastingStacked
    >>> timesteps = 400
    >>> e = np.random.normal(0,1, (timesteps,))
    >>> y = 2*np.sin(np.arange(timesteps)*(2*np.pi/24))+e
    >>> X = np.zeros((timesteps,1))
    >>> model = ForecastingStacked(
    ...     [Ridge(), DecisionTreeRegressor()],
    ...     test_size=24*3,
    ...     lags=range(1,24+1),
    ...     groups=[0]
    ... )
    >>> model.fit(X, y)
    >>> forecasts = model.predict(np.zeros((24*3,1)))
    """

    def __init__(
            self,
            estimators,
            *,
            test_size,
            lags,
            groups,
            exog_lags=None,
            target_diff=False,
            target_standardize=False,
            final_estimator=None,
            n_jobs=None,
            verbose=0
    ):
        self.estimators = estimators
        self.test_size = test_size
        self.lags = lags
        self.groups = groups
        self.exog_lags = exog_lags
        self.target_diff = target_diff
        self.target_standardize = target_standardize
        self.accept_nan = False
        self.final_estimator = final_estimator
        self.n_jobs = n_jobs
        self.verbose = verbose


class ForecastingRectified(BaseEnsembleForecaster):
    """Stacked Direct Time Series Forecaster

    Scikit-learn estimator for stacked global time series forecasting.
    It fits multiple direct time series forecasters (using ForecastingChain)
    and combines them with a meta-learner.
    Multivariate timeseries forecasting is natively supported. If the received
    scikit-learn models doesn't support multivariate targets, use it in
    MultiOutputRegressor.

    Parameters
    ----------
    estimators : object or iterable of objects
        A single or list of supervised learning estimators compatible with
        scikit-learn.

    test_size : float, int, or None, default=None
        Final portion of data used to make forecasts and fit the final_estimator.
        If float, should be between 0.0 and 1.0 and represents the final
        proportion of the dataset used to fit final_estimator.
        If int, represents the absolute number of last test samples used to
        fit final_estimator.
        If None, no final_estimator is fitted and estimator predictions are
        combined with the median.

    n_estimators : int or iterable
        How many estimators fit to forecast future time steps.
        Each estimator is fitted independently from each other and correspond to
        a future time step to forecast.
        When received an integer, a model for each forecasting horizont in
        the range [1, n_estimators) is build to make predictions.
        When received a interable of integers, a model for each specified
        forecasting horizont is build. The time horizonts not declared are
        forecated in recursive manner using the previous available forecaster.
        Outside the n_estimators limit (or max(n_estimators) in case of an iterable),
        forecasting is made in a recursive manner using the last fitted estimator.

    lags : iterable
        1D iterable of integers representing the lag features to create from
        the target.

    groups : iterable
        1D iterable of integers representing the column indexes used to separate
        the received time series into groups.

    exog_lags : iterable or dict, default=None
        1D iterable of integers representing the lag features to create from
        the exogenous features. In that case, lagged features are created for
        all the received features simultaneously.
        if dict: keys represent the index of the desired features from which
        create lags, while values must be 1D iterable of integers representing the
        lag features to create.

    use_exog : bool, default=False
        Use or ignore the received features during fitting as X.
        If False, exog_lags are ignored.

    target_diff : bool, deafult=False
        Apply the first-order differentiation on the target.
        Lagged target features are created on differenced target values.

    target_standardize : bool, deafult=False
        Apply standardization on targets by removing the mean and scaling to
        unit variance.
        Lagged target features are created on standardized target values.
        When used with target_diff=True, the standardization is applied on the
        differenced target values.

    final_estimator : object, default=None
        A regressor which will be used to combine the direct forecasts.
        When final_estimator is None predictions are combined using a
        LinearCombination estimator.
        Effetctive only if test_size is not None.

    n_jobs : int, default=None
        The number of jobs to run in parallel for model fitting.
        ``None`` means 1 using one processor. ``-1`` means using all
        processors.

    verbose : int, default=0
        Verbosity mode.

    Attributes
    ----------
    n_targets_ : int
        Number of detected targets to forecast.

    lags_ : list of integers
        Lag features to create from the target.

    groups_ : list of integers
        Column indexes present in X and used to separate time series into groups.

    exog_lags_ : list of integers, dict, or None
        Lag features to create from the exogenous features.

    chunk_size_ : int
        How many samples can be processed and forecasted simultaneously.
        Automatically detected from lags_ and exog_lags_.

    min_samples_ : int
        Minimum number of samples required to create desired lag features and
        operate a fit or a forecast.
        Automatically detected from lags_ and exog_lags_.

    min_ar_samples_ : int
        Minimum number of samples required to create desired lag features from
        the target.
        Automatically detected from lags_.

    min_exog_samples_ : int or None
        Minimum number of samples required to create desired lag features from
        the exogenous features.
        Automatically detected from exog_lags_.

    estimators_ : list
        A list of fitted scikit-learn estimator instances on the lagged features.

    final_estimator_ : list or None
        A list of final fitted estimators (one for each target) used to stack
        the estimators' forecasts.

    feature_names_ : list of strings
        A list containing the name of features (including the lagged ones)
        used to train the estimators.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from tspiral.forecasting import ForecastingRectified
    >>> timesteps = 400
    >>> e = np.random.normal(0,1, (timesteps,))
    >>> y = 2*np.sin(np.arange(timesteps)*(2*np.pi/24))+e
    >>> X = np.zeros((timesteps,1))
    >>> model = ForecastingRectified(
    ...     [Ridge(), DecisionTreeRegressor()],
    ...     test_size=24*3,
    ...     n_estimators=24*3,
    ...     lags=range(1,24+1),
    ...     groups=[0]
    )
    >>> model.fit(X, y)
    >>> forecasts = model.predict(np.zeros((24*3,1)))
    """

    def __init__(
            self,
            estimators,
            *,
            n_estimators,
            test_size,
            lags,
            groups,
            exog_lags=None,
            target_diff=False,
            target_standardize=False,
            final_estimator=None,
            n_jobs=None,
            verbose=0
    ):
        self.estimators = estimators
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.lags = lags
        self.groups = groups
        self.exog_lags = exog_lags
        self.target_diff = target_diff
        self.target_standardize = target_standardize
        self.accept_nan = False
        self.final_estimator = final_estimator
        self.n_jobs = n_jobs
        self.verbose = verbose