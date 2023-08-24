import numbers
import numpy as np
from scipy.optimize import fmin_slsqp

from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.utils.validation import \
    _check_sample_weight, has_fit_parameter, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

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


def _fit_recursive(est, X, y, sample_weight=None,
                   fit_params={}, fit_attr={}):
    """Utility function to fit a single recursive forecaster."""

    model = ForecastingCascade(est, **fit_params)
    for attr,val in fit_attr.items():
        setattr(model, attr, val)
    model = model._fit(X, y, sample_weight)

    return model


def _fit_direct(est, X, y, sample_weight=None,
                   fit_params={}, fit_attr={}):
    """Utility function to fit a single direct forecaster."""

    model = ForecastingChain(est, **fit_params)
    for attr,val in fit_attr.items():
        setattr(model, attr, val)
    model = model._fit(X, y, sample_weight)

    return model


class BaseForecaster(BaseEstimator, RegressorMixin):
    """Base class for Forecasting meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self):
        pass

    def _validate_lags(self):
        """Private method to validate lags."""

        def _check_lags(param_name, lags):

            msg = "Invalid {}. Lags must be an iterable of integers >0."
            if not np.iterable(lags) or isinstance(lags, str):
                raise ValueError(msg.format(param_name))

            lags = np.unique(lags).tolist()
            if len(lags) < 0: raise ValueError(msg)
            for l in lags:
                if not isinstance(l, numbers.Integral):
                    raise ValueError(msg.format(param_name))
                if l < 1:
                    raise ValueError(msg.format(param_name))

            return lags

        self.lags_ = _check_lags("lags", self.lags)
        self.min_ar_samples_ = max(self.lags_)
        if self.use_exog and self.exog_lags is not None:
            msg = ("Invalid exog_lags. exog_lags must be a dictionary "
                   "in the format {int: iterable} or a numeric iterable.")
            if isinstance(self.exog_lags, dict):
                self.exog_lags_ = {}
                for c, lags in self.exog_lags.items():
                    if not isinstance(c, numbers.Integral):
                        raise ValueError(msg)
                    self.exog_lags_[c] = _check_lags("exog_lags", lags)
                self.min_exog_samples_ = max(sum(self.exog_lags_.values(), []))
            elif np.iterable(self.exog_lags) and not isinstance(self.exog_lags, str):
                self.exog_lags_ = _check_lags("exog_lags", self.exog_lags)
                self.min_exog_samples_ = max(self.exog_lags_)
            else:
                raise ValueError(msg)
        else:
            self.exog_lags_ = None
            self.min_exog_samples_ = 0

        self.chunk_size_ = min(self.lags_)
        self.min_samples_ = max(self.min_ar_samples_, self.min_exog_samples_)

    def _validate_n_estimators(self):
        """Private method to validate estimator number."""

        msg = "Invalid n_estimators. n_estimators must be an integer >0 " \
              "or an iterable of integers. Got {} ({})."
        if isinstance(self.n_estimators, numbers.Integral):
            if self.n_estimators < 1:
                raise ValueError(msg.format(self.n_estimators), type(self.n_estimators))
            self.estimator_range_ = list(range(self.n_estimators))
        elif np.iterable(self.n_estimators) and not isinstance(self.n_estimators, str):
            n_estimators = np.unique(self.n_estimators).tolist()
            if len(n_estimators) < 0:
                raise ValueError(msg.format(self.n_estimators), type(self.n_estimators))
            self.estimator_range_ = []
            for e in n_estimators:
                if not isinstance(e, numbers.Integral):
                    raise ValueError(msg.format(self.n_estimators), type(self.n_estimators))
                if e < 1:
                    raise ValueError(msg.format(self.n_estimators), type(self.n_estimators))
                self.estimator_range_.append(e)
            self.estimator_range_ = [0] + self.estimator_range_[:]
        else:
            raise ValueError(msg.format(self.n_estimators), type(self.n_estimators))

    def _get_feature_names(self, X, y):

        feature_names = []
        if self.use_exog:
            n_features = X.shape[1]
            if hasattr(X, "columns"):
                exog_names = list(X.columns)
            else:
                exog_names = ['exog{}'.format(c) for c in range(n_features)]
            feature_names.extend(exog_names)

            if self.exog_lags is not None:
                if isinstance(self.exog_lags_, dict):
                    if min(self.exog_lags_.keys()) < 0 or \
                            max(self.exog_lags_.keys()) >= n_features:
                        raise ValueError(
                            "Invalid exog_lags. "
                            "Keys must be integers in [0, n_features_in_)."
                        )
                    feature_names.extend(
                        [['{}_lag{}'.format(exog_names[c], l)
                          for l in lags]
                         for c, lags in self.exog_lags_.items()][0]
                    )
                else:
                    feature_names.extend(
                        [['{}_lag{}'.format(exog_names[c], l)
                          for l in self.exog_lags_]
                         for c in range(n_features)][0]
                    )

        self.n_targets_ = y.shape[1] if len(y.shape) > 1 else 1
        if hasattr(y, "columns"):
            feature_names.extend(
                [['{}_lag{}'.format(c, l)
                  for l in self.lags_]
                 for c in y.columns][0]
            )
        elif hasattr(y, "name"):
            feature_names.extend(
                ['{}_lag{}'.format(y.name, l) for l in self.lags_]
            )
        else:
            feature_names.extend(
                [['target{}_lag{}'.format(c, l)
                  for l in self.lags_]
                 for c in range(self.n_targets_)][0]
            )

        self.feature_names_ = feature_names

    def _validate_data_fit(self, X, y, sample_weight=None):
        """Private method to validate data for fitting."""

        if self.use_exog:
            X, y = self._validate_data(
                X, y,
                reset=True,
                accept_sparse=False,
                dtype='float32',
                force_all_finite=not self.accept_nan,
                ensure_2d=True,
                allow_nd=False,
                ensure_min_samples=self.min_samples_ + 1 + int(self.target_diff),
                multi_output=True,
                y_numeric=True,
            )

        else:
            y = check_array(
                y,
                accept_sparse=False,
                dtype='float32',
                force_all_finite=not self.accept_nan,
                ensure_2d=False,
                allow_nd=False,
                ensure_min_samples=self.min_samples_ + 1 + int(self.target_diff),
            )

        if sample_weight is not None and \
                hasattr(self, 'estimator') and \
                has_fit_parameter(self.estimator, 'sample_weight'):
            sample_weight = _check_sample_weight(sample_weight, y)
        else:
            sample_weight = None

        return X, y, sample_weight

    def _validate_data_predict(self, X, last_y=None, last_X=None):
        """Private method to validate data for prediction."""

        X = self._validate_data(
            X,
            reset=False,
            accept_sparse=False,
            dtype='float32' if self.use_exog else None,
            force_all_finite=self.use_exog,
            ensure_2d=self.use_exog,
            allow_nd=False,
            ensure_min_features=self.n_features_in_ \
                if self.use_exog else 0
        )

        if self.use_exog and self.exog_lags is not None:
            if last_X is not None:
                last_X = check_array(
                    last_X,
                    accept_sparse=False,
                    dtype='float32',
                    force_all_finite=True,
                    ensure_2d=True,
                    allow_nd=False,
                    ensure_min_samples=self.min_samples_,
                    ensure_min_features=self.n_features_in_,
                )

        if last_y is not None:
            last_y = check_array(
                last_y,
                accept_sparse=False,
                dtype='float32',
                force_all_finite=True,
                ensure_2d=self.n_targets_ > 1,
                allow_nd=False,
                ensure_min_samples=self.min_samples_,
                ensure_min_features=self.n_targets_,
            )
            last_y = last_y.reshape(-1, self.n_targets_)

        return X, last_y, last_X

    def fit(self, X, y, sample_weight=None):
        """Fit a scikit-learn timeseries forecasting estimator.

        X is always required for compatibility reasons.
        With use_exog=False, no matter the values and the dimensionality X may assume.
        Also an array of nans or zeros suit fine. They aren't taken into account
        during fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features) if
            use_exog is set to True, or also None (only during fitting)
            Exogenous features in ascending temporal order.
            An array-like with the first dimension equal to at least min_samples_
            is required. Effective only with use_exog=True.

        y : array-like of shape (n_samples,) or also (n_samples, n_targets) if
            multiple outputs
            Series to forecast in ascending temporal order.
            An array-like with the first dimension equal to at least min_samples_
            is required.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Effective only if estimator accepts sample_weight.

        Returns
        -------
        self : object
        """

        msg = "{} must be bool. Got {} ({})."
        if not isinstance(self.use_exog, bool):
            raise ValueError(msg.format('use_exog', self.use_exog, type(self.use_exog)))
        if not isinstance(self.target_diff, bool):
            raise ValueError(msg.format('target_diff', self.target_diff, type(self.target_diff)))
        if not isinstance(self.accept_nan, bool):
            raise ValueError(msg.format('accept_nan', self.accept_nan, type(self.accept_nan)))

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

        if self.__class__.__name__ in ['ForecastingChain', 'ForecastingRectified']:
            self._validate_n_estimators()

        self._validate_lags()
        _X, _y, sample_weight = self._validate_data_fit(X, y, sample_weight)
        self._get_feature_names(X, y)
        del X, y

        _y = _y.reshape(-1, self.n_targets_)
        if self.target_diff:
            self._latest_y = _y[[-1]]
            _y = np.diff(_y, axis=0)

        self._fit(_X, _y, sample_weight)

        return self

    def predict(self, X, last_y=None, last_X=None, **predict_params):
        """Forecast future values.

        The number of time steps to forecast is automatically infered from
        the sample size of X.
        Also when use_exog=False, you must pass an array-like object with the first
        dimension equal to the number of steps to forecast.
        With use_exog=False, no matter the values X may assume. Also an array of nans
        or zeros suit fine. They aren't taken into account to make the forecasts.
        With use_exog=True, pass directly the expected values of the exogenous features.

        If last_y or last_X are not received (None), the latest values from the
        train data are used to initialize lag features creation.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features) if
            use_exog is set to True
            Exogenous features in ascending temporal order.

        last_y : array-like of shape (min_samples,) or also (min_samples, n_targets) if
            multiple outputs, default=None
            Useful past target values in ascending temporal order to initialize feature
            lags creation. An array-like with the first dimension equal to at least
            min_samples_ is required.

        last_X : array-like of shape (min_samples, n_features), default=None
            Useful past exogenous values in ascending temporal order to initialize feature
            lags creation. An array-like with the first dimension equal to at least
            min_samples_ is required. Effective only with use_exog=True.

        **predict_params : Additional prediction arguments used by estimator_

        Returns
        -------
        preds : ndarray of shape (n_samples,) or also (n_samples, n_targets) if
            multiple outputs
            Forecast values.
        """

        check_is_fitted(self, 'n_targets_')
        X, last_y, last_X = self._validate_data_predict(X, last_y, last_X)

        preds = self._predict(X, last_y=last_y, last_X=last_X, **predict_params)

        if self.target_diff:
            latest_y = self._latest_y if last_y is None else last_y[[-1]]
            preds = _vstack([latest_y, preds.reshape(-1, self.n_targets_)])
            preds = np.cumsum(preds, axis=0)[1:]
            preds = preds if self.n_targets_ > 1 else preds.ravel()

        return preds

    def transform(self, X, last_y=None, last_X=None):
        """Forecast future values.

        Get forecasts of every single estimator and stack them vertically.
        Only available for ForecastingStacked and ForecastingRectified.

        The number of time steps to forecast is automatically infered from
        the sample size of X.
        Also when use_exog=False, you must pass an array-like object with the first
        dimension equal to the number of steps to forecast.
        With use_exog=False, no matter the values X may assume. Also an array of nans
        or zeros suit fine. They aren't taken into account to make the forecasts.
        With use_exog=True, pass directly the expected values of the exogenous features.

        If last_y or last_X are not received (None), the latest values from the
        training are always used to initialize lag features creation.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features) if
            use_exog is set to True
            Exogenous features in ascending temporal order.

        last_y : array-like of shape (min_samples,) or also (min_samples, n_targets) if
            multiple outputs, default=None
            Useful past target values in ascending temporal order to initialize feature
            lags creation. An array-like with the first dimension equal to at least
            min_samples_ is required.

        last_X : array-like of shape (min_samples, n_features), default=None
            Useful past exogenous values in ascending temporal order to initialize feature
            lags creation. An array-like with the first dimension equal to at least
            min_samples_ is required. Effective only with use_exog=True.

        Returns
        -------
        preds : ndarray of shape (n_estiamtors, n_samples, n_targets)
            Forecast values.
        """

        if self.__class__.__name__ not in ['ForecastingStacked', 'ForecastingRectified']:
            raise AttributeError(
                "transform method not available for {}.".format(self.__class__.__name__)
            )

        check_is_fitted(self, 'n_targets_')
        X, last_y, last_X = self._validate_data_predict(X, last_y, last_X)

        preds = \
            _vstack([[est._predict(X=X, last_y=last_y, last_X=last_X) \
                    .reshape(-1, self.n_targets_)] for est in self.estimators_])
        preds = preds.transpose((2, 1, 0))

        return preds

    def score(self, X, y, sample_weight=None,
              scoring='mse', uniform_average=True,
              **predict_params):
        """Return the selected score on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features) if
            use_exog is set to True
            Exogenous features in ascending temporal order.
            An array-like with the first dimension equal to at least min_samples_
            is required. Effective only with use_exog=True.

        y : array-like of shape (n_samples,) or also (n_samples, n_targets) if
            multiple outputs
            Series to forecast in ascending temporal order.
            An array-like with the first dimension equal to at least min_samples_
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

    Scikit-learn estimator for recursive time series forecasting.
    It automatically builds lag features on target and optionally also on
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
    >>> model = ForecastingCascade(
    ...     Ridge(),
    ...     lags=range(1,24+1),
    ...     use_exog=False,
    ...     accept_nan=False
    ... )
    >>> model.fit(None, y)
    >>> forecasts = model.predict(np.arange(24*3))
    """

    def __init__(
            self,
            estimator,
            *,
            lags,
            exog_lags=None,
            use_exog=False,
            target_diff=False,
            accept_nan=False,
    ):
        self.estimator = estimator
        self.lags = lags
        self.exog_lags = exog_lags
        self.use_exog = use_exog
        self.target_diff = target_diff
        self.accept_nan = accept_nan

    def _create_lags(self, X, lags, end, start=0):
        """Private method to create lag features."""

        Xt = _hstack([X[(end - l):-(l - start) or None] for l in lags])

        return Xt

    def _fit(self, X, y, sample_weight=None):
        """Private method for recursive forecasting train."""

        if self.use_exog:
            self._last_X = X[-self.min_samples_:]

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

        Xar = self._create_lags(y, self.lags_, self.min_samples_)
        X = _hstack([X[int(self.target_diff):], Xar]) if self.use_exog else Xar
        self._last_y = y[-self.min_samples_:]
        y = y[self.min_samples_:]
        if sample_weight is not None:
            sample_weight = sample_weight[self.min_samples_ + int(self.target_diff):]

        if self.accept_nan:
            mask = ~(np.isnan(y).any(1))
            X, y = X[mask], y[mask]
            if sample_weight is not None: sample_weight = sample_weight[mask]
            if y.shape[0] < 1: raise Exception("Target contains only NaNs.")

        y = y if self.n_targets_ > 1 else y.ravel()
        self.estimator_ = clone(self.estimator)
        if sample_weight is None:
            self.estimator_.fit(X, y)
        else:
            self.estimator_.fit(X, y, sample_weight=sample_weight)

        return self

    def _predict_chunk(self, X, last_y, **predict_params):
        """Private method to predict chunks of data simultaneously."""

        y = last_y[-self.min_ar_samples_:]

        Xar = self._create_lags(
            y, self.lags_,
            end=self.min_ar_samples_, start=X.shape[0]
        )
        X = _hstack([X, Xar]) if self.use_exog else Xar

        pred = self.estimator_.predict(X, **predict_params)
        last_y = _vstack([y, pred.reshape(-1, self.n_targets_)])

        return pred, last_y

    def _predict(self, X, last_y=None, last_X=None, **predict_params):
        """Private method for recursive forecasting."""

        if self.use_exog and self.exog_lags_ is not None:
            if last_X is None: last_X = self._last_X
            X = _vstack([last_X[-self.min_exog_samples_:], X])
            if isinstance(self.exog_lags_, dict):
                X = _hstack(
                    [X[self.min_exog_samples_:]] + \
                    [self._create_lags(X[:, [c]], lags, self.min_exog_samples_)
                     for c, lags in self.exog_lags_.items()]
                )
            elif np.iterable(self.exog_lags_):
                X = _hstack(
                    [X[self.min_exog_samples_:]] + \
                    [self._create_lags(X, self.exog_lags_, self.min_exog_samples_)]
                )

        n_preds = X.shape[0]
        idchunks = list(range(self.chunk_size_, n_preds, self.chunk_size_))
        n_chunks = len(idchunks) + 1
        idchunks = [0] + idchunks + [n_preds]
        if last_y is None: last_y = self._last_y

        preds = []
        for i in range(n_chunks):
            pred, last_y = self._predict_chunk(
                X[idchunks[i]:idchunks[i + 1]],
                last_y,
                **predict_params
            )
            preds.append(pred)
        preds = _vstack(preds)

        return preds


class ForecastingChain(BaseForecaster):
    """Direct Time Series Forecaster

    Scikit-learn estimator for direct time series forecasting.
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
    >>> model = ForecastingChain(
    ...     Ridge(),
    ...     n_estimators=24,
    ...     lags=range(1,24+1),
    ...     use_exog=False,
    ...     accept_nan=False
    ... )
    >>> model.fit(None, y)
    >>> forecasts = model.predict(np.arange(24*3))
    """

    def __init__(
            self,
            estimator,
            *,
            n_estimators,
            lags,
            exog_lags=None,
            use_exog=False,
            target_diff=False,
            accept_nan=False,
            n_jobs=None,
            verbose=0
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.lags = lags
        self.exog_lags = exog_lags
        self.use_exog = use_exog
        self.target_diff = target_diff
        self.accept_nan = accept_nan
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit(self, X, y, sample_weight=None, check=False):
        """Private method for direct forecasting train."""

        self.min_samples_ += self.chunk_size_ * max(self.estimator_range_)

        self.estimators_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )(
            delayed(_fit_recursive)(
                est=self.estimator,
                X=X, y=y, sample_weight=sample_weight,
                fit_params={
                    'lags': list(map(
                        lambda x: x + self.chunk_size_ * e, self.lags_
                    )),
                    'exog_lags': self.exog_lags_,
                    'use_exog': self.use_exog,
                    'target_diff': self.target_diff,
                    'accept_nan': self.accept_nan
                },
                fit_attr={
                    'n_targets_': self.n_targets_,
                    'feature_names_': self.feature_names_,
                    'lags_': list(map(
                        lambda x: x + self.chunk_size_ * e, self.lags_
                    )),
                    'min_ar_samples_': self.lags_[-1] + self.chunk_size_ * e,
                    'chunk_size_': self.lags_[0] + self.chunk_size_ * e,
                    'min_samples_': max(
                        self.lags_[-1] + self.chunk_size_ * e,
                        self.min_exog_samples_
                    ),
                    'exog_lags_': self.exog_lags_,
                    'min_exog_samples_': self.min_exog_samples_,
                }
            )
            for e in self.estimator_range_
        )

        return self

    def _predict(self, X, last_y=None, last_X=None, **predict_params):
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
            pred[istart:iend] = self.estimators_[i]._predict(
                X=X[:iend],
                last_y=last_y,
                last_X=last_X,
                **predict_params
            )[istart:]
            pred_matrix.append([pred])
            if istart >= n_preds: break
            istart += e

        pred_matrix = np.nanmedian(_vstack(pred_matrix), axis=0)

        return pred_matrix


class ForecastingStacked(BaseForecaster):
    """Stacked Recursive Time Series Forecaster

    Scikit-learn estimator for stacked time series forecasting.
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
        If None, any final_estimator is fitted and estimator predictions are
        combined with the median.

    lags : iterable
        1D iterable of integers representing the lag features to create from
        the target.

    exog_lags : iterable or dict, default=None
        1D iterable of integers representing the lag features to create from
        the exogenous features. In that case, lagged features are created for
        all the received features simultaneously.
        if dict: keys represent the index of the desired features from which
        create lags, while values must be 1D iterable of integers representing the
        lag features to create.

    target_diff : bool, deafult=False
        Apply the first-order differentiation on the target.

    final_estimator : object, default=None
        A regressor which will be used to combine the recursive forecasts.
        When final_estimator is None predictions are combined using a
        LinearCombination estimator.
        Effetctive only if test_size is not None.

    use_exog : bool, default=False
        Use or ignore the received features during fitting as X.
        If False, exog_lags are ignored.

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
    >>> model = ForecastingStacked(
    ...     [Ridge(), DecisionTreeRegressor()],
    ...     test_size=24*3,
    ...     lags=range(1,24+1),
    ...     use_exog=False
    ... )
    >>> model.fit(None, y)
    >>> forecasts = model.predict(np.arange(24*3))
    """

    def __init__(
            self,
            estimators,
            *,
            test_size,
            lags,
            exog_lags=None,
            use_exog=False,
            target_diff=False,
            final_estimator=None,
            n_jobs=None,
            verbose=0
    ):
        self.estimators = estimators
        self.test_size = test_size
        self.lags = lags
        self.exog_lags = exog_lags
        self.use_exog = use_exog
        self.target_diff = target_diff
        self.accept_nan = False
        self.final_estimator = final_estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit(self, X, y, sample_weight=None, check=False):
        """Private method for stacked forecasting train."""

        fit_params = {
            'lags': self.lags_,
            'exog_lags': self.exog_lags_,
            'use_exog': self.use_exog,
            'target_diff': self.target_diff,
            'accept_nan': False
        }
        fit_attr = {
            'n_targets_': self.n_targets_,
            'feature_names_': self.feature_names_,
            'lags_': self.lags_,
            'min_ar_samples_': self.lags_[-1],
            'chunk_size_': self.lags_[0],
            'min_samples_': max(self.lags_[-1], self.min_exog_samples_),
            'exog_lags_': self.exog_lags_,
            'min_exog_samples_': self.min_exog_samples_,
        }
        if self.test_size is None:
            self.final_estimator_ = None
        else:
            train_id, test_id = train_test_split(
                np.arange(y.shape[0] + int(self.target_diff)),
                test_size=self.test_size, shuffle=False
            )
            if train_id.shape[0] < self.min_samples_:
                raise ValueError(
                    "Found array with {} sample(s) while a"
                    " minimum of {} is required." \
                        .format(train_id.shape[0],
                                self.min_samples_ + 1 + int(self.target_diff))
                )

            if sample_weight is not None:
                sample_weight = _check_sample_weight(sample_weight, y)

            preds = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )(
                delayed(_fit_recursive(
                    est=est,
                    X=X[train_id] if self.use_exog else train_id,
                    y=y[train_id][:-int(self.target_diff) or None],
                    sample_weight=(sample_weight if sample_weight is None
                        else sample_weight[train_id][:-int(self.target_diff) or None]),
                    fit_params=fit_params,
                    fit_attr=fit_attr
                )._predict)(
                    X[test_id] if self.use_exog else test_id
                )
                for est in self.estimators
            )

            preds = _vstack([[p.reshape(-1, self.n_targets_)] for p in preds])
            preds = preds.transpose((2, 1, 0))

            final_estimator = clone(self.final_estimator) \
                if self.final_estimator is not None else LinearCombination()
            self.final_estimator_ = \
                [clone(final_estimator).fit(
                    preds[t],
                    y[test_id - int(self.target_diff), t]
                )
                for t in range(self.n_targets_)]

        self.estimators_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )(
            delayed(_fit_recursive)(
                est=est,
                X=X, y=y,
                sample_weight=sample_weight,
                fit_params=fit_params,
                fit_attr=fit_attr
            )
            for est in self.estimators
        )

        return self

    def _predict(self, X, last_y=None, last_X=None, **predict_params):
        """Private method for stacked forecasting."""

        X = self.transform(X, last_y=last_y, last_X=last_X)

        if self.final_estimator_ is not None:
            preds = [est.predict(X[t], **predict_params)
                     for t, est in enumerate(self.final_estimator_)]
            preds = _vstack(preds).reshape(len(preds), -1).T

        else:
            preds = np.median(X, axis=-1).T

        return preds


class ForecastingRectified(BaseForecaster):
    """Stacked Direct Time Series Forecaster

    Scikit-learn estimator for stacked time series forecasting.
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
        If None, any final_estimator is fitted and estimator predictions are
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
    >>> model = ForecastingRectified(
    ...     [Ridge(), DecisionTreeRegressor()],
    ...     test_size=24*3,
    ...     n_estimators=24*3,
    ...     lags=range(1,24+1),
    ...     use_exog=False
    ... )
    >>> model.fit(None, y)
    >>> forecasts = model.predict(np.arange(24*3))
    """

    def __init__(
            self,
            estimators,
            *,
            test_size,
            n_estimators,
            lags,
            exog_lags=None,
            use_exog=False,
            target_diff=False,
            final_estimator=None,
            n_jobs=None,
            verbose=0
    ):
        self.estimators = estimators
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.lags = lags
        self.exog_lags = exog_lags
        self.use_exog = use_exog
        self.target_diff = target_diff
        self.accept_nan = False
        self.final_estimator = final_estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit(self, X, y, sample_weight=None, check=False):
        """Private method for stacked forecasting train."""

        self.min_samples_ += self.chunk_size_ * max(self.estimator_range_)

        fit_params = {
            'n_estimators': self.estimator_range_,
            'lags': self.lags_,
            'exog_lags': self.exog_lags_,
            'use_exog': self.use_exog,
            'target_diff': self.target_diff,
            'accept_nan': False,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
        }
        fit_attr = {
            'estimator_range_': self.estimator_range_,
            'n_targets_': self.n_targets_,
            'feature_names_': self.feature_names_,
            'lags_': self.lags_,
            'min_ar_samples_': self.lags_[-1],
            'chunk_size_': self.lags_[0],
            'min_samples_': max(self.lags_[-1], self.min_exog_samples_),
            'exog_lags_': self.exog_lags_,
            'min_exog_samples_': self.min_exog_samples_,
        }
        if self.test_size is None:
            self.final_estimator_ = None
        else:
            train_id, test_id = train_test_split(
                np.arange(y.shape[0] + int(self.target_diff)),
                test_size=self.test_size, shuffle=False
            )
            if train_id.shape[0] < self.min_samples_:
                raise ValueError(
                    "Found array with {} sample(s) while a"
                    " minimum of {} is required." \
                        .format(train_id.shape[0],
                                self.min_samples_ + 1 + int(self.target_diff))
                )

            if sample_weight is not None:
                sample_weight = _check_sample_weight(sample_weight, y)

            preds = [
                _fit_direct(
                    est=est,
                    X=X[train_id] if self.use_exog else train_id,
                    y=y[train_id][:-int(self.target_diff) or None],
                    sample_weight=(sample_weight if sample_weight is None
                        else sample_weight[train_id][:-int(self.target_diff) or None]),
                    fit_params=fit_params,
                    fit_attr=fit_attr
                )._predict(
                    X[test_id] if self.use_exog else test_id
                )
                for est in self.estimators
            ]

            preds = _vstack([[p.reshape(-1, self.n_targets_)] for p in preds])
            preds = preds.transpose((2, 1, 0))

            final_estimator = clone(self.final_estimator) \
                if self.final_estimator is not None else LinearCombination()
            self.final_estimator_ = \
                [clone(final_estimator).fit(
                    preds[t],
                    y[test_id - int(self.target_diff), t]
                )
                for t in range(self.n_targets_)]

        self.estimators_ = [
            _fit_direct(
                est=est,
                X=X, y=y,
                sample_weight=sample_weight,
                fit_params=fit_params,
                fit_attr=fit_attr
            )
            for est in self.estimators
        ]

        return self

    def _predict(self, X, last_y=None, last_X=None, **predict_params):
        """Private method for stacked forecasting."""

        X = self.transform(X, last_y=last_y, last_X=last_X)

        if self.final_estimator_ is not None:
            preds = [est.predict(X[t], **predict_params)
                     for t, est in enumerate(self.final_estimator_)]
            preds = _vstack(preds).reshape(len(preds), -1).T

        else:
            preds = np.median(X, axis=-1).T

        return preds