import numbers
import numpy as np

from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.utils.validation import \
    _check_sample_weight, has_fit_parameter, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin

from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from tsprial.model_selection import TemporalSplit


_vstack = lambda x: np.concatenate(x, axis=0)
_hstack = lambda x: np.concatenate(x, axis=1)


def _set_sample_weight(est, sample_weight, index=None):
    """Utility function to validate sample_weigth."""

    if sample_weight is not None and has_fit_parameter(est, 'sample_weight'):
        sample_weight = sample_weight[index] if index is not None \
            else sample_weight

    return sample_weight

def _fit_single(est, X, y, sample_weight=None, index=None, **fit_params):
    """Utility function to fit a single recursive forecaster."""

    model = ForecastingCascade(clone(est), **fit_params)
    sample_weight = _set_sample_weight(est, sample_weight, index)
    model.fit(X, y, sample_weight, check_input=False)

    return model


class BaseForecaster(BaseEstimator, RegressorMixin):
    """Base class for Forecasting meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self):
        pass

    def _validate_lags(self, check=True):
        """Private method to validate lags."""

        def _check_lags(param_name, lags):

            msg = "Invalid {}. Lags must be an iterable of integers >0."

            if not np.iterable(lags) or isinstance(lags, str):
                raise ValueError(msg.format(param_name))

            lags = sorted(list(set(lags)))
            if len(lags) < 0:
                raise ValueError(msg)
            for l in lags:
                if not isinstance(l, numbers.Integral):
                    raise ValueError(msg.format(param_name))
                if l < 1:
                    raise ValueError(msg.format(param_name))

            return lags

        if check:
            self.lags_ = _check_lags("lags", self.lags)
            self.min_ar_samples_ = max(self.lags_)

            self.exog_lags_ = None
            self.min_exog_samples_ = 0
            if self.use_exog and self.exog_lags is not None:
                msg = ("Invalid exog_lags. exog_lags must be a dictionary "
                       "in the format {int: iterable} or iterable.")

                if isinstance(self.exog_lags, dict):
                    self.exog_lags_ = {}
                    for c, lags in self.exog_lags.items():
                        if not isinstance(c, numbers.Integral):
                            raise ValueError(msg)
                        self.exog_lags_[c] = _check_lags("exog_lags", lags)
                        self.min_exog_samples_ = max(
                            self.min_exog_samples_, max(self.exog_lags_[c])
                        )
                elif np.iterable(self.exog_lags) and not isinstance(self.exog_lags, str):
                    self.exog_lags_ = _check_lags("exog_lags", self.exog_lags)
                    self.min_exog_samples_ = max(self.exog_lags_)
                else:
                    raise ValueError(msg)

        else:
            self.lags_ = self.lags
            self.min_ar_samples_ = max(self.lags)
            self.exog_lags_ = self.exog_lags
            if self.use_exog and self.exog_lags_ is not None:
                if isinstance(self.exog_lags_, dict):
                    self.min_exog_samples_ = max(sum(self.exog_lags_.values(), []))
                else:
                    self.min_exog_samples_ = max(self.exog_lags_)
            else:
                self.min_exog_samples_ = 0

        self.chunk_size_ = min(self.lags_)
        self.min_samples_ = max(self.min_ar_samples_, self.min_exog_samples_)

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
                ensure_min_samples=self.min_samples_ + 1,
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
                ensure_min_samples=self.min_samples_ + 1,
            )

        if sample_weight is not None and \
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

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from tsprial.forecasting import ForecastingCascade
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
            accept_nan=False
    ):
        self.estimator = estimator
        self.lags = lags
        self.exog_lags = exog_lags
        self.use_exog = use_exog
        self.accept_nan = accept_nan

    def _create_lags(self, X, lags, end, start=0):
        """Private method to create lag features."""

        Xt = _hstack([X[(end - l):-(l - start) or None] for l in lags])

        return Xt

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a recursive scikit-learn time series forecasting estimator.

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

        check_input: bool, default=True
            Allow to bypass several input checking.
            Don’t use this parameter unless you know what you do.

        Returns
        -------
        self : object
        """

        self._validate_lags(check=check_input)
        if check_input:
            X, y, sample_weight = self._validate_data_fit(X, y, sample_weight)
        self.n_targets_ = y.shape[1] if len(y.shape) > 1 else 1

        if self.use_exog:
            self._last_X = X[-self.min_samples_:]

            if isinstance(self.exog_lags_, dict):
                n_features = X.shape[1]
                if min(self.exog_lags_.keys()) < 0 or \
                    max(self.exog_lags_.keys()) >= n_features:
                    raise ValueError(
                        "Invalid exog_lags. "
                        "Keys must be integers in [0,n_features_in_)."
                    )
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

        y = y.reshape(-1, self.n_targets_)
        self._last_y = y[-self.min_samples_:]

        Xar = self._create_lags(y, self.lags_, self.min_samples_)
        X = _hstack([X, Xar]) if self.use_exog else Xar
        y = y[self.min_samples_:]

        if self.accept_nan:
            mask = ~(np.isnan(y).any(1))
            X, y = X[mask], y[mask]
            if sample_weight is not None:
                sample_weight = sample_weight[mask]

            if len(y) < 1:
                raise ValueError("Target contains only NaNs after lag creation.")

        y = y if self.n_targets_ > 1 else y.ravel()
        self.estimator_ = clone(self.estimator)
        if sample_weight is None:
            self.estimator_.fit(X, y)
        else:
            self.estimator_.fit(X, y, sample_weight=sample_weight)

        return self

    def _predict_chunk(self, X, last_y, **predict_params):
        """Private method to simultaneously predict chunks of data."""

        y = last_y[-self.min_ar_samples_:]

        Xar = self._create_lags(
            y, self.lags_,
            end=self.min_ar_samples_, start=len(X)
        )
        X = _hstack([X, Xar]) if self.use_exog else Xar

        pred = self.estimator_.predict(X, **predict_params)
        last_y = _vstack([y, pred.reshape(-1, self.n_targets_)])

        return pred, last_y

    def predict(self, X, last_y=None, last_X=None, check_input=True, **predict_params):
        """Forecast future values.

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

        check_input: bool, default=True
            Allow to bypass several input checking.
            Don’t use this parameter unless you know what you do.

        **predict_params : Additional prediction arguments used by estimator_

        Returns
        -------
        preds : ndarray of shape (n_samples,) or also (n_samples, n_targets) if
            multiple outputs
            Forecast values.
        """

        check_is_fitted(self, 'n_targets_')
        if check_input:
            X, last_y, last_X = self._validate_data_predict(X, last_y, last_X)

        if self.use_exog and self.exog_lags_ is not None:
            if last_X is None:
                last_X = self._last_X.copy()
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

        if last_y is None:
            last_y = self._last_y.copy()

        if self.chunk_size_ > 1:
            chunks = np.arange(self.chunk_size_, len(X), self.chunk_size_)
            X_chunks = np.split(X, chunks)
        else:
            X_chunks = np.split(X, len(X))

        preds = []
        for c_x in X_chunks:
            pred, last_y = self._predict_chunk(c_x, last_y, **predict_params)
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
    The optimal forecast time step is evaluated considering the received
    value for n_estimators and the chunk_size.
    Multivariate timeseries forecasting is natively supported. If the received
    scikit-learn model doesn't support multivariate targets, use it in
    MultiOutputRegressor.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator compatible with scikit-learn.

    n_estimators : int
        How many estimators fit to forecast future time steps.
        Each estimator is fitted independently from each other.

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

    accept_nan : bool, default=False
        Use or ignore NaNs target observations during fitting.
        At least one not-nan observation is required, after lags creation,
        to make a successful fit.

    n_jobs : int, default=None
        Effective only with grid and random search.
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

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from tsprial.forecasting import ForecastingChain
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
            accept_nan=False,
            n_jobs=None,
            verbose=0
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.lags = lags
        self.exog_lags = exog_lags
        self.use_exog = use_exog
        self.accept_nan = accept_nan
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit a direct scikit-learn time series forecasting estimator.

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

        self._validate_lags()
        self.min_samples_ += self.chunk_size_ * (self.n_estimators - 1)
        X, y, sample_weight = self._validate_data_fit(X, y, sample_weight)
        self.n_targets_ = y.shape[1] if len(y.shape) > 1 else 1

        msg = "n_estimators must be an integer >0."
        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError(msg)
        if self.n_estimators < 1:
            raise ValueError(msg)

        lags = [
            list(map(lambda x: x + self.chunk_size_ * i, self.lags_))
            for i in range(self.n_estimators)
        ]

        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(ForecastingCascade(
                clone(self.estimator),
                lags=ls,
                exog_lags=self.exog_lags_,
                use_exog=self.use_exog,
                accept_nan=self.accept_nan
            ).fit)(X, y, sample_weight, False)
            for ls in lags
        )
        # self.estimators_ = sorted(self.estimators_, key=lambda m: m.chunk_size_)

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

        **predict_params : Additional prediction arguments used by each estimator
            in estimators_

        Returns
        -------
        preds : ndarray of shape (n_samples,) or also (n_samples, n_targets) if
            multiple outputs
            Forecast values.
        """

        check_is_fitted(self, 'n_targets_')
        X, last_y, last_X = self._validate_data_predict(X, last_y, last_X)

        if self.n_targets_ > 1:
            preds = np.zeros((len(X), self.n_targets_))
        else:
            preds = np.zeros((len(X),))

        i = 0
        for i, est in enumerate(self.estimators_[:-1]):
            i = (i + 1) * self.chunk_size_
            if (i - self.chunk_size_) > len(preds):
                break
            preds[i - self.chunk_size_:i] = est.predict(
                X[:i], last_y=last_y, last_X=last_X,
                check_input=False, **predict_params
            )[i - self.chunk_size_:]

        if i <= len(preds):
            preds[i:] = self.estimators_[-1].predict(
                X, last_y=last_y, last_X=last_X,
                check_input=False, **predict_params
            )[i:]

        return preds


class ForecastingStacked(BaseForecaster):
    """Stacked Time Series Forecaster

    Scikit-learn estimator for stacked time series forecasting.
    It fits multiple recursive time series forecasters (using ForecastingCascade)
    and combines them on a final portion of received data with a meta-learner.
    Multivariate timeseries forecasting is natively supported. If the received
    scikit-learn models doesn't support multivariate targets, use it in
    MultiOutputRegressor.

    Parameters
    ----------
    estimators : object or iterable of objects
        A single or list of supervised learning estimators compatible with
        scikit-learn.

    final_estimator : object, default=None
        A regressor which will be used to combine the recursive forecasts.
        The default final_estimator is a Ridge() from scikit-learn.

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

    test_size : float or int, default=None
        Final portion of data used to make forecasts and fit the final_estimator.
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to 0.25.

    n_jobs : int, default=None
        Effective only with grid and random search.
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

    final_estimator_ : estimator
        The final estimator used to stack the estimators' forecasts.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from tsprial.forecasting import ForecastingStacked
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
            final_estimator=None,
            *,
            lags,
            exog_lags=None,
            use_exog=False,
            test_size=None,
            n_jobs=None,
            verbose=0
    ):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.test_size = test_size
        self.lags = lags
        self.exog_lags = exog_lags
        self.use_exog = use_exog
        self.accept_nan = False
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit a stacked scikit-learn time series forecasting estimator.

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
            Effective only if estimators accept sample_weight.

        Returns
        -------
        self : object
        """

        if not isinstance(self.estimators, (list, tuple, set)):
            estimators = [self.estimators]
        else:
            estimators = self.estimators

        for est in estimators:
            if not hasattr(est, 'fit') or not hasattr(est, 'predict'):
                raise ValueError(
                    "estimators must implement fit and predict methods. "
                    "{} doesn't.".format(est)
                )

        if self.test_size is not None and \
                not isinstance(self.test_size, (numbers.Integral, numbers.Real)):
            raise ValueError(
                "Expected test_size as integer or float. Got {}.".format(self.test_size)
            )

        self._validate_lags()
        X, y, sample_weight = self._validate_data_fit(X, y, sample_weight)
        self.n_targets_ = y.shape[1] if len(y.shape) > 1 else 1

        train_id, test_id = train_test_split(
            np.arange(len(y)), test_size=self.test_size, shuffle=False
        )

        fit_params = {
            'lags': self.lags_,
            'exog_lags': self.exog_lags_,
            'use_exog': self.use_exog,
            'accept_nan' : False
        }
        preds = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_single(
                est,
                X[train_id] if self.use_exog else train_id,
                y[train_id],
                sample_weight,
                train_id,
                **fit_params
            ).predict)(X[test_id] if self.use_exog else test_id, check_input=False)
            for est in estimators
        )

        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_single)(
                est, X, y, sample_weight, None, **fit_params
            )
            for est in estimators
        )

        preds = _hstack([p.reshape(-1, self.n_targets_) for p in preds])
        X = _hstack([preds, X[test_id]]) if self.use_exog else preds

        self.final_estimator_ = clone(self.final_estimator) \
            if self.final_estimator is not None else Ridge()
        self.final_estimator_.fit(X, y[test_id])

        return self

    def transform(self, X, last_y=None, last_X=None):
        """Forecast future values.

        Get forecasts of every single estimator and stack them vertically.

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
        preds : ndarray of shape (n_samples, estimators)
            Forecast values.
        """

        check_is_fitted(self, 'n_targets_')
        X, last_y, last_X = self._validate_data_predict(X, last_y, last_X)

        preds = []
        for est in self.estimators_:
            pred = est.predict(X, last_y=last_y, last_X=last_X, check_input=False)
            preds.append(pred.reshape(-1, self.n_targets_))

        preds = _hstack(preds)
        if self.use_exog:
            X = _hstack([preds, X])
        else:
            X = preds

        return X

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

        **predict_params : Additional prediction arguments used by final_estimator_

        Returns
        -------
        preds : ndarray of shape (n_samples,) or also (n_samples, n_targets) if
            multiple outputs
            Forecast values.
        """

        X = self.transform(X, last_y=last_y, last_X=last_X)
        pred = self.final_estimator_.predict(X, **predict_params)

        return pred


class ForecastingRectified(BaseForecaster):
    """Rectified Time Series Forecaster

    Scikit-learn estimator for rectified time series forecasting.
    It fits multiple recursive time series forecasters (using ForecastingCascade)
    on different sliding window bunches.
    Forecasts are adjusted and combined fitting a meta-learner for each
    forecasting step.
    Multivariate timeseries forecasting is natively supported. If the received
    scikit-learn models doesn't support multivariate targets, use it in
    MultiOutputRegressor.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator compatible with scikit-learn used
        to produce recursive forecasts.

    final_estimator : object, default=None
        A regressor which will be used to combine the recursive forecasts.
        The default final_estimator is a Ridge() from scikit-learn.

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

    test_size : float or int, default=None
        The size of the sliding window bunches used to make recursive forecasts and
        fit a final_estimator for each forecasting step.
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to 0.25.

    n_estimators : int
        How many sliding window bunches must be generated.
        In other words, how many ForecastingCascade must be fitted.

    n_jobs : int, default=None
        Effective only with grid and random search.
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

    final_estimators_ : list
        A list of fitted scikit-learn estimator instances on recursive forecasts,
        one for each forecast step.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from tsprial.forecasting import ForecastingRectified
    >>> timesteps = 400
    >>> e = np.random.normal(0,1, (timesteps,))
    >>> y = 2*np.sin(np.arange(timesteps)*(2*np.pi/24))+e
    >>> model = ForecastingRectified(
    ...     Ridge(),
    ...     n_estimators=200,
    ...     test_size=24*3,
    ...     lags=range(1,24+1),
    ...     use_exog=False
    ... )
    >>> model.fit(None, y)
    >>> forecasts = model.predict(np.arange(24*3))
    """

    def __init__(
            self,
            estimator,
            final_estimator=None,
            *,
            n_estimators,
            lags,
            exog_lags=None,
            use_exog=False,
            test_size=None,
            n_jobs=None,
            verbose=0
    ):
        self.estimator = estimator
        self.final_estimator = final_estimator
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.lags = lags
        self.exog_lags = exog_lags
        self.use_exog = use_exog
        self.accept_nan = False
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit a rectified scikit-learn time series forecasting estimator.

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

        msg = "n_estimators must be an integer >4."
        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError(msg)
        if self.n_estimators < 5:
            raise ValueError(msg)

        if self.test_size is not None and \
                not isinstance(self.test_size, (numbers.Integral, numbers.Real)):
            raise ValueError(
                "Expected test_size as integer or float. Got {}.".format(self.test_size)
            )

        self._validate_lags()
        X, y, sample_weight = self._validate_data_fit(X, y, sample_weight)
        self.n_targets_ = y.shape[1] if len(y.shape) > 1 else 1

        cv = TemporalSplit(self.n_estimators, test_size=self.test_size, gap=self.chunk_size_)

        fit_params = {
            'lags': self.lags_,
            'exog_lags': self.exog_lags_,
            'use_exog': self.use_exog,
            'accept_nan': False
        }
        self.estimator_ = _fit_single(
            self.estimator, X, y, sample_weight, None, **fit_params
        )

        preds = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_single(
                self.estimator,
                X[train_id] if self.use_exog else train_id,
                y[train_id],
                sample_weight,
                train_id,
                **fit_params
            ).predict)(X[test_id] if self.use_exog else test_id, check_input=False)
            for train_id, test_id in cv.split(y)
        )

        preds = _hstack([p.reshape(-1, self.n_targets_) for p in preds]).T
        trues = _hstack([y[test_id].reshape(-1, self.n_targets_)
                         for _, test_id in cv.split(y)]).T
        if self.use_exog:
            X = _vstack([X[test_id][None, ...] for _, test_id in cv.split(y)])

        self.final_estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(
                lambda x:
                clone(self.final_estimator).fit(*x)
                if self.final_estimator is not None else
                Ridge().fit(*x)
            )(
                (_hstack([X[:, h], preds[:, h].reshape(-1, self.n_targets_)])
                 if self.use_exog else
                 preds[:, h].reshape(-1, self.n_targets_),
                 trues[:, h].reshape(-1, self.n_targets_))
                if self.n_targets_ > 1 else
                (_hstack([X[:, h], preds[:, [h]]])
                 if self.use_exog else
                 preds[:, [h]], trues[:, h])
            )
            for h in range(preds.shape[1])
        )

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

        **predict_params : Additional prediction arguments used by final_estimators_

        Returns
        -------
        preds : ndarray of shape (n_samples,) or also (n_samples, n_targets) if
            multiple outputs
            Forecast values.
        """

        check_is_fitted(self, 'n_targets_')
        X, last_y, last_X = self._validate_data_predict(X, last_y, last_X)

        pred = self.estimator_.predict(
            X, last_y=last_y, last_X=last_X, check_input=False, **predict_params
        )

        if self.use_exog:
            X = _hstack([X, pred.reshape(-1, self.n_targets_)])
        else:
            X = pred.reshape(-1, self.n_targets_)

        for i in range(len(X)):
            if i >= len(self.final_estimators_):
                pred[i:] = self.final_estimators_[-1].predict(X[i:])
                break
            pred[i] = self.final_estimators_[i].predict(X[[i]])

        return pred