# tspiral
A python package for time series forecasting with scikit-learn estimators.

tspiral is not a library that works as a wrapper for other tools and methods for time series forecasting. tspiral directly provides scikit-learn estimators for time series forecasting. it leverages the benefit of using scikit-learn syntax and components to easily access the open source ecosystem built on top of the scikit-learn community. It easily maps a complex time series forecasting problems into a tabular supervised regression task, solving it with a standard approach. 

## Overview

tspiral provides 4 optimized forecasting techniques:

- **Recursive Forecasting** 

Lagged target features are combined with exogenous regressors (if provided) and lagged exogenous features (if specified). A scikit-learn compatible regressor is fitted on the whole merged data. The fitted estimator is called iteratively to predict multiple steps ahead.

![recursive-standard](https://raw.githubusercontent.com/cerlymarco/tspiral/master/imgs/recursive-standard.PNG)

Which in a compact way we can summarize in:

![recursive-compact](https://raw.githubusercontent.com/cerlymarco/tspiral/master/imgs/recursive-compact.PNG)

- **Direct Forecasting** 

A scikit-learn compatible regressor is fitted on the lagged data for each time step to forecast.

![direct](https://raw.githubusercontent.com/cerlymarco/tspiral/master/imgs/direct.PNG)

- **Stacking Forecasting** 

Multiple recursive time series forecasters are fitted and combined on the final portion of the training data with a meta-learner.

![stacked](https://raw.githubusercontent.com/cerlymarco/tspiral/master/imgs/stacked.PNG)

- **Rectified Forecasting** 

Multiple recursive time series forecasters are fitted on different sliding window training bunches. Forecasts are adjusted and combined fitting a meta-learner for each forecasting step.

![rectify](https://raw.githubusercontent.com/cerlymarco/tspiral/master/imgs/rectify.PNG)

Multivariate time series forecasting is natively supported for all the forecasting methods available.

## Installation
```shell
pip install --upgrade tspiral
```
The module depends only on NumPy and Scikit-Learn (>=0.24.2). Python 3.6 or above is supported.

## Usage

- **Recursive Forecasting** 
```python
import numpy as np
from sklearn.linear_model import Ridge
from tsprial.forecasting import ForecastingCascade
timesteps = 400
e = np.random.normal(0,1, (timesteps,))
y = 2*np.sin(np.arange(timesteps)*(2*np.pi/24))+e
model = ForecastingCascade(
    Ridge(),
    lags=range(1,24+1),
    use_exog=False,
    accept_nan=False
)
model.fit(np.arange(len(y)), y)
forecasts = model.predict(np.arange(24*3))
```

- **Direct Forecasting** 
```python
import numpy as np
from sklearn.linear_model import Ridge
from tsprial.forecasting import ForecastingChain
timesteps = 400
e = np.random.normal(0,1, (timesteps,))
y = 2*np.sin(np.arange(timesteps)*(2*np.pi/24))+e
model = ForecastingChain(
    Ridge(),
    n_estimators=24,
    lags=range(1,24+1),
    use_exog=False,
    accept_nan=False
)
model.fit(np.arange(len(y)), y)
forecasts = model.predict(np.arange(24*3))
```

- **Stacking Forecasting** 
```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from tsprial.forecasting import ForecastingStacked
timesteps = 400
e = np.random.normal(0,1, (timesteps,))
y = 2*np.sin(np.arange(timesteps)*(2*np.pi/24))+e
model = ForecastingStacked(
    [Ridge(), DecisionTreeRegressor()],
    test_size=24*3,
    lags=range(1,24+1),
    use_exog=False
)
model.fit(np.arange(len(y)), y)
forecasts = model.predict(np.arange(24*3))
```

- **Rectified Forecasting** 
```python
import numpy as np
from sklearn.linear_model import Ridge
from tsprial.forecasting import ForecastingRectified
timesteps = 400
e = np.random.normal(0,1, (timesteps,))
y = 2*np.sin(np.arange(timesteps)*(2*np.pi/24))+e
model = ForecastingRectified(
    Ridge(),
    n_estimators=200,
    test_size=24*3,
    lags=range(1,24+1),
    use_exog=False
)
model.fit(np.arange(len(y)), y)
forecasts = model.predict(np.arange(24*3))
```

More examples in the [notebooks folder](https://github.com/cerlymarco/tspiral/tree/main/notebooks).