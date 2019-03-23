# AutoImpute
[![Build Status](https://travis-ci.org/kearnz/autoimpute.svg?branch=master)](https://travis-ci.org/kearnz/autoimpute)
<span style="font-size:1.5em;">A Python package for analysis and implementation of <b>Imputation Methods!</b></span>

## Motivation
Most machine learning algorithms expect clean and complete datasets, but most real-world data is messy and missing. Unfortunately, handling missing data is quite complex, so programming languages generally punt this this responsibility to the end user! By default, R drops all records with missing data - a method that is easy to implement but often problematic. For richer imputation strategies, R has multiple packages to deal with missing data (`MICE`, `Amelia`, `TSImpute`, etc.). Python users are not as fortunate. Python's `scikit-learn` throws a runtime error when an end user attempts to deploy models on datasets with missing records, and few 3rd-party packages exist to handle imputation.

Therefore, this package strives to aid the Python user by providing more clarity to the imputation process, making imputation methods more accessible, and measuring the impact imputation methods have in supervised regression and classification. In doing so, this package brings missing data imputation methods to the Python world and makes them work nicely in Python machine learning projects and specifically `Pipelines` in `scikit-learn`.

## Features
* Utility functions to explore missingness patterns
* Missingness classifier and automatic test set generator
* Cross-sectional and time series imputation methods. Imputation methods currently supported:
    - Mean
    - Median
    - Mode
    - Random
    - Norm
    - Categorical
    - Linear interpolation
    - Time-weighted interpolation
    - Last observation carried forward (LOCF)
    - Next observation carried backward (NOCB)
    - Least squares (linear regression)
    - Binary logistic regression
    - Multinomial logistic regression
    - Linear regression with stochastic error
    - Bayesian linear regression
    - Bayesian binary logistic regression
    - Predictive mean matching
* Note that some methods offer multiple choices within them:
    - For mode, use first (default in scipy and autoimpute) or randomly sample from modes (if more than one)
    - For bayesian methods, use random draw from or $\mu$ of posterior predictive dist. (or posterior for pmm)
    - Also for bayesian methods, multiple MCMC options possible to pass explicitly. See [pymc3 docs](https://docs.pymc.io/) for more. 
    - For missingness classifier, default is XGBoost, although any valid sklearn classifier can be used so long as it implements `predict_proba` method from sklearn. Therefore, classifiers must support probabilistic assignments.

## Todo
* Native support for Mutliple imputation using the imputation methods implemented.
* Additional cross-sectional methods, including random forest, multivariate sampling, copula sampling, and ML.
* Additional time-series methods, including ARIMA, Kalman filters, splines, and state-space models.
* Native support for visualization of missing data patterns and imputation results.
* Native support for imputation analysis (bias, MI variance, etc.)
* Multiprocessing support and GPU support for larger datasets.

## Example Usage
Autoimpute is designed to be user friendly and flexible. Additionally, autoimpute fits directly into sklearn machine learning projects. Imputers inherit from sklearn's `BaseEstimator` and `TransformerMixin` and implement `fit` and `transform` methods, making them valid Transformers in an sklearn pipeline.

Right now, there are four main classes you'll work with:
```python
si = SingleImputer() # basic and quick methods for imputation
ti = TimeSeriesImputer() # optimized for time series imputation
pi = PredictiveImputer() # methods using all or some available information
mc = MissingnessClassifier() # predicting missingness and generating test sets for imputation analysis
```

Imputations can be as simple as:
```python
imp = PredictiveImputer()
imp.fit_transform(data)
```

Or quite complex, such as:
```python
from sklearn.preprocessing import MinMaxScaler
imp = PredictiveImputer(
    strategy={"x1": "pmm", "x2": "stochastic", "y": "norm"},
    predictors={"x1": "x2", "y": "all"},
    scaler=MinMaxScaler,
    fill_value="random",
    verbose=True
)
imp.fit_transform(data)
```

For a deeper understanding of how the package works and its available features, see our [tutorials](https://github.com/kearnz/autoimpute/blob/master/tutorials)

## Versions and Dependencies
* Python 3.6+
* Dependencies:
    - `numpy` >= 1.15.4
    - `scipy` >= 1.2.1
    - `pandas` >= 0.20.3
    - `scikit-learn` >= 0.20.2
    - `xgboost` >= 0.83
    - `pymc3` >= 3.5

## Installation
OS X & Linux:

```sh
git clone https://github.com/kearnz/autoimpute.git
cd autoimpute
python setup.py install
```

Using a Virtual Environment:

```sh
virtualenv imp
source imp/bin/activate
git clone https://github.com/kearnz/autoimpute.git
cd autoimpute
python setup.py install
```

A note for Windows Users:
* AutoImpute is tested and works on Windows as well, although some users may have trouble with bayesian methods using pymc3. [(See discourse here)](https://discourse.pymc.io/t/an-error-message-about-cant-pickle-fortran-objects/1073)
* Users may receive a runtime error `‘can’t pickle fortran objects’` when sampling using multiple chains.
* There are a couple of things to do to try to overcome this error:
    - Reinstall theano and pymc3. Make sure to delete .theano cache in your home folder.
    - Upgrade joblib in the process, which is reponsible for generating the error (pymc3 uses joblib under the hood).
    - Set `cores=1` in `pm.sample`. This should be a last resort, as it means posterior sampling will use 1 core only. Not using multiprocessing will slow down bayesian imputation methods significantly.
* Reach out and let us know if you've worked through this issue successfully on Windows and have a better solution!

## Contact
Joseph Kearney – [@kearnz](https://github.com/kearnz)  
Shahid Barkat - [@shabarka](https://github.com/shabarka)

## License
Distributed under the MIT license. See [LICENSE](https://github.com/kearnz/autoimpute/blob/master/LICENSE) for more information.

## Contributor Code of Conduct
Adapted from Contributor Covenant, version 1.0.0. See [Code of Conduct](https://github.com/kearnz/autoimpute/blob/master/CODE_OF_CONDUCT)