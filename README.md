# AutoImpute
[![Build Status](https://travis-ci.com/kearnz/autoimpute.svg?branch=master)](https://travis-ci.com/kearnz/autoimpute)  
<span style="font-size:1.5em;">A Python package for analysis and implementation of <b>Imputation Methods!</b></span>

## Motivation
Most machine learning algorithms expect clean and complete datasets, but most real-world data is messy and missing. Unfortunately, handling missing data is quite complex, so programming languages generally punt this responsibility to the end user. By default, R drops all records with missing data - a method that is easy to implement but often problematic in practice. For richer imputation strategies, R has multiple packages to deal with missing data (`MICE`, `Amelia`, `TSImpute`, etc.). Python users are not as fortunate. Python's `scikit-learn` throws a runtime error when an end user attempts to deploy models on datasets with missing records, and few 3rd-party packages exist to handle imputation.

Therefore, this package aids the Python user by providing more clarity to the imputation process, making imputation methods more accessible, and measuring the impact imputation methods have in supervised regression and classification. In doing so, this package brings missing data imputation methods to the Python world and makes them work nicely in Python machine learning projects (and specifically ones that utilize `scikit-learn`). Lastly, this package provides its own implementation of supervised machine learning methods that extend both `scikit-learn` and `statsmodels` to mutiply imputed datasets.

## Features
* Utility functions to explore missingness patterns
* Missingness classifier and automatic missing data test set generator
* Single and Multiple Imputation
* Analysis methods and parameter inference using multiply imputed datasets
* Cross-sectional and time series imputation methods. Imputation methods currently supported:
    - Mean
    - Median
    - Mode
    - Random
    - Norm
    - Categorical
    - Linear interpolation
    - Time-weighted interpolation
    - quadratic, cubic, and polynomial interpolation
    - Spline interpolation
    - Last observation carried forward (LOCF)
    - Next observation carried backward (NOCB)
    - Least squares (linear regression)
    - Binary logistic regression
    - Multinomial logistic regression
    - Linear regression with stochastic error
    - Bayesian linear regression
    - Bayesian binary logistic regression
    - Predictive mean matching

## Todo
* Additional cross-sectional methods, including random forest, multivariate sampling, copula sampling, and ML
* Additional time-series methods, including ARIMA, Kalman filters, and state-space models
* Native support for visualization of missing data patterns and imputation results
* Additional support for analysis (bias, MI variance, etc.) after multiple imputation
* Multiprocessing and GPU support for larger datasets, as well as integration with `dask` dataframes.

## Example Usage
Autoimpute is designed to be user friendly and flexible. When performing imputation, autoimpute fits directly into sklearn machine learning projects. Imputers inherit from sklearn's `BaseEstimator` and `TransformerMixin` and implement `fit` and `transform` methods, making them valid Transformers in an sklearn pipeline.

Right now, there are two imputer classes you'll work with:
```python
from autoimpute.imputations import SingleImputer, MultipleImputer
si = SingleImputer() # imputation methods, passing through the data once
mi = MultipleImputer() # imputation methods, passing through the data multiple times
```

Imputations can be as simple as:
```python
# simple example using default instance of MultipleImputer
imp = MultipleImputer()

# fit transform returns a generator by default, calculating each imputation method lazily
imp.fit_transform(data)
```

Or quite complex, such as:
```python
from sklearn.preprocessing import StandardScaler

# create a complex instance of the MultipleImputer
# Here, we specify strategies by column and predictors for each column
# We also specify what additional arguments any `pmm` strategies should take
imp = MultipleImputer(
    n=10,
    strategy={"salary": "pmm", "gender": "bayesian binary logistic", "age": "norm"},
    predictors={"salary": "all", "gender": ["salary", "education", "weight"]},
    imp_kwgs={"pmm": {"fill_value": "random"}},
    scaler=StandardScaler(),
    visit="left-to-right",
    return_list=True
    verbose=True
)

# Because we set return_list=True, imputations are done all at once, not evaluated lazily.
# This will return M*N, where M is the number of imputations and N is the size of original dataframe.
imp.fit_transform(data)
```

Autoimpute also extends supervised machine learning methods from `scikit-learn` and `statsmodels` to apply them to multiply imputed datasets (using the `MultipleImputer` under the hood). Right now, autoimpute supports linear regression and binary logistic regression. Additional supervised methods are currently under development.

As with Imputers, Autoimpute's analysis methods can be simple or complex:
```python
from autoimpute.analysis import MiLinearRegression

# By default, use statsmodels OLS and MultipleImputer()
simple_lm = MiLinearRegression()

# fit the model on each multiply imputed dataset and pool parameters
simple_lm.fit(X_train, y_train)

# retrieve pooled parameters under Rubin's rules
print(simple_lm.statistics_["coefs"]) # pooled means for alpha and betas
print(simple_lm.statistics_["var_within"]) # variance within imputations (Vw)
print(simple_lm.statistics_["var_between"]) # variance between imputations (Vb)
print(simple_lm.statistics_["var_total"]) # Total variance (Vw + Vb + Vb / M) where M = # imputations

# make predictions on a new dataset using pooled parameters
simple_lm.predict(X_test)

# Control both the regression used and the MultipleImputer itself
multiple_imputer_arguments = dict(
    n=3,
    strategy={"salary": "pmm", "gender": "bayesian binary logistic", "age": "norm"},
    predictors={"salary": "all", "gender": ["salary", "education", "weight"]},
    imp_kwgs={"pmm": {"fill_value": "random"}},
    scaler=StandardScaler(),
    visit="left-to-right",
    verbose=True
)
complex_lm = MiLinearRegression(
    model_lib="sklearn", # use sklearn linear regression
    mi_kwgs=multiple_imputer_arguments # control the multiple imputer
)

# fit the model on each multiply imputed dataset
complex_lm.fit(X_train, y_train)

# Note - using sklearn means NO POOLED VARIANCE. Pooled coefficients only
print(complex_lm.statistics_)

# make predictions on new dataset using pooled parameters
complex_lm.predict(X_test)
```

For a deeper understanding of how the package works and its available features, see our [tutorials](https://github.com/kearnz/autoimpute-tutorials/tree/master/tutorials).

## Versions and Dependencies
* Python 3.6+
* Dependencies:
    - `numpy` >= 1.15.4
    - `scipy` >= 1.2.1
    - `pandas` >= 0.20.3
    - `statsmodels` >= 0.8.0
    - `scikit-learn` >= 0.20.2
    - `xgboost` >= 0.83
    - `pymc3` >= 3.5

## Installation
* Autoimpute will be registered with PyPI soon after its first release, so `pip install` coming soon!
* In the meantime, the following work for Mac OS & Linux as well as Windows (with a couple caveats).

*Master*
```sh
git clone https://github.com/kearnz/autoimpute.git
cd autoimpute
python setup.py install
```

*Development*
```sh
git clone -b dev --single-branch https://github.com/kearnz/autoimpute.git
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
* AutoImpute works on Windows but users may have trouble with pymc3 for bayesian methods. [(See discourse)](https://discourse.pymc.io/t/an-error-message-about-cant-pickle-fortran-objects/1073)
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

## Contributing
Guidelines for contributing to our project. See [CONTRIBUTING](https://github.com/kearnz/autoimpute/blob/master/CONTRIBUTING.md)

## Contributor Code of Conduct
Adapted from Contributor Covenant, version 1.0.0. See [Code of Conduct](https://github.com/kearnz/autoimpute/blob/master/CODE_OF_CONDUCT.md)