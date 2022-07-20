<img alt="autoimpute-logo" class="autoimpute-logo" height="250" width="500" src="https://kearnz.github.io/autoimpute-tutorials/img/home/autoimpute-logo-transparent.png">

# Autoimpute
[![PyPI version](https://badge.fury.io/py/autoimpute.svg)](https://badge.fury.io/py/autoimpute) [![Build Status](https://github.com/kearnz/autoimpute/actions/workflows/build.yml/badge.svg)](https://github.com/kearnz/autoimpute/actions) [![Documentation Status](https://readthedocs.org/projects/autoimpute/badge/?version=latest)](https://autoimpute.readthedocs.io/en/latest/?badge=latest) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

<span style="font-size:1.5em;">[`Autoimpute`](https://pypi.org/project/autoimpute/) is a Python package for analysis and implementation of <b>Imputation Methods!</b></span>

<span style="font-size:1.5em;">[View our website](https://kearnz.github.io/autoimpute-tutorials/) to explore `Autoimpute` in more detail. New tutorials coming soon!</span>

<span style="font-size:1.5em;">[Check out our docs](https://autoimpute.readthedocs.io/en/latest/) to get the developer guide to `Autoimpute`.</span>

## Conference Talks
* We presented `Autoimpute` at a couple of PyData conferences!
* **[PyData NYC](https://pydata.org/nyc2019/schedule/presentation/96/up-and-coming/)**: New and Upcoming slot in November 2019
* **[PyData LA](https://pydata.org/la2019/schedule/presentation/22/introducing-autoimpute-a-python-package-for-grappling-with-missing-data/)**: Main talk slot in December 2019. See the video [here!](https://www.youtube.com/watch?v=QZRwg1NHB_Y&t=2041s)

## Notes on Development
* v 0.13+ has upgraded to support pymc, the next generation of the pymc3 library
* v 0.12+ contains the new `MiceImputer`. Thanks to [@gjdv](https://github.com/gjdv) for the help on this issue!  
* We're looking to collaborate and happy to work with those interested!
* If you'd like to get involved, feel free to reach out! Our info is on the [Authors](https://github.com/kearnz/autoimpute/blob/master/AUTHORS.rst) page.

## Creators and Maintainers
Joseph Kearney – [@kearnz](https://github.com/kearnz)  
Shahid Barkat - [@shabarka](https://github.com/shabarka)  
Arnab Bose (Advisor) - [@bosearnab](https://github.com/bosearnab)  
See the [Authors](https://github.com/kearnz/autoimpute/blob/master/AUTHORS.rst) page to get in touch!

## Installation
* `Autoimpute` is **registered with PyPI!** Download with `pip install autoimpute`.
* If `pip` cached an older version, try `pip install --no-cache-dir --upgrade autoimpute`.
* If you want to work with the development branch, use the script below:

*Development*
```sh
git clone -b dev --single-branch https://github.com/kearnz/autoimpute.git
cd autoimpute
python setup.py install
```

## Motivation
Most machine learning algorithms expect clean and complete datasets, but real-world data is messy and missing. Unfortunately, handling missing data is quite complex, so programming languages generally punt this responsibility to the end user. By default, R drops all records with missing data - a method that is easy to implement but often problematic in practice. For richer imputation strategies, R has multiple packages to deal with missing data (`MICE`, `Amelia`, `TSImpute`, etc.). Python users are not as fortunate. Python's `scikit-learn` throws a runtime error when an end user deploys models on datasets with missing records, and few third-party packages exist to handle imputation end-to-end.

Therefore, this package aids the Python user by providing more clarity to the imputation process, making imputation methods more accessible, and measuring the impact imputation methods have in supervised regression and classification. In doing so, this package brings missing data imputation methods to the Python world and makes them work nicely in Python machine learning projects (and specifically ones that utilize `scikit-learn`). Lastly, this package provides its own implementation of supervised machine learning methods that extend both `scikit-learn` and `statsmodels` to mutiply imputed datasets.

## Main Features
* Utility functions to examine patterns in missing data and decide on relevant features for imputation
* Missingness classifier and automatic missing data test set generator
* Native handling for categorical variables (as predictors and targets of imputation)
* Single and multiple imputation classes for `pandas` `DataFrames`
* Custom visualization support for utility functions and imputation methods
* Analysis methods and pooled parameter inference using multiply imputed datasets
* Numerous imputation methods, as specified in the table below:

## Imputation Methods Supported

| Univariate                  | Multivariate                        | Time Series / Interpolation
| :-------------------------- | :---------------------------------- | ---------------------------
| Mean                        | Linear Regression                   | Linear 
| Median                      | Binomial Logistic Regression        | Quadratic 
| Mode                        | Multinomial Logistic Regression     | Cubic
| Random                      | Stochastic Regression               | Polynomial
| Norm                        | Bayesian Linear Regression          | Spline
| Categorical                 | Bayesian Binary Logistic Regression | Time-weighted
|                             | Predictive Mean Matching            | Next Obs Carried Backward
|                             | Local Residual Draws                | Last Obs Carried Forward

## Todo
* Additional cross-sectional methods, including random forest, KNN, EM, and maximum likelihood
* Additional time-series methods, including EWMA, ARIMA, Kalman filters, and state-space models
* Extended support for visualization of missing data patterns, imputation methods, and analysis models
* Additional support for analysis metrics and analyis models after multiple imputation
* Multiprocessing and GPU support for larger datasets, as well as integration with `dask` DataFrames

## Example Usage
Autoimpute is designed to be user friendly and flexible. When performing imputation, Autoimpute fits directly into `scikit-learn` machine learning projects. Imputers inherit from sklearn's `BaseEstimator` and `TransformerMixin` and implement `fit` and `transform` methods, making them valid Transformers in an sklearn pipeline.

Right now, there are three `Imputer` classes we'll work with:
```python
from autoimpute.imputations import SingleImputer, MultipleImputer, MiceImputer
si = SingleImputer() # pass through data once
mi = MultipleImputer() # pass through data multiple times
mice = MiceImputer() # pass through data multiple times and iteratively optimize imputations in each column
```

#### Which to use, and When?
* There are tradeoffs between the three imputers 
* We won't get into the specifics regarding why, but here are a couple points to keep in mind:  
* Execution time (best to worst): `SingleImputer`, `MultipleImputer`, `MiceImputer`  
* Imputation quality (best to worst): `MiceImputer`, `MultipleImputer`, `SingleImputer`  
* This shouldn't come as a surprise. The `MiceImputer` does the most work, while the `SingleImputer` does the least  
* The example below use the `MiceImputer`, but you can swap in the `MultipleImputer` or `SingleImputer` as well  

Imputations can be as simple as:
```python
# simple example using default instance of MiceImputer
imp = MiceImputer()

# fit transform returns a generator by default, calculating each imputation method lazily
imp.fit_transform(data)
```

Or quite complex, such as:
```python
# create a complex instance of the MiceImputer
# Here, we specify strategies by column and predictors for each column
# We also specify what additional arguments any `pmm` strategies should take
imp = MiceImputer(
    n=10,
    strategy={"salary": "pmm", "gender": "bayesian binary logistic", "age": "norm"},
    predictors={"salary": "all", "gender": ["salary", "education", "weight"]},
    imp_kwgs={"pmm": {"fill_value": "random"}},
    visit="left-to-right",
    return_list=True
)

# Because we set return_list=True, imputations are done all at once, not evaluated lazily.
# This will return M*N, where M is the number of imputations and N is the size of original dataframe.
imp.fit_transform(data)
```

Autoimpute also extends supervised machine learning methods from `scikit-learn` and `statsmodels` to apply them to multiply imputed datasets (using the `MiceImputer` under the hood). Right now, Autoimpute supports linear regression and binary logistic regression. Additional supervised methods are currently under development.

As with Imputers, Autoimpute's analysis methods can be simple or complex:
```python
from autoimpute.analysis import MiLinearRegression

# By default, use statsmodels OLS and MiceImputer()
simple_lm = MiLinearRegression()

# fit the model on each multiply imputed dataset and pool parameters
simple_lm.fit(X_train, y_train)

# get summary of fit, which includes pooled parameters under Rubin's rules
# also provides diagnostics related to analysis after multiple imputation
simple_lm.summary()

# make predictions on a new dataset using pooled parameters
predictions = simple_lm.predict(X_test)

# Control both the regression used and the MiceImputer itself
mice_imputer_arguments = dict(
    n=3,
    strategy={"salary": "pmm", "gender": "bayesian binary logistic", "age": "norm"},
    predictors={"salary": "all", "gender": ["salary", "education", "weight"]},
    imp_kwgs={"pmm": {"fill_value": "random"}},
    visit="left-to-right"
)
complex_lm = MiLinearRegression(
    model_lib="sklearn", # use sklearn linear regression
    mi_kwgs=mice_imputer_arguments # control the multiple imputer
)

# fit the model on each multiply imputed dataset
complex_lm.fit(X_train, y_train)

# get summary of fit, which includes pooled parameters under Rubin's rules
# also provides diagnostics related to analysis after multiple imputation
complex_lm.summary()

# make predictions on new dataset using pooled parameters
predictions = complex_lm.predict(X_test)
```

Note that we can also pass a pre-specified `MiceImputer` (or `MultipleIputer`) to either analysis model instead of using `mi_kwgs`. The option is ours, and it's a matter of preference. If we pass a pre-specified `MiceImputer`, anything in `mi_kwgs` is ignored, although the `mi_kwgs` argument is still validated.

```python
from autoimpute.imputations import MiceImputer
from autoimpute.analysis import MiLinearRegression

# create a multiple imputer first
custom_imputer = MiceImputer(n=3, strategy="pmm", return_list=True)

# pass the imputer to a linear regression model
complex_lm = MiLinearRegression(mi=custom_imputer, model_lib="statsmodels")

# proceed the same as the previous examples
complex_lm.fit(X_train, y_train).predict(X_test)
complex_lm.summary()
```

For a deeper understanding of how the package works and its available features, see our [tutorials website](https://kearnz.github.io/autoimpute-tutorials/).

## Versions and Dependencies
* Python 3.8+
* Dependencies:
    - `numpy`
    - `scipy`
    - `pandas`
    - `statsmodels`
    - `scikit-learn`
    - `xgboost`
    - `pymc`
    - `seaborn`
    - `missingno`
* See `requirements.txt` for versioning info

*A note for Windows Users*:
* Autoimpute has NOT been tested on Windows using the upgrade to pymc. Historically, autoimpute has had trouble on Windows, as demonstrated by bullets below.
* Autoimpute works on Windows but users may have trouble with pymc for bayesian methods. [(See discourse)](https://discourse.pymc.io/t/an-error-message-about-cant-pickle-fortran-objects/1073)
* Users may receive a runtime error `‘can’t pickle fortran objects’` when sampling using multiple chains.
* There are a couple of things to do to try to overcome this error:
    - Reinstall theano and pymc. Make sure to delete .theano cache in your home folder.
    - Upgrade joblib in the process, which is reponsible for generating the error (pymc uses joblib under the hood).
    - Set `cores=1` in `pm.sample`. This should be a last resort, as it means posterior sampling will use 1 core only. Not using multiprocessing will slow down bayesian imputation methods significantly.
* Reach out and let us know if you've worked through this issue successfully on Windows and have a better solution!

## License
Distributed under the MIT license. See [LICENSE](https://github.com/kearnz/autoimpute/blob/master/LICENSE) for more information.

## Contributing
Guidelines for contributing to our project. See [CONTRIBUTING](https://github.com/kearnz/autoimpute/blob/master/CONTRIBUTING.md) for more information.

## Contributor Code of Conduct
Adapted from Contributor Covenant, version 1.0.0. See [Code of Conduct](https://github.com/kearnz/autoimpute/blob/master/CODE_OF_CONDUCT.md) for more information.
