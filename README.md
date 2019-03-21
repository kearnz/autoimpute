# AutoImpute
<span style="font-size:1.5em;">A Python package for analysis and implementation of <b>Imputation Methods!</b></span>

## Motivation
Most machine learning algorithms expect clean and complete datasets, but most real-world data is messy and missing. Unfortunately, handling missing data is quite complex, so programming languages generally punt this this responsibility to the end user! By default, R drops all records with missing data - a method that is easy to implement but often problematic. For richer imputation strategies, R has multiple packages to deal with missing data (`MICE`, `Amelia`, `TSImpute`, etc.). Python users are not as fortunate. Python's `scikit-learn` throws a runtime error when an end user attempts to deploy models on datasets with missing records, and few 3rd-party packages exist to handle imputation.

Therefore, this package strives to aid the Python user by providing more clarity to the imputation process, making imputation methods more accessible, and measuring the impact imputation methods have in supervised regression and classification. In doing so, this package brings missing data imputation methods to the Python world and makes them work nicely in Python machine learning projects and specifically `Pipelines` in `sckikit-learn`.

## Versions and Dependencies
* Python 3.6+
* Requires:
    - `numpy` >= 1.15.4
    - `scipy` >= 1.2.1
    - `pandas` >= 0.20.3
    - `scikit-learn` >= 0.20.2
    - `xgboost` >= 0.83

## Installation
OS X & Linux:

```sh
git clone https://github.com/kearnz/autoimpute.git
cd autoimpute
python setup.py install
```

## Contact
Joseph Kearney – [@kearnz](https://github.com/kearnz)  
Shahid Barkat - [@shabarka](https://github.com/shabarka)

## License
Distributed under the MIT license. See [LICENSE](https://github.com/kearnz/autoimpute/blob/master/LICENSE) for more information.