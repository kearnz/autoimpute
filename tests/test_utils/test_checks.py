"""Tests written to ensure the decorators in the utils package work correctly.

Tests use the pytest library. The tests in this module ensure the following:
- check_data_structure requires pandas dataframe
- check_data_structure raises errors for any other type of data structure
- check_missingness should enforce dataframes have observed and missing values
- check_missingness raises errors for fully missing dataframes
- check_missingness issues a warning for fully complete dataframes

Tests:
    test_data_structures_not_allowed(ds)
    test_data_structures_allowed(ds)
    test_missingness_not_allowed(ds)

Todo:
    * Rewrite tests when numpy array and scipy sparse matrices accepted
    * Extend missingness tests to arrays and sparse matrices
"""

import pytest
import numpy as np
import pandas as pd
from autoimpute.utils.checks import check_data_structure, check_missingness

@check_data_structure
def check_data(data):
    """Helper function to test data structure decorator"""
    return data

@check_missingness
def check_miss(data):
    """Helper function to test missingness decorator"""
    return data

def data_structures_not_allowed():
    """Types that should throw an error for check_data_structure"""
    str_ = "string"
    int_ = 1
    float_ = 1.0
    set_ = set([1, 2, 3])
    dict_ = dict(a=str_, b=int_, c=float_)
    list_ = [str_, int_, float_]
    tuple_ = tuple(list_)
    arr_ = np.array([1, 2, 3, 4])
    ser_ = pd.Series({"a": arr_})
    return [str_, int_, float_, set_, dict_,
            list_, tuple_, arr_, ser_]

def data_stuctures_allowed():
    """Types that should not throw an error and should return a valid array"""
    df_ = pd.DataFrame({"A": [1, 2, 3, 4],
                        "B": ["a", "b", "c", "d"]})
    return [df_]

def missingness_not_allowed():
    """Can't impute datasets that are fully complete or incomplete"""
    df_none = pd.DataFrame({"A": [np.nan, np.nan, np.nan],
                            "B": [None, None, None]})
    return [df_none]

@pytest.mark.parametrize("ds", data_structures_not_allowed())
def test_data_structures_not_allowed(ds):
    """Check data structure func raises type error for disallowed types

    Utilizes the pytest.mark.parametize method to run test on numerous data
    structures. Those data structures are returned from the helper method
    `data_structures_not_allowed()` which returns a list of data structures
    for which this method should throw an error. Each item in the list
    takes the on the "ds" name in pytest.

    Args:
        ds (any -> iterator): any data structure within an iterator. ds is the
            alias each item in the iterator takes when being tested

    Raises:
        TypeError: data structure ds is not allowed
    """
    with pytest.raises(TypeError):
        check_data(ds)

@pytest.mark.parametrize("ds", data_stuctures_allowed())
def test_data_structures_allowed(ds):
    """Check that data structure func returns expected types

    Utilizes the pytest.mark.parametize method to run test on numerous data
    structures. Those data structures are returned from the helper method
    `data_structures_allowed()`, which right now returns a DataFrame only.

    Args:
        ds (any -> iterator): any data structure within an iterator. ds is the
            alias each item in the iterator takes when being tested

    Returns:
        None: makes an assertion as to what the appropriate type is
    """
    assert isinstance(ds, pd.DataFrame)

@pytest.mark.parametrize("ds", missingness_not_allowed())
def test_missingness_not_allowed(ds):
    """Check missingness func raises ValueError for fully missing DataFrame

    Also utilizes the pytest.mark.parametize method to run test. Tests run on
    items in iterator returned from `missingness_not_allowed()`, which right
    now returns a fully missing DataFrame only.

    Args:
        ds (any -> iterator): any data structure within an iterator. ds is the
            alias each item in the iterator takes when being tested

    Raises:
        ValueError: if the DataFrame is fully missing
    """
    with pytest.raises(ValueError):
        check_miss(ds)
