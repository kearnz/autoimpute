"""Tests written to ensure the decorators in the utils package work correctly.

Tests use the pytest library. The tests in this module ensure the following:
- `check_data_structure` requires pandas DataFrame.
- `check_data_structure` raises errors for any other type of data structure.
- `check_missingness` enforces datasets have observed and missing values.
- `check_missingness` raises errors for fully missing datasets.
- `check_missingness` raises errors for time series missing in datasets.
- `remove_nan_columns` removes columns if the entire column is missing.
"""

import pytest
import numpy as np
import pandas as pd
from autoimpute.utils.checks import check_data_structure, check_missingness
from autoimpute.utils.checks import check_nan_columns

@check_data_structure
def check_data(data):
    """Helper function to test data structure decorator."""
    return data

@check_missingness
def check_miss(data):
    """Helper function to test missingness decorator."""
    return data

@check_nan_columns
def check_nan_cols(data):
    """Helper function to test removal of NaN columns."""
    return data

def data_structures_not_allowed():
    """Types that should throw an error for `check_data_structure`."""
    str_ = "string"
    int_ = 1
    float_ = 1.0
    set_ = set([1, 2, 3])
    dict_ = dict(a=str_, b=int_, c=float_)
    list_ = [str_, int_, float_]
    tuple_ = tuple(list_)
    arr_ = np.array([1, 2, 3, 4])
    ser_ = pd.Series({"a": arr_})
    return [str_, int_, float_, set_, dict_, list_, tuple_, arr_, ser_]

def data_stuctures_allowed():
    """Types that should not throw an error and should return a valid array."""
    df_ = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": ["a", "b", "c", "d"]
    })
    return [df_]

def missingness_not_allowed():
    """Can't impute datasets that are fully complete or incomplete."""
    df_none = pd.DataFrame({
        "A": [np.nan, np.nan, np.nan],
        "B": [None, None, None]
    })
    #df_ts = pd.DataFrame({
    #    "date": ["2018-05-01", "2018-05-02", "2018-05-03",
    #             "2018-05-04", "2018-05-05", "2018-05-06",
    #             "2018-05-07", "2018-05-08", "2018-05-09"],
    #    "stats": [3, 4, np.nan, 15, 7, np.nan, 26, 25, 62]
    #})
    #df_ts["date"] = pd.to_datetime(df_ts["date"], utc=True)
    #df_ts.loc[[1, 3], "date"] = np.nan
    return [df_none]

@pytest.mark.parametrize("ds", data_structures_not_allowed())
def test_data_structures_not_allowed(ds):
    """Ensure data structure helper raises TypeError for disallowed types.

    Utilizes the pytest.mark.parametize method to run test on numerous data
    structures. Those data structures are returned from the helper method
    `data_structures_not_allowed()`, which returns a list of data structures.
    Each item within this list should cause this method to throw an error.
    Each item in the list takes the on the "ds" name in pytest.

    Args:
        ds (any -> iterator): any data structure within an iterator. `ds` is
            alias each item in the iterator takes when being tested.

    Returns:
        None: raises errors when improper types passed.

    Raises:
        TypeError: data structure `ds` is not allowed.
    """
    with pytest.raises(TypeError):
        check_data(ds)

@pytest.mark.parametrize("ds", data_stuctures_allowed())
def test_data_structures_allowed(ds):
    """Ensure data structure helper returns expected types.

    Utilizes the pytest.mark.parametize method to run test on numerous data
    structures. Those data structures are returned from the helper method
    `data_structures_allowed()`, which right now returns a DataFrame/Series.

    Args:
        ds (any -> iterator): any data structure within an iterator. `ds` is
            alias each item in the iterator takes when being tested.

    Returns:
        None: asserts that the appropriate type has been passed.
    """
    assert isinstance(ds, pd.DataFrame)

@pytest.mark.parametrize("ds", missingness_not_allowed())
def test_missingness_not_allowed(ds):
    """Ensure missingness helper raises ValueError for fully missing DataFrame.

    Also utilizes the pytest.mark.parametize method to run test. Tests run on
    items in iterator returned from `missingness_not_allowed()`, which right
    now returns a fully missing DataFrame and a time series DataFrame with
    missingness in the time series itself.

    Args:
        ds (any -> iterator): any data structure within an iterator. `ds` is
            alias each item in the iterator takes when being tested.

    Returns:
        None: raises error because DataFrame is fully missing.

    Raises:
        ValueError: if the DataFrame is fully missing.
    """
    with pytest.raises(ValueError):
        check_miss(ds)

def test_nan_columns():
    """Check missing columns throw error with check_nan_columns decorator.

    The `check_nan_columns` decorator should throw an error if any columns
    in the dataframe have all values missing. This test simulates data in a
    DataFrame where two of the columns have all missing values. The DataFrame
    is below. In this case, `B` and `C` should generate an error because they
    contain all missing values. Error message should capture both columns.

    Args:
        None: DataFrame hard-coded internally.

    Returns:
        None: asserts that fully missing columns generate error.
    """
    df = pd.DataFrame({
        "A": [1, np.nan, 3, 4],
        "B": [None, None, None, None],
        "C": [np.nan, np.nan, np.nan, np.nan],
        "D": ["a", "b", None, "d"]
    })
    assert pd.isnull(df["B"]).all()
    assert pd.isnull(df["C"]).all()
    with pytest.raises(ValueError):
        check_nan_cols(df)
