"""
Pytest for utils.checks
- check_data_structure requires pandas dataframe
- check_data_structure raises errors for any other type of data structure
- check_dimensions should enforce that underlying dataframe is 2D
- check_dimensions raises errors for data structures of 1D or 3D+ dimensions
- check_missingness should enforce dataframes have observed and missing values
- check_missingness raises errors for fully complete or missing dataframes
"""

import pytest
import numpy as np
import pandas as pd
import autoimpute.utils.checks as auc

@auc.check_data_structure
def check_data(data):
    """wrapper function to test data structure decorator"""
    return data

@auc.check_dimensions
def check_dims(data):
    """wrapper function to test data dimensions decorator"""
    return data

@auc.check_missingness
def check_miss(data):
    """wrapper function to test missingness decorator"""
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

def data_frame_allowed():
    """Types that should not throw an error and should return a valid array"""
    df_ = pd.DataFrame({"A": [1, 2, 3, 4],
                        "B": ["a", "b", "c", "d"]})
    return [df_]

def missingness_not_allowed():
    """Can't impute datasets that are fully complete or incomplete"""
    df_none = pd.DataFrame({"A": [np.nan, np.nan, np.nan],
                            "B": [None, None, None]})
    df_full = pd.DataFrame({"A": [4, 5, 6],
                            "B": ["a", "b", "c"]})
    return [df_none, df_full]

@pytest.mark.parametrize("ds", data_structures_not_allowed())
def test_data_structures_not_allowed(ds):
    """check data structure func raises type error for disallowed types"""
    with pytest.raises(TypeError):
        check_data(ds)

@pytest.mark.parametrize("ds", data_frame_allowed())
def test_data_structures_allowed(ds):
    """check that data structure func returns expected types"""
    assert isinstance(ds, pd.DataFrame)

@pytest.mark.parametrize("ds", data_frame_allowed())
def test_dimensions_allowed(ds):
    """check that dimensions func allows 2D data structures"""
    arr = check_dims(ds)
    assert len(arr.shape) == 2

@pytest.mark.parametrize("ds", missingness_not_allowed())
def test_missingness_not_allowed(ds):
    """check data structure func raises type error for disallowed types"""
    with pytest.raises(ValueError):
        check_miss(ds)
