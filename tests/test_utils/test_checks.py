"""
Pytest for utils.checks
- check_data_structure requires numpy array, pandas dataframe or list/tuple
- check_data_structure should return with np.array
    - lists, tuples, and dataframes are arrays of dtype object
    - traditional numpy arrays and subclasses of arrays are float
"""

import pytest
import numpy as np
import pandas as pd
from autoimpute.utils.checks import check_data_structure, check_dimensions

@check_data_structure
def check_data(data):
    """wrapper function to test data structure decorator"""
    return data

@check_dimensions
def check_dims(data):
    """wrapper function to test data dimensions decorator"""
    return data

def data_structures_not_allowed():
    """Types that should throw an error for check_data_structure"""
    str_ = "string"
    int_ = 1
    float_ = 1.0
    set_ = set([1, 2, 3])
    dict_ = dict(a=str_, b=int_, c=float_)
    ser_ = pd.Series({"a":[1, 2, 3, 4]})
    return [str_, int_, float_, set_, dict_, ser_]

def data_structures_allowed():
    """Types that should not throw an error and should return a valid array"""
    list_ = [[1, 2, 3, 4, np.nan]]
    tuple_ = ((1, 2, 3, 4, np.nan))
    array_ = np.array([[1, 2, 3, 4, np.nan]])
    dataframe_ = pd.DataFrame({"A": list_, "B": list_})
    return [list_, tuple_, array_, dataframe_]

@pytest.mark.parametrize("ds", data_structures_not_allowed())
def test_data_structures_not_allowed(ds):
    """check that data structure decorator raise a type error for strings"""
    with pytest.raises(TypeError):
        check_data(ds)

@pytest.mark.parametrize("ds", data_structures_allowed())
def test_data_structures_allowed(ds):
    """check that data structure decoractor returns expected types"""
    arr = check_data(ds)
    dtype = arr.dtype
    assert isinstance(arr, np.ndarray)
    if isinstance(ds, (list, tuple, pd.DataFrame)):
        assert dtype == np.dtype('object')
    if isinstance(ds, np.ndarray):
        assert dtype in (np.dtype('int64'), np.dtype('float64'))
     