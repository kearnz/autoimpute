"""
Pytest for utils.checks
- check_data_structure requires numpy array, pandas dataframe or list/tuple
- check_data_structure should return a tuple, with np.array
    - lists, tuples, and dataframes are arrays of dtype object
    - traditional numpy arrays and subclasses of arrays are float
"""

import pytest
from pandas import Series
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
    dict_ = dict(a="string", b=1, c=1.0)
    ser_ = Series({"a":[1, 2, 3, 4]})
    return [str_, int_, float_, set_, dict_, ser_]

@pytest.mark.parametrize("ds", data_structures_not_allowed())
def test_data_structure_not_allowed(ds):
    """check that both decorators raise a type error for strings"""
    with pytest.raises(TypeError):
        check_data(ds)
