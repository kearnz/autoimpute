"""
Pytest for utils.checks
- check_data_structure requires numpy array, pandas dataframe or list/tuple
- check_data_structure should return a tuple, with np.array
    - lists, tuples, and dataframes are arrays of dtype object
    - traditional numpy arrays and subclasses of arrays are float
"""

import pytest
from autoimpute.utils.checks import check_data_structure, check_dimensions

@check_data_structure
def check_data(data):
    """wrapper function to test data structure decorator"""
    return data

@check_dimensions
def check_dims(data):
    """wrapper function to test data dimensions decorator"""
    return data

def test_check_data_structure_string():
    """check that both decorators raise a type error for strings"""
    with pytest.raises(TypeError):
        check_data("string")
        check_dims("string")

def test_check_data_structure_number():
    """check that both decorators return a type error for integers"""
    with pytest.raises(TypeError):
        check_data(5)
        check_dims(5)
