"""Module to check and validate data types that play nicely with imputation"""

import functools
import numpy as np
import pandas as pd

def check_data_structure(func):
    """
    Decorator that returns a numpy ndarray from an iterator.
    Allowed iterators include:
    - native: tuple, list
    - numpy: ndarray or sublcass of ndarray (such as np.mat)
    - pandas: Series, DataFrame
    """
    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        """Wrap function to check type"""
        if isinstance(data, np.ndarray):
            return func(data, *args, **kwargs)
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            return func(data.values, *args, **kwargs)
        elif isinstance(data, (tuple, list)):
            return func(np.array(data, dtype=object), *args, **kwargs)
        else:
            err = data.__class__.__name__
            raise TypeError(f"Type '{err}' is not an accepted data structure")
    return wrapper

def check_dimensions(func):
    """
    Check if data structure is 2D
    If not, throw an error b/c 3D+ arrays not supported
    Leverages the _check_data_structure decorator
    """
    @functools.wraps(func)
    @check_data_structure
    def wrapper(data, *args, **kwargs):
        """Wrap function to dims"""
        if len(data.shape) == 1:
            return func(data.reshape(len(data), 1), *args, **kwargs)
        elif len(data.shape) == 2:
            return func(data, *args, **kwargs)
        else:
            err = len(data.shape)
            raise TypeError(f"{err}-dimensional arrays not supported")
    return wrapper
