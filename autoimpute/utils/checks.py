"""Module to check and validate data types that play nicely with imputation"""

import functools
import warnings
import pandas as pd

def check_data_structure(func):
    """
    Check if input value is a dataframe.
    Note: Future allowed iterators include:
    - native: tuple, list
    - numpy: ndarray or sublcass of ndarray (such as np.mat)
    Right now, decorator returns original dataframe
    Least restrictive - simply ensures data type.
    """
    @functools.wraps(func)
    def wrapper(d, *args, **kwargs):
        """Wrapper function to data structure"""
        if isinstance(d, pd.DataFrame):
            return func(d, *args, **kwargs)
        elif isinstance(args[0], pd.DataFrame):
            return func(d, *args, **kwargs)
        else:
            err = d.__class__.__name__
            raise TypeError(f"Type '{err}' is not an accepted data structure")
    return wrapper

def check_dimensions(func):
    """
    Check if data structure is 2D.
    If not, throw an error b/c 1D or 3D+ arrays not supported.
    Leverages the _check_data_structure decorator.
    Decorator returns data as is or throws an error.
    Medium restrictive - ensures data type and shape.
    """
    @functools.wraps(func)
    @check_data_structure
    def wrapper(d, *args, **kwargs):
        """Wrap function to dims"""
        if len(d.shape) == 2:
            return func(d, *args, **kwargs)
        elif len(args[0].shape) == 2:
            return func(d, *args, **kwargs)
        else:
            err = len(d.shape)
            raise TypeError(f"{err}-dimensional arrays not supported")
    return wrapper

def check_missingness(func):
    """
    Check if data structure is all missing or no missing.
    If all missing, give warning, as nothing to impute.
    If no missing, throw error, as nothing to impute.
    If some missing, simply return func and data.
    Most restrictive - ensures data type, shape, and missingness.
    """
    @functools.wraps(func)
    @check_dimensions
    def wrapper(d, *args, **kwargs):
        """Wrap function to missignness"""
        if isinstance(d, pd.DataFrame):
            missing = pd.isnull(d.values)
        if isinstance(args[0], pd.DataFrame):
            missing = pd.isnull(args[0].values)
        if missing.all():
            raise ValueError("All values missing, need some complete")
        else:
            if not missing.any():
                warnings.warn("No missing values, so nothing to impute")
            return func(d, *args, **kwargs)
    return wrapper
