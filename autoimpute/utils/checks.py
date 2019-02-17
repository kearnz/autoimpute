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
        else:
            if args:
                a = args[0]
                if isinstance(a, pd.DataFrame):
                    return func(d, *args, **kwargs)
                else:
                    err = a.__class__.__name__
                    raise TypeError(f"Type '{err}' not a dataframe")
            else:
                err = d.__class__.__name__
                raise TypeError(f"Type '{err}' not a dataframe'")
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
    @check_data_structure
    def wrapper(d, *args, **kwargs):
        """Wrap function to missignness"""
        if isinstance(d, pd.DataFrame):
            missing = pd.isnull(d.values)
        else:
            if args:
                a = args[0]
                if isinstance(a, pd.DataFrame):
                    missing = pd.isnull(args[0].values)
        if missing.all():
            raise ValueError("All values missing, need some complete")
        else:
            if not missing.any():
                warnings.warn("No missing values, so nothing to impute")
            return func(d, *args, **kwargs)
    return wrapper
