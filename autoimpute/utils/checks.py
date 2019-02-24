"""Check and validate data that play nicely with imputation

This module is a series of decorators used throughout this package.
The decorators perform checks or transformations on data to ensure it
is well formatted prior to the rest of the function running.
There are three main decorators at the time of this writing:
check_data_structure, check_missingness, & remove_nan_columns.

Methods:
    check_data_structure(func)
    check_missingness(func)
    remove_nan_columns(func)

Todo:
    * Support data structures other than pandas dataframe. Consider:
        - list, tuple, and other built-in iterators / collections
        - numpy array, pandas Series
        - scipy sparse matrices
    * If other data structures added, bring back decorator to verify 2D
        - Right now, we only accept pandas dataframes. All pandas datafrmes
          are 2-dimensional, so no need to verify their shape.
        - If we allow for arrays or other data types, we don't get
          this verification for free and need to enforce. We can introduce
          `check_dimensions` decorator, which checks shape of input data.
"""

import functools
import warnings
import pandas as pd
from autoimpute.utils.helpers import _nan_col_dropper

def check_data_structure(func):
    """Check if the data input to a function is a pandas dataframe.

    This method acts as a decorator. It takes a function that takes data
    as its first argument and verifies that the data is a pandas dataframe.
    Because this package has many functions which require a dataframe
    as their first argument, this decorator makes it easy to verify this
    requirement regardless of what the rest of the function does. It
    utilizes the `functools.wrap` decorator to keep function names in tact.

    Args:
        func (function): The function that will be decorated

    Returns:
        function: decorators return function they wrap
    """
    @functools.wraps(func)
    def wrapper(d, *args, **kwargs):
        """Wrapper function that does the pandas dataframe verification

        The wrapper within the decorator does the actual verification. It
        is flexible in that it checks whether the first argument, d, is a
        pandas dataframe. If it's not, it checks whether the second arg
        (which would be part of *args) is a dataframe. This flexible
        checking means this decorator can be used to validate methods
        within a class (which have self as their first argument). Class
        methods using this decorator must take a dataframe as their second
        argument (which in this method comes up in *args as args[0])

        Args:
            d (any): Any data type allowed, but all data types raise an
                error unless they are a pandas dataframe.
            *args: Any number of arguments. If d is NOT a pandas dataframe,
                first arg MUST be a pandas dataframe, or error raised.
            **kwargs: Keyword arguments for original function.

        Returns:
            function: Returns original function being decorated.

        Raises:
            TypeError: If one of d, args[0] not pandas dataframe.

        """
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
    """Check if dataframe contains all missing values or no missing values.

    This method acts as a decorator. It takes a function that takes data
    as its first argument and checks whether that data contains any
    missing or real values. This method leverages the check_data_structure
    decorator above to first verify that the data is a dataframe, and then
    verify that the data contains real and missing values. This decorator
    also works well with classes and utilizes the `functools.wrap`
    decorator to keep function names in tact.

    Args:
        func (function): The function that will be decorated.

    Returns:
        function: decorators return functions they wrap.
    """
    @functools.wraps(func)
    @check_data_structure
    def wrapper(d, *args, **kwargs):
        """Wrap function that checks a dataframe for missing and real values.

        This wrapper within the decorator does the actual verification. It
        checks that a dataframe has both missing and real values. If the
        dataframe is already complete, a warning is issued that there will
        be nothing to impute should an imputation method be used. If the
        dataframe is fully incomplete, an error is raised.

        Args:
            d (any): Again, anything can be passed, but dataframe required
                to continue without error from check_data_structure
            *args: A number of arguments. First must be dataframe if
                working within a class.
            **kwargs: Keyword arguments from original function.

        Returns:
            function: Returns original function being decorated.

        Raises:
            ValueError: If all values in dataframe are missing
        """
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

def remove_nan_columns(func):
    """Removes columns inplace in DataFrame if column all missing values.

    This method acts as a decorator. It first leverages the decorator
    `check_missingness` to verify that a proper DataFrame is passed. It
    then drops the columns with all rows missing from said DataFrame.
    Again, leverages `functools.wrap` to keep func names in place.

    Args:
        func (function): The function that will be decorated.

    Returns:
        function: decorators return functions they wrap.
    """
    @functools.wraps(func)
    @check_missingness
    def wrapper(d, *args, **kwargs):
        """Wrap function that removes columns w/ all rows missing.

        The wrapper leverages the `_nan_col_dropper` helper function,
        which removes columns from a DataFrame that have all rows missing.

        Args:
            d (pd.DataFrame): Dataframe to check, if not in args
            *args: Variable number of arguments, first of which may be df.
            **kwargs: Keyword arguments from original function.
        """
        if isinstance(d, pd.DataFrame):
            _nan_col_dropper(d)
        else:
            if args:
                _nan_col_dropper(args[0])
        return func(d, *args, **kwargs)
    return wrapper
